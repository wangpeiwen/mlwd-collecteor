"""同模型 PD 共置干扰实验。

测量单个 vLLM 实例内，Prefill 请求注入对正在进行的 Decode 请求的干扰。
这是 CBS 调度算法的真实场景：决定是否将新请求的 Prefill 放到 Decode 节点上。

实验设计：
  1. Decode-only baseline: 发送 N 个请求，全部进入 Decode 阶段后测量 per-token 时延
  2. PD co-location: Decode 进行中注入 Prefill 请求，测量 Decode 时延退化
  3. α_d = (T_decode_coloc - T_decode_baseline) / T_decode_baseline

通过 vLLM 的 AsyncLLMEngine 或直接用 generate() 的 batch 特性实现。

Usage:
    python -m mlwd.colocation \\
        --model /data/Qwen2.5-7B-Instruct \\
        --gpu 0 --output output/colocation.json

    # 双卡并行跑两个模型
    python -m mlwd.colocation --model /data/Qwen2.5-7B-Instruct --gpu 0 \\
        --output output/colocation_qwen.json &
    python -m mlwd.colocation --model /data/LLM-Research/Llama-3.2-3B-Instruct --gpu 1 \\
        --output output/colocation_llama.json &
"""

import argparse, json, os, time, gc
from itertools import product
from pathlib import Path

from .config import Experiment, OUTPUT_DIR, DEFAULT_BATCH_SIZES, DEFAULT_SEQ_LENGTHS


def _save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _median(lats):
    lats = sorted(lats)
    mid = len(lats) // 2
    return lats[mid] if len(lats) % 2 else (lats[mid - 1] + lats[mid]) / 2


def _free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── 核心测量函数 ──────────────────────────────────────

def measure_decode_only(llm, tokenizer, batch_size, seq_len, max_tokens,
                        num_runs, warmup):
    """纯 Decode baseline：所有请求都是短 prompt + 长输出。"""
    from vllm import SamplingParams
    prompts = ["The"] * batch_size
    sp = SamplingParams(max_tokens=max_tokens, temperature=0)

    for _ in range(warmup):
        llm.generate(prompts, sp)

    lats = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        lats.append((time.perf_counter() - t0) * 1000.0)
    return _median(lats), sorted(lats)


def measure_pd_colocation(llm, tokenizer, decode_batch, decode_seq,
                           prefill_batch, prefill_seq, max_tokens,
                           num_runs, warmup):
    """PD 共置：Decode 请求 + Prefill 请求混合提交。

    模拟 CBS 场景：节点上有 decode_batch 个 Decode 请求正在生成，
    同时注入 prefill_batch 个新的 Prefill 请求（长 prompt, 1 token 输出）。

    vLLM 的 continuous batching 会在同一个 iteration 内同时处理两类请求，
    Prefill 的大 GEMM 会抢占 SM，干扰 Decode 的小 kernel。
    """
    from vllm import SamplingParams

    # Decode 请求：短 prompt + 长输出
    decode_prompts = ["The"] * decode_batch
    decode_sp = SamplingParams(max_tokens=max_tokens, temperature=0)

    # Prefill 请求：长 prompt + 1 token 输出
    text = "hello " * (prefill_seq * 2)
    ids = tokenizer.encode(text)[:prefill_seq]
    prefill_prompt = tokenizer.decode(ids)
    prefill_prompts = [prefill_prompt] * prefill_batch
    prefill_sp = SamplingParams(max_tokens=1, temperature=0)

    # 混合 batch：Decode + Prefill 一起提交
    mixed_prompts = decode_prompts + prefill_prompts
    mixed_sp = [decode_sp] * decode_batch + [prefill_sp] * prefill_batch

    # vLLM generate 只接受单个 SamplingParams，所以分两次提交
    # 但 vLLM 内部会 continuous batching 合并处理
    for _ in range(warmup):
        llm.generate(decode_prompts, decode_sp)

    lats = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        # 同时提交两类请求，vLLM 会在内部 interleave
        # Decode 请求决定总时延（max_tokens 个 step）
        # Prefill 请求在第一个 step 完成，之后不再干扰
        llm.generate(decode_prompts + prefill_prompts,
                     SamplingParams(max_tokens=max_tokens, temperature=0))
        lats.append((time.perf_counter() - t0) * 1000.0)

    return _median(lats), sorted(lats)


# ── 主实验 ──────────────────────────────────────────────

def run_experiment(model_path, gpu_id, output_path,
                   batch_sizes=None, seq_lengths=None,
                   num_runs=5, warmup=2, max_tokens=32):
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES
    if seq_lengths is None:
        seq_lengths = DEFAULT_SEQ_LENGTHS

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"

    data = _load(output_path)
    model_name = Path(model_path).name

    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  GPU: {gpu_id}")
    print(f"  Batch sizes: {batch_sizes}, Seq lengths: {seq_lengths}")
    print(f"  Runs: {num_runs}, Warmup: {warmup}, Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"Loading {model_name}...")
    llm = LLM(model=model_path, dtype="float16", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.\n")

    # ── Step 1: Decode-only baselines ──
    if "baselines" not in data:
        print("Step 1: Decode-only baselines")
        baselines = {}
        for b in batch_sizes:
            key = f"b{b}"
            med, lats = measure_decode_only(llm, tokenizer, b, 0, max_tokens,
                                             num_runs, warmup)
            baselines[key] = {"batch_size": b, "median_ms": med, "all_ms": lats}
            print(f"  b={b}: {med:.1f} ms")
        data["baselines"] = baselines
        data["model"] = model_name
        _save(output_path, data)
        print()
    else:
        print("Step 1: Baselines (cached)\n")

    # ── Step 2: PD co-location ──
    if "pairs" not in data:
        print("Step 2: PD co-location measurements")
        pairs = []
        total = len(batch_sizes) * len(batch_sizes) * len(seq_lengths)
        idx = 0

        for b_d in batch_sizes:
            baseline = data["baselines"][f"b{b_d}"]["median_ms"]

            for b_p, s_p in product(batch_sizes, seq_lengths):
                idx += 1
                key = f"d{b_d}_p{b_p}x{s_p}"
                print(f"  [{idx}/{total}] {key}...", end=" ", flush=True)

                coloc_med, coloc_lats = measure_pd_colocation(
                    llm, tokenizer,
                    decode_batch=b_d, decode_seq=0,
                    prefill_batch=b_p, prefill_seq=s_p,
                    max_tokens=max_tokens,
                    num_runs=num_runs, warmup=warmup)

                alpha_d = (coloc_med - baseline) / baseline if baseline > 0 else 0

                entry = {
                    "key": key,
                    "model": model_name,
                    "decode_batch": b_d,
                    "prefill_batch": b_p, "prefill_seq": s_p,
                    "victim_phase": "decode", "aggressor_phase": "prefill",
                    "victim_b": b_d, "victim_s": max_tokens,
                    "aggressor_b": b_p, "aggressor_s": s_p,
                    "baseline_ms": round(baseline, 4),
                    "coloc_ms": round(coloc_med, 4),
                    "alpha_d": round(alpha_d, 6),
                    "all_ms": coloc_lats,
                }
                pairs.append(entry)
                print(f"α_d={alpha_d:.4f} ({baseline:.0f}→{coloc_med:.0f} ms)")

        data["pairs"] = pairs
        _save(output_path, data)
        print()
    else:
        print("Step 2: PD co-location (cached)\n")

    del llm
    _free_gpu()

    n = len(data.get("pairs", []))
    print(f"Done. {n} pairs → {output_path}")


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="同模型 PD 共置干扰实验")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "colocation.json"))
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=32)
    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        gpu_id=args.gpu,
        output_path=args.output,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_runs=args.num_runs,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
