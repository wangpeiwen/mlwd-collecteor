"""共置干扰实验：两模型同卡运行，测量真实 α_p 和 α_d。

需要 NVIDIA MPS (Multi-Process Service) 支持同卡多进程共享 GPU。
实验流程：
  1. 启动 MPS daemon
  2. 进程 A 运行 victim (decode)，进程 B 运行 aggressor (prefill)
  3. 分别测量单独运行和共置运行的时延
  4. 计算 α_d = (T_coloc - T_baseline) / T_baseline

Usage:
    # 先启动 MPS
    nvidia-cuda-mps-control -d

    # 运行共置实验
    python -m mlwd.colocation \\
        --victim /data/Qwen/Qwen2.5-7B-Instruct \\
        --aggressor /data/Llama-3.2-3B \\
        --gpu 0 --output output/colocation.json

    # 双卡并行（卡0和卡1各跑一组，角色互换）
    python -m mlwd.colocation \\
        --victim /data/Qwen/Qwen2.5-7B-Instruct \\
        --aggressor /data/Llama-3.2-3B \\
        --gpu 0 --output output/colocation_0.json &
    python -m mlwd.colocation \\
        --victim /data/Llama-3.2-3B \\
        --aggressor /data/Qwen/Qwen2.5-7B-Instruct \\
        --gpu 1 --output output/colocation_1.json &
"""

import argparse, json, os, time, sys
import multiprocessing as mp
from itertools import product
from pathlib import Path

from .config import Experiment, OUTPUT_DIR, DEFAULT_BATCH_SIZES, DEFAULT_SEQ_LENGTHS


# ── 子进程工作函数 ──────────────────────────────────────

def _worker_run(model_path, prompts_data, max_tokens, num_runs, warmup,
                gpu_id, result_queue, barrier, role, max_model_len=2048,
                gpu_mem_util=0.45):
    """在子进程中加载模型并运行推理。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(model=model_path, dtype="float16", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=gpu_mem_util,
              max_model_len=max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sp = SamplingParams(max_tokens=max_tokens, temperature=0)
    prompts = prompts_data

    # warmup
    for _ in range(warmup):
        llm.generate(prompts, sp)

    # 同步：等待两个进程都 warmup 完毕
    barrier.wait()

    # 测量
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    mid = len(latencies) // 2
    median = latencies[mid] if len(latencies) % 2 else (latencies[mid-1] + latencies[mid]) / 2

    result_queue.put({
        "role": role,
        "median_ms": median,
        "all_ms": latencies,
    })


def _make_prompts(model_path, seq_len, batch_size):
    """构造 prompts（在主进程中执行，避免子进程重复加载 tokenizer）。"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = "hello " * (seq_len * 2)
    ids = tokenizer.encode(text)[:seq_len]
    prompt = tokenizer.decode(ids)
    return [prompt] * batch_size


# ── 单独运行基线 ──────────────────────────────────────

def measure_baseline(model_path, prompts, max_tokens, num_runs, warmup, gpu_id,
                     max_model_len=2048):
    """单模型单独运行，测量基线时延。"""
    result_queue = mp.Queue()
    barrier = mp.Barrier(1)

    p = mp.Process(target=_worker_run,
                   args=(model_path, prompts, max_tokens, num_runs, warmup,
                         gpu_id, result_queue, barrier, "baseline",
                         max_model_len, 0.90))
    p.start()
    p.join(timeout=600)
    if p.is_alive():
        p.terminate()
        return None
    if result_queue.empty():
        return None
    return result_queue.get()


# ── 共置运行 ──────────────────────────────────────

def measure_colocation(victim_model, victim_prompts, victim_max_tokens,
                       aggressor_model, aggressor_prompts, aggressor_max_tokens,
                       num_runs, warmup, gpu_id, max_model_len=2048):
    """两模型同卡共置运行。"""
    result_queue = mp.Queue()
    barrier = mp.Barrier(2)

    p_victim = mp.Process(
        target=_worker_run,
        args=(victim_model, victim_prompts, victim_max_tokens,
              num_runs, warmup, gpu_id, result_queue, barrier, "victim",
              max_model_len, 0.45))
    p_aggressor = mp.Process(
        target=_worker_run,
        args=(aggressor_model, aggressor_prompts, aggressor_max_tokens,
              num_runs, warmup, gpu_id, result_queue, barrier, "aggressor",
              max_model_len, 0.45))

    p_victim.start()
    p_aggressor.start()
    p_victim.join(timeout=600)
    p_aggressor.join(timeout=600)

    for p in [p_victim, p_aggressor]:
        if p.is_alive():
            p.terminate()

    results = {}
    while not result_queue.empty():
        r = result_queue.get()
        results[r["role"]] = r
    return results


# ── 保存/加载 ──────────────────────────────────────

def _save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


# ── 主实验循环 ──────────────────────────────────────

def run_experiment(victim_model, aggressor_model, gpu_id, output_path,
                   batch_sizes=None, seq_lengths=None,
                   num_runs=5, warmup=2, max_tokens=32):
    """运行完整共置实验矩阵。

    对每个 (b_v, s_v, b_a, s_a) 组合：
    - victim 运行 decode（max_tokens 个 token）
    - aggressor 运行 prefill（max_tokens=1）
    - 测量 baseline 和 colocation 时延
    - 计算 α_d 和 α_p
    """
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES
    if seq_lengths is None:
        seq_lengths = DEFAULT_SEQ_LENGTHS

    results = _load(output_path)
    done_keys = {r["key"] for r in results}

    # 预构造 prompts
    print("Preparing prompts...")
    victim_prompts_cache = {}
    aggressor_prompts_cache = {}
    for b, s in product(batch_sizes, seq_lengths):
        victim_prompts_cache[(b, s)] = ["The"] * b  # decode: 短 prompt + 长输出
        aggressor_prompts_cache[(b, s)] = _make_prompts(aggressor_model, s, b)

    total = len(batch_sizes) * len(seq_lengths)
    total_pairs = total * total
    print(f"Experiment matrix: {total} victim configs × {total} aggressor configs = {total_pairs} pairs")
    print(f"GPU: {gpu_id}, num_runs: {num_runs}\n")

    idx = 0
    for b_v, s_v in product(batch_sizes, seq_lengths):
        # 先测 victim baseline（decode）
        bl_key = f"baseline_victim_b{b_v}_s{s_v}"
        baseline_victim = None
        for r in results:
            if r.get("key") == bl_key:
                baseline_victim = r.get("victim_baseline_ms")
                break

        if baseline_victim is None:
            print(f"[Baseline] victim decode b={b_v},s={s_v}...")
            v_prompts = victim_prompts_cache[(b_v, s_v)]
            bl = measure_baseline(victim_model, v_prompts, max_tokens,
                                  num_runs, warmup, gpu_id)
            if bl:
                baseline_victim = bl["median_ms"]
                results.append({"key": bl_key, "victim_baseline_ms": baseline_victim,
                                "all_ms": bl["all_ms"]})
                _save(output_path, results)
                print(f"  baseline = {baseline_victim:.2f} ms")
            else:
                print(f"  FAILED, skipping")
                continue

        for b_a, s_a in product(batch_sizes, seq_lengths):
            idx += 1
            key = f"coloc_v{b_v}x{s_v}_a{b_a}x{s_a}"
            if key in done_keys:
                print(f"[{idx}/{total_pairs}] {key} SKIP (cached)")
                continue

            print(f"[{idx}/{total_pairs}] {key}...")

            # 测 aggressor baseline（prefill）
            a_prompts = aggressor_prompts_cache[(b_a, s_a)]
            bl_a = measure_baseline(aggressor_model, a_prompts, 1,
                                    num_runs, warmup, gpu_id)
            baseline_aggressor = bl_a["median_ms"] if bl_a else None

            # 共置运行
            v_prompts = victim_prompts_cache[(b_v, s_v)]
            coloc = measure_colocation(
                victim_model, v_prompts, max_tokens,
                aggressor_model, a_prompts, 1,
                num_runs, warmup, gpu_id)

            if "victim" not in coloc or "aggressor" not in coloc:
                print(f"  FAILED: incomplete results")
                continue

            victim_coloc = coloc["victim"]["median_ms"]
            aggressor_coloc = coloc["aggressor"]["median_ms"]

            alpha_d = (victim_coloc - baseline_victim) / baseline_victim if baseline_victim > 0 else 0
            alpha_p = ((aggressor_coloc - baseline_aggressor) / baseline_aggressor
                       if baseline_aggressor and baseline_aggressor > 0 else 0)

            entry = {
                "key": key,
                "victim_model": Path(victim_model).name,
                "aggressor_model": Path(aggressor_model).name,
                "victim_b": b_v, "victim_s": s_v, "victim_phase": "decode",
                "aggressor_b": b_a, "aggressor_s": s_a, "aggressor_phase": "prefill",
                "victim_baseline_ms": round(baseline_victim, 4),
                "victim_coloc_ms": round(victim_coloc, 4),
                "aggressor_baseline_ms": round(baseline_aggressor, 4) if baseline_aggressor else None,
                "aggressor_coloc_ms": round(aggressor_coloc, 4),
                "alpha_d": round(alpha_d, 6),
                "alpha_p": round(alpha_p, 6),
            }
            results.append(entry)
            done_keys.add(key)
            _save(output_path, results)
            print(f"  α_d={alpha_d:.4f}  α_p={alpha_p:.4f}  "
                  f"(victim: {baseline_victim:.0f}→{victim_coloc:.0f}ms, "
                  f"aggressor: {baseline_aggressor:.0f}→{aggressor_coloc:.0f}ms)")

    print(f"\nDone. {len([r for r in results if r.get('alpha_d') is not None])} pairs saved to {output_path}")


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="共置干扰实验")
    parser.add_argument("--victim", required=True, help="Victim 模型路径 (decode)")
    parser.add_argument("--aggressor", required=True, help="Aggressor 模型路径 (prefill)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "colocation.json"))
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=32)
    args = parser.parse_args()

    run_experiment(
        victim_model=args.victim,
        aggressor_model=args.aggressor,
        gpu_id=args.gpu,
        output_path=args.output,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_runs=args.num_runs,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
