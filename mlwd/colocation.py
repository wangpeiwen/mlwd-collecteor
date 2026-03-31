"""共置干扰实验 v3：单进程双 vLLM 引擎方案。

设计原则：
  - 同一进程中加载两个 vLLM LLM() 实例，共享 GPU 显存
  - 先加载小模型 (aggressor, util=0.28)，再加载大模型 (victim, util=0.85)
  - vLLM 第二个实例会基于剩余显存计算 KV cache pool，不会 OOM
  - Aggressor 在后台线程持续调用 llm.generate() 制造真实推理干扰

显存预算 (V100-32GB):
  Phase 1: vLLM victim only (util=0.90) → baseline 测量
  Phase 2: vLLM aggressor (Llama 3B, util=0.28 ~9GB)
         + vLLM victim (Qwen 7B, util=0.85 ~22GB 剩余空间) → 共置测量
  Phase 3: vLLM aggressor only (util=0.90) → baseline 测量

Usage:
    python -m mlwd.colocation \\
        --victim /data/Qwen2.5-7B-Instruct \\
        --aggressor /data/LLM-Research/Llama-3.2-3B-Instruct \\
        --gpu 0 --output output/colocation.json
"""

import argparse, json, os, time, threading, gc
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
        torch.cuda.synchronize()


# ── Phase 1 & 3: 单模型 baseline ──────────────────────

def measure_baselines(model_path, batch_sizes, seq_lengths, max_tokens,
                      num_runs, warmup, mode="decode"):
    """单模型 vLLM baseline，独占 GPU。"""
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"  Loading {Path(model_path).name} (vLLM, util=0.90)...")
    llm = LLM(model=model_path, dtype="float16", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    results = {}
    for b, s in product(batch_sizes, seq_lengths):
        key = f"b{b}_s{s}"
        if mode == "decode":
            prompts = ["The"] * b
            sp = SamplingParams(max_tokens=max_tokens, temperature=0)
        else:
            text = "hello " * (s * 2)
            ids = tokenizer.encode(text)[:s]
            prompt = tokenizer.decode(ids)
            prompts = [prompt] * b
            sp = SamplingParams(max_tokens=1, temperature=0)

        for _ in range(warmup):
            llm.generate(prompts, sp)

        lats = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            llm.generate(prompts, sp)
            lats.append((time.perf_counter() - t0) * 1000.0)

        med = _median(lats)
        results[key] = {"median_ms": med, "all_ms": sorted(lats)}
        print(f"    {key} ({mode}): {med:.1f} ms")

    del llm
    _free_gpu()
    print("  Model unloaded.\n")
    return results


# ── Phase 2: 双 vLLM 共置 ──────────────────────────────

class VllmAggressor:
    """后台线程持续调用 vLLM aggressor 推理，制造真实干扰。"""

    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self._stop = threading.Event()
        self._thread = None

    def start(self, batch_size, seq_len):
        from vllm import SamplingParams
        text = "hello " * (seq_len * 2)
        ids = self.tokenizer.encode(text)[:seq_len]
        prompt = self.tokenizer.decode(ids)
        prompts = [prompt] * batch_size
        sp = SamplingParams(max_tokens=1, temperature=0)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(prompts, sp), daemon=True)
        self._thread.start()

    def _loop(self, prompts, sp):
        while not self._stop.is_set():
            try:
                self.llm.generate(prompts, sp)
            except Exception:
                break

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=60)
            self._thread = None


def measure_colocation(victim_model_path, aggressor_model_path,
                       batch_sizes, seq_lengths, max_tokens,
                       num_runs, warmup,
                       aggressor_util=0.28, victim_util=0.85):
    """Phase 2: 同进程加载两个 vLLM 引擎，先小后大。"""
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # 先加载小模型 (aggressor)
    print(f"  Loading aggressor {Path(aggressor_model_path).name} (vLLM, util={aggressor_util})...")
    llm_a = LLM(model=aggressor_model_path, dtype="float16", trust_remote_code=True,
                enforce_eager=True, gpu_memory_utilization=aggressor_util)
    tok_a = AutoTokenizer.from_pretrained(aggressor_model_path, trust_remote_code=True)

    # 再加载大模型 (victim)，vLLM 会基于剩余显存分配 KV cache
    print(f"  Loading victim {Path(victim_model_path).name} (vLLM, util={victim_util})...")
    llm_v = LLM(model=victim_model_path, dtype="float16", trust_remote_code=True,
                enforce_eager=True, gpu_memory_utilization=victim_util)

    aggressor = VllmAggressor(llm_a, tok_a)
    print(f"  Both models loaded. Starting co-location...\n")

    results = {}
    total = len(batch_sizes) ** 2 * len(seq_lengths) ** 2
    idx = 0

    for b_v, s_v in product(batch_sizes, seq_lengths):
        v_prompts = ["The"] * b_v
        v_sp = SamplingParams(max_tokens=max_tokens, temperature=0)

        for b_a, s_a in product(batch_sizes, seq_lengths):
            idx += 1
            key = f"v{b_v}x{s_v}_a{b_a}x{s_a}"
            print(f"  [{idx}/{total}] {key}...", end=" ", flush=True)

            # 启动 aggressor 后台推理
            aggressor.start(b_a, s_a)
            time.sleep(0.5)

            # warmup victim
            for _ in range(warmup):
                llm_v.generate(v_prompts, v_sp)

            # 测量
            lats = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                llm_v.generate(v_prompts, v_sp)
                lats.append((time.perf_counter() - t0) * 1000.0)

            aggressor.stop()

            med = _median(lats)
            results[key] = {
                "victim_b": b_v, "victim_s": s_v,
                "aggressor_b": b_a, "aggressor_s": s_a,
                "victim_coloc_ms": med, "all_ms": sorted(lats),
            }
            print(f"{med:.1f} ms")

    del llm_v, llm_a
    aggressor.stop()
    _free_gpu()
    print("  Both models unloaded.\n")
    return results


# ── 主流程 ──────────────────────────────────────────────

def run_experiment(victim_model, aggressor_model, gpu_id, output_path,
                   batch_sizes=None, seq_lengths=None,
                   num_runs=5, warmup=2, max_tokens=32):
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES
    if seq_lengths is None:
        seq_lengths = DEFAULT_SEQ_LENGTHS

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    data = _load(output_path)

    v_name = Path(victim_model).name
    a_name = Path(aggressor_model).name
    n_configs = len(batch_sizes) * len(seq_lengths)

    print(f"{'='*60}")
    print(f"  Victim:     {v_name} (decode)")
    print(f"  Aggressor:  {a_name} (prefill)")
    print(f"  GPU: {gpu_id}, configs: {n_configs}, pairs: {n_configs**2}")
    print(f"  Runs: {num_runs}, Warmup: {warmup}, Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    # Phase 1: Victim baseline (独占 GPU)
    if "victim_baselines" not in data:
        print("Phase 1/3: Victim baseline (vLLM, full GPU)")
        data["victim_baselines"] = measure_baselines(
            victim_model, batch_sizes, seq_lengths, max_tokens,
            num_runs, warmup, mode="decode")
        data["victim_model"] = v_name
        _save(output_path, data)
    else:
        print("Phase 1/3: Victim baseline (cached)\n")

    # Phase 2: Co-location (双 vLLM 引擎)
    if "colocation" not in data:
        print("Phase 2/3: Co-location (dual vLLM engines)")
        data["colocation"] = measure_colocation(
            victim_model, aggressor_model,
            batch_sizes, seq_lengths, max_tokens,
            num_runs, warmup)
        _save(output_path, data)
    else:
        print("Phase 2/3: Co-location (cached)\n")

    # Phase 3: Aggressor baseline (独占 GPU)
    if "aggressor_baselines" not in data:
        print("Phase 3/3: Aggressor baseline (vLLM, full GPU)")
        data["aggressor_baselines"] = measure_baselines(
            aggressor_model, batch_sizes, seq_lengths, max_tokens=1,
            num_runs=num_runs, warmup=warmup, mode="prefill")
        data["aggressor_model"] = a_name
        _save(output_path, data)
    else:
        print("Phase 3/3: Aggressor baseline (cached)\n")

    # 计算 α_d
    print("Computing interference coefficients...")
    pairs = []
    for key, coloc in data["colocation"].items():
        b_v, s_v = coloc["victim_b"], coloc["victim_s"]
        b_a, s_a = coloc["aggressor_b"], coloc["aggressor_s"]

        bl_v = data["victim_baselines"].get(f"b{b_v}_s{s_v}", {}).get("median_ms")
        coloc_v = coloc["victim_coloc_ms"]

        alpha_d = (coloc_v - bl_v) / bl_v if bl_v and bl_v > 0 else None

        pairs.append({
            "key": key,
            "victim_model": v_name, "aggressor_model": a_name,
            "victim_b": b_v, "victim_s": s_v, "victim_phase": "decode",
            "aggressor_b": b_a, "aggressor_s": s_a, "aggressor_phase": "prefill",
            "victim_baseline_ms": round(bl_v, 4) if bl_v else None,
            "victim_coloc_ms": round(coloc_v, 4),
            "alpha_d": round(alpha_d, 6) if alpha_d is not None else None,
        })
        if alpha_d is not None:
            print(f"  {key}: α_d={alpha_d:.4f} ({bl_v:.0f}→{coloc_v:.0f} ms)")

    data["pairs"] = pairs
    _save(output_path, data)
    n_valid = sum(1 for p in pairs if p.get("alpha_d") is not None)
    print(f"\nDone. {n_valid} valid pairs → {output_path}")


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="共置干扰实验 (双 vLLM 引擎)")
    parser.add_argument("--victim", required=True, help="Victim 模型路径")
    parser.add_argument("--aggressor", required=True, help="Aggressor 模型路径")
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
    main()
