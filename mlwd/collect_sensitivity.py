"""
干扰敏感度采集。每个维度采集完立即写入 JSON。

Usage:
    python -m mlwd.collect_sensitivity --model /data/Qwen/Qwen2.5-7B-Instruct
    python -m mlwd.collect_sensitivity --batch_sizes 1 4 --seq_lengths 32 64 128
"""

import argparse, json, os, time, threading
from .config import Experiment, OUTPUT_DIR
from .runner import load_model, make_prompts, run_inference
from .stress import StressKernels

DIMS = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]


def _save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _measure(run_fn, num_runs):
    lats = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        run_fn()
        lats.append((time.perf_counter() - t0) * 1000.0)
    lats.sort()
    mid = len(lats) // 2
    return (lats[mid] if len(lats) % 2 else (lats[mid-1]+lats[mid])/2), lats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Experiment.model)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--lib", default=None)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "sensitivity.json"))
    args = parser.parse_args()

    exp = Experiment(model=args.model)
    if args.batch_sizes: exp.batch_sizes = args.batch_sizes
    if args.seq_lengths: exp.seq_lengths = args.seq_lengths

    sk = StressKernels(args.lib)
    results = _load(args.output)

    print(f"Loading model: {exp.model}...")
    llm, tokenizer = load_model(exp.model)
    print("Model loaded.\n")

    from vllm import SamplingParams
    cfg = exp.stress

    stress_fns = {
        "sigma_bs": lambda: sk.bs(cfg),
        "sigma_cu": lambda: sk.cu(cfg),
        "sigma_l2": lambda: sk.l2(cfg),
        "sigma_bw": lambda: sk.bw(cfg),
    }

    for b, s, phase in exp.iter_points():
        key = f"b{b}_s{s}_{phase}"
        existing = results.get(key, {})

        # 构造推理函数
        if phase == "prefill":
            prompts = make_prompts(tokenizer, s, b)
            sp = SamplingParams(max_tokens=1, temperature=0)
        else:
            prompts = ["The"] * b
            sp = SamplingParams(max_tokens=s, temperature=0)
        run_fn = lambda _p=prompts, _sp=sp: llm.generate(_p, _sp)

        print(f"\n=== {key} ===")

        # 基线
        if "baseline_ms" not in existing:
            for _ in range(exp.warmup_runs):
                run_fn()
            baseline, all_lats = _measure(run_fn, exp.num_runs)
            existing.update({"batch_size": b, "seq_len": s, "phase": phase,
                             "baseline_ms": round(baseline, 4),
                             "baseline_all_ms": [round(l, 4) for l in all_lats]})
            results[key] = existing
            _save(args.output, results)
            print(f"  baseline: {baseline:.2f} ms")
        else:
            baseline = existing["baseline_ms"]
            print(f"  baseline (cached): {baseline:.2f} ms")

        # 逐维度
        for dim in DIMS:
            if dim in existing and existing[dim] is not None:
                print(f"  {dim}: {existing[dim]:.4f} (cached)")
                continue
            print(f"  measuring {dim}...")
            try:
                stop = threading.Event()
                def bg(_fn=stress_fns[dim], _stop=stop):
                    while not _stop.is_set(): _fn()
                t = threading.Thread(target=bg, daemon=True)
                t.start()
                for _ in range(exp.warmup_runs):
                    run_fn()
                stressed, all_lats = _measure(run_fn, exp.num_runs)
                stop.set(); t.join(timeout=30)
                sigma = max((stressed - baseline) / baseline, 0.0)
                existing[dim] = round(sigma, 6)
                existing[f"{dim}_stressed_ms"] = round(stressed, 4)
                existing[f"{dim}_all_ms"] = [round(l, 4) for l in all_lats]
                print(f"  {dim} = {sigma:.4f} (stressed={stressed:.2f}ms)")
            except Exception as e:
                print(f"  {dim} FAILED: {e}")
                existing[dim] = None
                existing[f"{dim}_error"] = str(e)
            results[key] = existing
            _save(args.output, results)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
