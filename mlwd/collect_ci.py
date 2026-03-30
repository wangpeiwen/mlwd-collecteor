"""
CI 估算：vLLM profiler 采集 kernel 时延 + 理论 FLOPs 计算。

Usage:
    python -m mlwd.collect_ci --model /data/Qwen/Qwen2.5-7B-Instruct
"""

import argparse, json, os, time, glob, gzip
from itertools import product
from .config import Experiment, OUTPUT_DIR, QWEN_HIDDEN, QWEN_LAYERS, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_INTER, V100_BW_GBS
from .classifier import classify, Cat


def attn_flops(b, s, max_tokens=32):
    h, d, nh, nkv, L = QWEN_HIDDEN, QWEN_HEAD_DIM, QWEN_HEADS, QWEN_KV_HEADS, QWEN_LAYERS
    # Prefill
    qkv = 2 * b * s * h * (h + 2 * nkv * d)
    qk = 2 * b * nh * s * s * d
    av = 2 * b * nh * s * s * d
    out = 2 * b * s * h * h
    prefill = (qkv + qk + av + out) * L
    # Decode (per step seq_len=1)
    decode_per_step = (2*b*1*h*(h+2*nkv*d) + 2*b*nh*1*1*d*2 + 2*b*1*h*h) * L
    return prefill + decode_per_step * max_tokens


def ffn_flops(b, s, max_tokens=32):
    h, inter, L = QWEN_HIDDEN, QWEN_INTER, QWEN_LAYERS
    prefill = 3 * 2 * b * s * h * inter * L
    decode_per_step = 3 * 2 * b * 1 * h * inter * L
    return prefill + decode_per_step * max_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Experiment.model)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--profile_dir", default="/tmp/vllm_profile")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "ci.json"))
    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"
    os.makedirs(args.profile_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from .runner import make_prompts

    print(f"Loading model: {args.model}...")
    llm = LLM(model=args.model, dtype="float16", trust_remote_code=True,
              enforce_eager=True,
              profiler_config={"profiler": "torch", "torch_profiler_dir": args.profile_dir})
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Model loaded.\n")

    llm.generate(["Hello"], SamplingParams(max_tokens=1, temperature=0))

    exp = Experiment(model=args.model)
    if args.batch_sizes: exp.batch_sizes = args.batch_sizes
    if args.seq_lengths: exp.seq_lengths = args.seq_lengths

    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    for b, s in product(exp.batch_sizes, exp.seq_lengths):
        key = f"b{b}_s{s}"
        if key in results and results[key].get("ci_ffn") is not None:
            print(f"[{key}] SKIP")
            continue

        print(f"\n=== {key} ===")
        prompts = make_prompts(tokenizer, s, b)
        sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)

        for _ in range(2):
            llm.generate(prompts, sp)

        # 清理旧 trace
        for f in glob.glob(os.path.join(args.profile_dir, "*")):
            os.remove(f)

        llm.start_profile()
        for i in range(args.num_runs):
            llm.generate(prompts, sp)
            print(f"  run {i} done")
        llm.stop_profile()
        time.sleep(2)

        # 解析 trace
        traces = glob.glob(os.path.join(args.profile_dir, "**/*.json.gz"), recursive=True)
        traces += glob.glob(os.path.join(args.profile_dir, "**/*.json"), recursive=True)

        attn_time_us = 0
        ffn_time_us = 0
        if traces:
            traces.sort(key=os.path.getmtime, reverse=True)
            tp = traces[0]
            print(f"  Parsing: {tp}")
            opener = gzip.open if tp.endswith(".gz") else open
            with opener(tp, "rt") as f:
                data = json.load(f)
            events = data if isinstance(data, list) else data.get("traceEvents", [])
            for evt in events:
                name = evt.get("name", "")
                dur = evt.get("dur", 0)
                if dur <= 0: continue
                cat = classify(name)
                if cat == Cat.ATTN: attn_time_us += dur
                elif cat == Cat.FFN: ffn_time_us += dur

        # 理论 FLOPs
        af = attn_flops(b, s, args.max_tokens)
        ff = ffn_flops(b, s, args.max_tokens)
        bw = V100_BW_GBS * 1e9  # bytes/s

        entry = {"batch_size": b, "seq_len": s,
                 "attn_flops": af, "ffn_flops": ff,
                 "attn_time_us": attn_time_us, "ffn_time_us": ffn_time_us}

        nr = args.num_runs
        if attn_time_us > 0:
            t_s = (attn_time_us / nr) * 1e-6
            entry["ci_attn"] = round(af / (t_s * bw), 4)
            entry["attn_tflops"] = round(af / t_s / 1e12, 2)
        if ffn_time_us > 0:
            t_s = (ffn_time_us / nr) * 1e-6
            entry["ci_ffn"] = round(ff / (t_s * bw), 4)
            entry["ffn_tflops"] = round(ff / t_s / 1e12, 2)

        print(f"  CI_attn={entry.get('ci_attn','-')}, CI_ffn={entry.get('ci_ffn','-')}")
        results[key] = entry

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
