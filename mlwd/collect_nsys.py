"""
nsys 采集 + 解析。两种模式：

  --profile: 作为 nsys 包装的推理脚本（模型加载一次，NVTX 标记）
  --parse:   解析 nsys SQLite → output/nsys.json

Usage:
    # Profile（被 nsys 包装）
    nsys profile -o /tmp/trace --trace cuda,nvtx --trace-fork-before-exec=true \
      --sample none --cpuctxsw none --force-overwrite true \
      python -m mlwd.collect_nsys --profile --model /data/Qwen/Qwen2.5-7B-Instruct

    # 导出 + 解析
    nsys export --type sqlite --output /tmp/trace.sqlite --force-overwrite true /tmp/trace.nsys-rep
    python -m mlwd.collect_nsys --parse /tmp/trace.sqlite --key b1_s32
"""

import argparse, json, os, time, sqlite3, statistics
from itertools import product
from .config import Experiment, OUTPUT_DIR
from .classifier import classify, Cat


def _profile_mode(args):
    """模型加载一次，循环跑所有实验点，NVTX 标记。"""
    import torch
    from .runner import load_model, make_prompts
    from vllm import SamplingParams

    exp = Experiment(model=args.model)
    if args.batch_sizes: exp.batch_sizes = args.batch_sizes
    if args.seq_lengths: exp.seq_lengths = args.seq_lengths

    print(f"Loading model: {exp.model}...")
    llm, tokenizer = load_model(exp.model)
    print("Model loaded.\n")

    llm.generate(["Hello"], SamplingParams(max_tokens=1, temperature=0))
    torch.cuda.synchronize()

    meta = {}
    for b, s in product(exp.batch_sizes, exp.seq_lengths):
        key = f"b{b}_s{s}"
        prompts = make_prompts(tokenizer, s, b)
        sp = SamplingParams(max_tokens=exp.max_tokens, temperature=0)

        # warmup
        for _ in range(exp.warmup_runs):
            llm.generate(prompts, sp)
        torch.cuda.synchronize()
        time.sleep(1.0)
        torch.cuda.synchronize()

        lats = []
        for i in range(exp.num_runs):
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"{key}_run{i}")
            t0 = time.perf_counter()
            llm.generate(prompts, sp)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) * 1000.0
            torch.cuda.nvtx.range_pop()
            lats.append(lat)
            print(f"  {key} run {i}: {lat:.2f} ms")

        torch.cuda.synchronize()
        time.sleep(1.0)
        torch.cuda.synchronize()

        lats.sort()
        mid = len(lats) // 2
        median = lats[mid] if len(lats) % 2 else (lats[mid-1]+lats[mid])/2
        meta[key] = {"batch_size": b, "seq_len": s, "median_ms": round(median, 4),
                      "all_ms": [round(l, 4) for l in lats]}
        print(f"  {key} median: {median:.2f} ms\n")

    if args.output_meta:
        with open(args.output_meta, "w") as f:
            json.dump(meta, f, indent=2)
    print("Profiling complete.")


def _parse_mode(args):
    """解析 nsys SQLite，提取执行模式特征。"""
    conn = sqlite3.connect(args.parse)
    conn.row_factory = sqlite3.Row

    # 读取 kernel
    try:
        kernels = conn.execute("""
            SELECT s.value as name, k.start, k.end, (k.end - k.start) as duration_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start ASC
        """).fetchall()
    except sqlite3.OperationalError:
        print("[NSYS] No kernel table found")
        conn.close()
        return

    conn.close()
    print(f"[NSYS] {len(kernels)} kernels")

    # 跳过前 20% (模型加载 + warmup)
    skip = len(kernels) // 5
    kernels = kernels[skip:]

    # 计算特征
    attn_dur, ffn_dur, all_dur, cats = [], [], [], []
    for k in kernels:
        dur = k["duration_ns"] / 1000.0
        cat = classify(k["name"])
        all_dur.append(dur)
        cats.append(cat)
        if cat == Cat.ATTN: attn_dur.append(dur)
        elif cat == Cat.FFN: ffn_dur.append(dur)

    entry = {}
    if attn_dur:
        entry["t_attn"] = round(statistics.mean(attn_dur), 4)
        entry["t_attn_std"] = round(statistics.stdev(attn_dur), 4) if len(attn_dur) > 1 else 0.0
    if ffn_dur:
        entry["t_ffn"] = round(statistics.mean(ffn_dur), 4)
        entry["t_ffn_std"] = round(statistics.stdev(ffn_dur), 4) if len(ffn_dur) > 1 else 0.0
    if len(kernels) > 1:
        intervals = [(kernels[i]["start"] - kernels[i-1]["start"]) / 1000.0
                     for i in range(1, len(kernels))]
        entry["g_launch"] = round(statistics.mean([iv for iv in intervals if iv > 0]), 4)
    total = sum(all_dur)
    if total > 0:
        entry["r_attn"] = round(sum(attn_dur) / total, 6)
        entry["r_ffn"] = round(sum(ffn_dur) / total, 6)
    if len(kernels) > 1:
        dur_s = (kernels[-1]["end"] - kernels[0]["start"]) / 1e9
        if dur_s > 0:
            switches = 0
            prev = None
            for c in cats:
                if c == Cat.OTHER: continue
                is_compute = (c == Cat.FFN)
                if prev is not None and is_compute != prev: switches += 1
                prev = is_compute
            entry["f_switch"] = round(switches / dur_s, 4)
    entry["num_kernels"] = len(kernels)
    entry["num_attn"] = len(attn_dur)
    entry["num_ffn"] = len(ffn_dur)

    key = args.key or "all"
    print(f"  {key}: {len(kernels)} kernels, {len(attn_dur)} attn, {len(ffn_dur)} ffn")

    # 保存
    out_path = args.output or str(OUTPUT_DIR / "nsys.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing[key] = entry
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Profile 模式")
    parser.add_argument("--parse", type=str, help="解析 nsys SQLite 文件")
    parser.add_argument("--model", default=Experiment.model)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_meta", type=str, default=None)
    args = parser.parse_args()

    if args.profile:
        _profile_mode(args)
    elif args.parse:
        _parse_mode(args)
    else:
        parser.error("指定 --profile 或 --parse")


if __name__ == "__main__":
    main()
