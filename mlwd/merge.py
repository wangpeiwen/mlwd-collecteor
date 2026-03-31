"""合并 sensitivity + nsys + ci → mlwd_complete.json"""

import argparse, json, os
from pathlib import Path
from .config import DEFAULT_BATCH_SIZES, DEFAULT_SEQ_LENGTHS, Experiment, get_model_params
from .estimate_missing import patch_entry

NSYS_FIELDS = ["t_attn", "t_attn_std", "t_ffn", "t_ffn_std",
               "g_launch", "r_attn", "r_ffn", "f_switch"]


def _load(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="output", help="数据目录")
    parser.add_argument("--model", default=Experiment.model, help="模型路径或标识")
    args = parser.parse_args()
    d = Path(args.dir)

    sens = _load(str(d / "sensitivity.json"))
    nsys = _load(str(d / "nsys.json"))
    ci = _load(str(d / "ci.json"))

    print(f"Loaded: sensitivity={len(sens)}, nsys={len(nsys)}, ci={len(ci)}")

    complete = {}
    for b in DEFAULT_BATCH_SIZES:
        for s in DEFAULT_SEQ_LENGTHS:
            for phase in ["prefill", "decode"]:
                key = f"b{b}_s{s}_{phase}"
                entry = {"batch_size": b, "seq_len": s, "phase": phase}

                # sensitivity
                if key in sens:
                    for f in ["baseline_ms", "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw",
                              "sigma_bs_stressed_ms", "sigma_cu_stressed_ms",
                              "sigma_l2_stressed_ms", "sigma_bw_stressed_ms"]:
                        if f in sens[key]: entry[f] = sens[key][f]

                # nsys: 精确匹配 → 合并匹配 → 插值
                nsys_entry = None
                for try_key in [f"b{b}_s{s}_{phase}", f"b{b}_s{s}"]:
                    cand = nsys.get(try_key)
                    if cand and cand.get("num_kernels", 0) >= 100 and cand.get("t_ffn", 0) < 10000:
                        nsys_entry = cand
                        break
                if nsys_entry:
                    for f in NSYS_FIELDS:
                        if f in nsys_entry and nsys_entry[f] is not None:
                            entry[f] = nsys_entry[f]

                # ci
                ci_key = f"b{b}_s{s}"
                if ci_key in ci:
                    for f in ["ci_attn", "ci_ffn", "attn_tflops", "ffn_tflops"]:
                        if f in ci[ci_key] and ci[ci_key][f] is not None:
                            entry[f] = ci[ci_key][f]

                # 理论估算 l2_attn, l2_ffn, ipc
                try:
                    mp = get_model_params(args.model)
                    patch_entry(entry, mp)
                except ValueError:
                    pass

                has_sens = all(entry.get(f"sigma_{d}") is not None for d in ["bs","cu","l2","bw"])
                has_nsys = entry.get("t_ffn") is not None
                has_ci = entry.get("ci_ffn") is not None
                has_est = entry.get("l2_ffn") is not None and entry.get("ipc") is not None
                entry["complete"] = has_sens and has_nsys and has_ci and has_est

                complete[key] = entry

    n = len(complete)
    nc = sum(1 for v in complete.values() if v["complete"])
    print(f"\n{nc}/{n} entries fully complete")

    for key, val in complete.items():
        s = "OK" if val["complete"] else "--"
        print(f"  {key:25s} {s}")

    out = str(d / "mlwd_complete.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(complete, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
