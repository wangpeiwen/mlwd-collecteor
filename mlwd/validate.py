"""数据验证。"""

import json
from .config import OUTPUT_DIR


def main():
    with open(str(OUTPUT_DIR / "mlwd_complete.json")) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries\n")
    ok = True

    # 完整性
    required = ["baseline_ms", "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw", "t_ffn", "g_launch", "r_ffn"]
    print("=== 完整性 ===")
    for key, val in data.items():
        missing = [f for f in required if val.get(f) is None]
        if missing:
            print(f"  {key}: 缺少 {missing}")
            ok = False
    if ok: print("  OK\n")

    # 值域
    print("=== 值域 ===")
    for key, val in data.items():
        for d in ["bs", "cu", "l2", "bw"]:
            s = val.get(f"sigma_{d}")
            if s is not None and s < 0:
                print(f"  {key}: sigma_{d}={s} < 0"); ok = False
        ra, rf = val.get("r_attn", 0) or 0, val.get("r_ffn", 0) or 0
        if ra + rf > 1.01:
            print(f"  {key}: r_attn+r_ffn={ra+rf} > 1"); ok = False
    print("  OK\n" if ok else "")

    # 趋势
    print("=== 趋势 ===")
    for b in [1, 4]:
        bls = [(s, data.get(f"b{b}_s{s}_decode", {}).get("baseline_ms", 0)) for s in [32, 64, 128]]
        for i in range(1, len(bls)):
            if bls[i][1] < bls[i-1][1] and bls[i][1] > 0:
                print(f"  b{b} decode baseline 非单调: s={bls[i-1][0]}({bls[i-1][1]:.0f}) > s={bls[i][0]}({bls[i][1]:.0f})")
    print("  OK\n")

    print("=== 验证通过 ===" if ok else "=== 存在问题 ===")


if __name__ == "__main__":
    main()
