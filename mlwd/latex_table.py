"""生成 LaTeX 表格：15 维 MLWD 向量示例。"""

import json
from .config import OUTPUT_DIR

FIELDS = [
    (r"$\mathrm{CI}_{\mathrm{attn}}$", "ci_attn", "FLOP/Byte", "资源竞争强度"),
    (r"$\mathrm{CI}_{\mathrm{ffn}}$", "ci_ffn", "FLOP/Byte", None),
    (r"$\mathrm{L2}_{\mathrm{attn}}$", "l2_attn", "", None),
    (r"$\mathrm{L2}_{\mathrm{ffn}}$", "l2_ffn", "", None),
    (r"$\sigma_{\mathrm{bs}}$", "sigma_bs", "", "干扰敏感度"),
    (r"$\sigma_{\mathrm{cu}}$", "sigma_cu", "", None),
    (r"$\sigma_{\mathrm{l2}}$", "sigma_l2", "", None),
    (r"$\sigma_{\mathrm{bw}}$", "sigma_bw", "", None),
    (r"$\bar{t}_{\mathrm{attn}}$", "t_attn", r"$\mu$s", "执行模式"),
    (r"$\bar{t}_{\mathrm{ffn}}$", "t_ffn", r"$\mu$s", None),
    (r"$\bar{g}_{\mathrm{launch}}$", "g_launch", r"$\mu$s", None),
    (r"$r_{\mathrm{attn}}$", "r_attn", "", None),
    (r"$r_{\mathrm{ffn}}$", "r_ffn", "", None),
    (r"$f_{\mathrm{switch}}$", "f_switch", "次/秒", None),
    (r"$\overline{\mathrm{IPC}}$", "ipc", "", None),
]

EXAMPLES = [
    ("b1_s128_prefill", r"\makecell{Prefill\\(b=1,s=128)}"),
    ("b1_s128_decode",  r"\makecell{Decode\\(b=1,s=128)}"),
    ("b4_s128_prefill", r"\makecell{Prefill\\(b=4,s=128)}"),
    ("b4_s128_decode",  r"\makecell{Decode\\(b=4,s=128)}"),
]


def _fmt(v):
    if v is None: return "—"
    if isinstance(v, float):
        return f"{v:.1f}" if v > 100 else f"{v:.4f}"
    return str(v)


def main():
    with open(str(OUTPUT_DIR / "mlwd_complete.json")) as f:
        data = json.load(f)

    ncols = len(EXAMPLES) + 2
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Qwen2.5-7B-Instruct 在 V100 上的 MLWD 向量示例}")
    print(r"\label{tab:mlwd_example}")
    print(r"\small")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\begin{tabular}{l c " + "c " * len(EXAMPLES) + "}")
    print(r"\toprule")
    header = "特征 & 单位 & " + " & ".join(h for _, h in EXAMPLES) + r" \\"
    print(header)
    print(r"\midrule")

    for name, key, unit, group in FIELDS:
        if group:
            print(rf"\midrule")
            print(rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{group}}}}} \\")
        vals = [_fmt(data.get(ek, {}).get(key)) for ek, _ in EXAMPLES]
        print(f"{name} & {unit} & {' & '.join(vals)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # 向量表示
    print("\n% 15 维向量")
    for ek, label in EXAMPLES:
        d = data.get(ek, {})
        vec = [_fmt(d.get(key)) for _, key, _, _ in FIELDS]
        print(f"% {label}: W = [{', '.join(vec)}]")


if __name__ == "__main__":
    main()
