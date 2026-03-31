"""可视化（中文宋体 + Times New Roman，字体从项目 fonts/ 目录加载）。"""

import argparse, json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties, fontManager

# 从项目 fonts/ 目录加载字体，不依赖系统安装
_FONT_DIR = Path(__file__).parent.parent / "fonts"
_CN_FONT = None
_EN_FONT = None

for fname in ["Songti.ttc", "SimSun.ttf", "FangSong.ttf", "STFangsong.ttf"]:
    fpath = _FONT_DIR / fname
    if fpath.exists():
        fontManager.addfont(str(fpath))
        _CN_FONT = FontProperties(fname=str(fpath))
        break

for fname in ["Times New Roman.ttf", "TimesNewRoman.ttf"]:
    fpath = _FONT_DIR / fname
    if fpath.exists():
        fontManager.addfont(str(fpath))
        _EN_FONT = FontProperties(fname=str(fpath))
        break

# 也注册 Bold 版本
_bold_path = _FONT_DIR / "Times New Roman Bold.ttf"
if _bold_path.exists():
    fontManager.addfont(str(_bold_path))

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'

if _CN_FONT:
    cn_name = _CN_FONT.get_name()
    matplotlib.rcParams['font.sans-serif'] = [cn_name, 'Times New Roman', 'DejaVu Sans']
    print(f"Chinese font: {cn_name} ({_CN_FONT.get_file()})")
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'FangSong', 'DejaVu Sans']
    print("Warning: No Chinese font found in fonts/, using fallback")

if _EN_FONT:
    matplotlib.rcParams['font.serif'] = [_EN_FONT.get_name(), 'DejaVu Serif']
else:
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

DIMS = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]
DIM_CN = ["线程块\n调度器", "计算\n单元", "L2\n缓存", "显存\n带宽"]
COLORS = {"prefill": "#D32F2F", "decode": "#1976D2"}


def fig1_phase_compare(data, out):
    """Prefill vs Decode 四维敏感度对比。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for row, b in enumerate([1, 4]):
        for col, s in enumerate([32, 64, 128]):
            ax = axes[row][col]
            pv = [data.get(f"b{b}_s{s}_prefill", {}).get(d, 0) or 0 for d in DIMS]
            dv = [data.get(f"b{b}_s{s}_decode", {}).get(d, 0) or 0 for d in DIMS]
            x = np.arange(4); w = 0.35
            ax.bar(x-w/2, pv, w, label="Prefill", color=COLORS["prefill"], alpha=0.85)
            ax.bar(x+w/2, dv, w, label="Decode", color=COLORS["decode"], alpha=0.85)
            for bars in [ax.containers[0], ax.containers[1]]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0: ax.text(bar.get_x()+bar.get_width()/2, h+0.02, f'{h:.2f}', ha='center', fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(DIM_CN, fontsize=8)
            ax.set_title(f'b={b}, s={s}', fontsize=11, family='Times New Roman')
            if col == 0: ax.set_ylabel('干扰敏感度 ($\\sigma$)', fontsize=10)
            if row == 0 and col == 2: ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')
    fig.suptitle('Prefill 与 Decode 阶段的四维干扰敏感度对比', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out, "fig1_phase_compare.png"), dpi=200); plt.close()


def fig2_heatmap(data, out):
    """敏感度热力图。"""
    order = [f"b{b}_s{s}_{p}" for p in ["prefill","decode"] for b in [1,4] for s in [32,64,128]]
    matrix, labels = [], []
    for k in order:
        if k in data:
            matrix.append([data[k].get(d, 0) or 0 for d in DIMS])
            d = data[k]; labels.append(f"b={d['batch_size']}, s={d['seq_len']}, {d['phase']}")
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["线程块调度器","计算单元","L2 缓存","显存带宽"], fontsize=10)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    n_p = sum(1 for k in order if "prefill" in k and k in data)
    ax.axhline(y=n_p-0.5, color='black', linewidth=2)
    for i in range(len(labels)):
        for j in range(4):
            v = matrix[i,j]; c = "white" if v > matrix.max()*0.65 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=c)
    plt.colorbar(im, ax=ax, label="干扰敏感度 ($\\sigma$)", shrink=0.8)
    ax.set_title("MLWD 干扰敏感度画像", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig2_heatmap.png"), dpi=200); plt.close()


def fig3_trends(data, out):
    """敏感度随 (b,s) 变化趋势。"""
    colors = ["#D32F2F", "#1976D2", "#388E3C", "#F57C00"]
    markers = ["o", "s", "^", "D"]
    labels = ["$\\sigma_{bs}$", "$\\sigma_{cu}$", "$\\sigma_{l2}$", "$\\sigma_{bw}$"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (row, col, phase, b, title) in [
        (0,0,"prefill",1,"Prefill, b=1"), (0,1,"prefill",4,"Prefill, b=4"),
        (1,0,"decode",1,"Decode, b=1"), (1,1,"decode",4,"Decode, b=4")]:
        ax = axes[row][col]
        for i, d in enumerate(DIMS):
            vals = [data.get(f"b{b}_s{s}_{phase}", {}).get(d, 0) or 0 for s in [32,64,128]]
            ax.plot([32,64,128], vals, f"{markers[i]}-", color=colors[i], label=labels[i], linewidth=2, markersize=7)
        ax.set_xlabel("序列长度 (s)", fontsize=10); ax.set_ylabel("干扰敏感度 ($\\sigma$)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks([32,64,128])
    fig.suptitle("干扰敏感度随 (batch_size, seq_len) 的变化趋势", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out, "fig3_trends.png"), dpi=200); plt.close()


def fig4_execution(data, out):
    """执行模式特征。"""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # 时间占比
    ax = axes[0]; keys, ra, rf, ro = [], [], [], []
    for b in [1,4]:
        for s in [32,128]:
            for p in ["prefill","decode"]:
                k = f"b{b}_s{s}_{p}"
                if k in data:
                    keys.append(f"b{b}s{s}\n{p[:3]}"); d = data[k]
                    a, f_ = d.get("r_attn",0) or 0, d.get("r_ffn",0) or 0
                    ra.append(a); rf.append(f_); ro.append(max(0, 1-a-f_))
    x = np.arange(len(keys))
    ax.bar(x, rf, label="FFN", color="#1976D2", alpha=0.85)
    ax.bar(x, ra, bottom=rf, label="Attention", color="#D32F2F", alpha=0.85)
    ax.bar(x, ro, bottom=[a+f for a,f in zip(ra,rf)], label="Other", color="#BDBDBD", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("时间占比"); ax.set_title("Attention/FFN 时间占比", fontweight='bold'); ax.legend(fontsize=8); ax.set_ylim(0,1.05)
    # kernel 时延
    ax = axes[1]; ks, ta, tf = [], [], []
    for b in [1,4]:
        for s in [32,128]:
            k = f"b{b}_s{s}_prefill"
            if k in data and data[k].get("t_ffn"):
                ks.append(f"b{b}s{s}"); ta.append(data[k].get("t_attn",0) or 0); tf.append(data[k].get("t_ffn",0))
    if ks:
        x2 = np.arange(len(ks)); w = 0.35
        ax.bar(x2-w/2, ta, w, label="$\\bar{t}_{attn}$", color="#D32F2F", alpha=0.85)
        ax.bar(x2+w/2, tf, w, label="$\\bar{t}_{ffn}$", color="#1976D2", alpha=0.85)
        ax.set_xticks(x2); ax.set_xticklabels(ks); ax.set_ylabel("时延 ($\\mu$s)"); ax.set_title("Kernel 平均时延", fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.2, axis='y')
    # 交替频率
    ax = axes[2]; ks3, fs = [], []
    for b in [1,4]:
        for s in [32,64,128]:
            for p in ["prefill","decode"]:
                k = f"b{b}_s{s}_{p}"
                if k in data and data[k].get("f_switch") is not None:
                    ks3.append(f"b{b}s{s}\n{p[:3]}"); fs.append(data[k]["f_switch"])
    if ks3:
        x3 = np.arange(len(ks3)); cs = [COLORS.get(("prefill" if "pre" in k else "decode"), "#999") for k in ks3]
        ax.bar(x3, fs, color=cs, alpha=0.85); ax.set_xticks(x3); ax.set_xticklabels(ks3, fontsize=7)
        ax.set_ylabel("$f_{switch}$ (次/秒)"); ax.set_title("计算-访存交替频率", fontweight='bold'); ax.grid(True, alpha=0.2, axis='y')
        ax.legend(handles=[Patch(color=COLORS["prefill"], label="Prefill"), Patch(color=COLORS["decode"], label="Decode")], fontsize=9)
    fig.suptitle("MLWD 执行模式特征", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out, "fig4_execution.png"), dpi=200); plt.close()


def fig5_baseline(data, out):
    """基线时延缩放。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, phase in enumerate(["prefill", "decode"]):
        ax = axes[idx]
        for b, c, m in [(1, "#D32F2F", "o"), (4, "#1976D2", "s")]:
            ss, bls = [], []
            for s in [32, 64, 128]:
                bl = data.get(f"b{b}_s{s}_{phase}", {}).get("baseline_ms")
                if bl: ss.append(s); bls.append(bl)
            ax.plot(ss, bls, f"{m}-", color=c, label=f"batch={b}", linewidth=2, markersize=8)
            for s_, bl_ in zip(ss, bls):
                ax.annotate(f"{bl_:.0f}ms", (s_, bl_), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
        ax.set_xlabel("序列长度 (s)"); ax.set_ylabel("基线时延 (ms)")
        ax.set_title(f"{phase.capitalize()}", fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks([32,64,128])
    fig.suptitle("基线时延随 (batch_size, seq_len) 的缩放关系", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out, "fig5_baseline.png"), dpi=200); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(OUTPUT_DIR / "mlwd_complete.json"))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "plots"))
    args = parser.parse_args()
    with open(args.data) as f: data = json.load(f)
    os.makedirs(args.output, exist_ok=True)
    print(f"Loaded {len(data)} entries, generating plots...")
    fig1_phase_compare(data, args.output); print("  fig1")
    fig2_heatmap(data, args.output); print("  fig2")
    fig3_trends(data, args.output); print("  fig3")
    fig4_execution(data, args.output); print("  fig4")
    fig5_baseline(data, args.output); print("  fig5")
    print(f"Saved to {args.output}/")


if __name__ == "__main__":
    main()
