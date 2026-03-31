"""双模型 MLWD 对比可视化。"""

import argparse, json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties, fontManager

# 字体加载
_FONT_DIR = Path(__file__).parent.parent / "fonts"
for fname in ["Songti.ttc", "SimSun.ttf", "FangSong.ttf"]:
    fpath = _FONT_DIR / fname
    if fpath.exists():
        fontManager.addfont(str(fpath))
        cn = FontProperties(fname=str(fpath)).get_name()
        matplotlib.rcParams['font.sans-serif'] = [cn, 'Times New Roman', 'DejaVu Sans']
        break
for fname in ["Times New Roman.ttf", "Times New Roman Bold.ttf"]:
    fpath = _FONT_DIR / fname
    if fpath.exists(): fontManager.addfont(str(fpath))
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

DIMS = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]
DIM_CN = ["线程块调度器", "计算单元", "L2 缓存", "显存带宽"]
M_COLORS = {"Qwen-2.5-7B": "#D32F2F", "Llama-3.2-3B": "#1976D2"}


def fig_sensitivity_compare(models, out):
    """图1: 两模型四维敏感度对比（Prefill vs Decode）。"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    model_names = list(models.keys())
    colors = [M_COLORS.get(n, f"C{i}") for i, n in enumerate(model_names)]

    for row, phase in enumerate(["prefill", "decode"]):
        for col, s in enumerate([32, 64, 128]):
            ax = axes[row][col]
            x = np.arange(4)
            w = 0.35
            for mi, (mname, mdata) in enumerate(models.items()):
                # 用 b=1 的数据
                key = f"b1_s{s}_{phase}"
                vals = [mdata.get(key, {}).get(d, 0) or 0 for d in DIMS]
                offset = -w/2 + mi * w
                bars = ax.bar(x + offset, vals, w, label=mname, color=colors[mi], alpha=0.85)
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                                f'{h:.2f}', ha='center', fontsize=6)

            ax.set_xticks(x)
            ax.set_xticklabels(DIM_CN, fontsize=8)
            ax.set_title(f'{phase.capitalize()}, s={s}', fontsize=11, family='Times New Roman')
            if col == 0:
                ax.set_ylabel('干扰敏感度 ($\\sigma$)', fontsize=10)
            if row == 0 and col == 2:
                ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('不同模型的四维干扰敏感度对比 (b=1)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out, "compare_sensitivity.png"), dpi=200)
    plt.close()
    print("  compare_sensitivity.png")


def fig_ci_compare(models, out):
    """图2: 两模型 CI 对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    model_names = list(models.keys())
    colors = [M_COLORS.get(n, f"C{i}") for i, n in enumerate(model_names)]
    markers = ["o", "s", "^", "D"]

    for idx, feat in enumerate(["ci_attn", "ci_ffn"]):
        ax = axes[idx]
        for mi, (mname, mdata) in enumerate(models.items()):
            ss, vals = [], []
            for s in [32, 64, 128]:
                key = f"b1_s{s}_prefill"
                v = mdata.get(key, {}).get(feat)
                if v is not None:
                    ss.append(s)
                    vals.append(v)
            ax.plot(ss, vals, f"{markers[mi]}-", color=colors[mi],
                    label=mname, linewidth=2, markersize=8)
            for s_, v_ in zip(ss, vals):
                ax.annotate(f"{v_:.2f}", (s_, v_), textcoords="offset points",
                           xytext=(0, 8), ha='center', fontsize=8)

        title = "Attention 计算强度" if "attn" in feat else "FFN 计算强度"
        ax.set_xlabel("序列长度 (s)", fontsize=10)
        ax.set_ylabel("CI (FLOP/Byte)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([32, 64, 128])

    fig.suptitle('不同模型的计算强度 (CI) 对比 (b=1)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out, "compare_ci.png"), dpi=200)
    plt.close()
    print("  compare_ci.png")


def fig_execution_compare(models, out):
    """图3: 两模型执行模式对比（r_ffn, g_launch, f_switch）。"""
    feats = [("r_ffn", "FFN 时间占比", ""),
             ("g_launch", "Kernel Launch 间隔", "μs"),
             ("f_switch", "计算-访存交替频率", "次/秒")]
    model_names = list(models.keys())
    colors = [M_COLORS.get(n, f"C{i}") for i, n in enumerate(model_names)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (feat, title, unit) in enumerate(feats):
        ax = axes[idx]
        x_labels = []
        bar_data = {n: [] for n in model_names}

        for s in [32, 64, 128]:
            for phase in ["prefill", "decode"]:
                x_labels.append(f"s{s}\n{phase[:3]}")
                for mname, mdata in models.items():
                    key = f"b1_s{s}_{phase}"
                    bar_data[mname].append(mdata.get(key, {}).get(feat, 0) or 0)

        x = np.arange(len(x_labels))
        w = 0.35
        for mi, mname in enumerate(model_names):
            offset = -w/2 + mi * w
            ax.bar(x + offset, bar_data[mname], w, label=mname, color=colors[mi], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)
        ylabel = f"{title}" + (f" ({unit})" if unit else "")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('不同模型的执行模式特征对比 (b=1)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out, "compare_execution.png"), dpi=200)
    plt.close()
    print("  compare_execution.png")


def fig_baseline_compare(models, out):
    """图4: 两模型基线时延对比。"""
    model_names = list(models.keys())
    colors = [M_COLORS.get(n, f"C{i}") for i, n in enumerate(model_names)]
    markers = ["o", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, phase in enumerate(["prefill", "decode"]):
        ax = axes[idx]
        for mi, (mname, mdata) in enumerate(models.items()):
            ss, bls = [], []
            for s in [32, 64, 128]:
                key = f"b1_s{s}_{phase}"
                bl = mdata.get(key, {}).get("baseline_ms")
                if bl:
                    ss.append(s)
                    bls.append(bl)
            ax.plot(ss, bls, f"{markers[mi]}-", color=colors[mi],
                    label=mname, linewidth=2, markersize=8)
            for s_, bl_ in zip(ss, bls):
                ax.annotate(f"{bl_:.0f}", (s_, bl_), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8)

        ax.set_xlabel("序列长度 (s)", fontsize=10)
        ax.set_ylabel("基线时延 (ms)", fontsize=10)
        ax.set_title(f"{phase.capitalize()}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([32, 64, 128])

    fig.suptitle('不同模型的基线时延对比 (b=1)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out, "compare_baseline.png"), dpi=200)
    plt.close()
    print("  compare_baseline.png")


def fig_heatmap_compare(models, out):
    """图5: 两模型敏感度热力图并排，b=1 和 b=4。"""
    model_names = list(models.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={"wspace": 0.4, "hspace": 0.35})

    vmax = 0
    for mdata in models.values():
        for b in [1, 4]:
            for k in [f"b{b}_s{s}_{p}" for p in ["prefill","decode"] for s in [32,64,128]]:
                for d in DIMS:
                    v = mdata.get(k, {}).get(d, 0) or 0
                    vmax = max(vmax, v)
    vmax = max(2.5, vmax)

    im = None
    for row, b in enumerate([1, 4]):
        for col, (mname, mdata) in enumerate(models.items()):
            ax = axes[row][col]
            order = [f"b{b}_s{s}_{p}" for p in ["prefill", "decode"] for s in [32, 64, 128]]
            matrix, labels = [], []
            for k in order:
                if k in mdata:
                    matrix.append([mdata[k].get(d, 0) or 0 for d in DIMS])
                    dd = mdata[k]
                    labels.append(f"s={dd['seq_len']}, {dd['phase']}")

            if not matrix:
                ax.set_visible(False)
                continue

            matrix = np.array(matrix)
            im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(4))
            ax.set_xticklabels(DIM_CN, fontsize=9)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)

            n_p = sum(1 for k in order if "prefill" in k and k in mdata)
            ax.axhline(y=n_p - 0.5, color='black', linewidth=1.5)

            for i in range(len(labels)):
                for j in range(4):
                    v = matrix[i, j]
                    c = "white" if v > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=c)

            ax.set_title(f"{mname} (b={b})", fontsize=11, fontweight='bold')

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="干扰敏感度 ($\\sigma$)", shrink=0.6, pad=0.02)
    fig.suptitle('', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(out, "compare_heatmap.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  compare_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="双模型 MLWD 对比可视化")
    parser.add_argument("--files", nargs="+", required=True,
                        help="模型数据 JSON 文件，格式: name:path")
    parser.add_argument("--output", default="output/compare_plots")
    args = parser.parse_args()

    models = {}
    for spec in args.files:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name = Path(spec).stem
            path = spec
        with open(path) as f:
            models[name] = json.load(f)
        print(f"Loaded {name}: {len(models[name])} entries")

    os.makedirs(args.output, exist_ok=True)
    print(f"\nGenerating comparison plots...")

    fig_sensitivity_compare(models, args.output)
    fig_ci_compare(models, args.output)
    fig_execution_compare(models, args.output)
    fig_baseline_compare(models, args.output)
    fig_heatmap_compare(models, args.output)

    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
