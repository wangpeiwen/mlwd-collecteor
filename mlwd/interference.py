"""基于 MLWD 的规则化干扰系数估算。

实现 chap3 §3.4 的加权映射规则：
- 资源竞争强度向量 A(r) 映射
- 逐对干扰系数 α_p, α_d 估算
- 节点级聚合
- OLS 权重标定
- 干扰矩阵可视化

Usage:
    python -m mlwd.interference estimate --victim output/Qwen-2.5-7B.json --aggressor output/Llama-3.2-3B.json
    python -m mlwd.interference matrix --files output/Qwen-2.5-7B.json output/Llama-3.2-3B.json
    python -m mlwd.interference calibrate --data output/colocation.json --output output/weights.json
"""

import argparse, json, os
from itertools import product
from pathlib import Path
import numpy as np

# 默认权重：CU 和 BW 为 LLM 推理主要竞争维度
DEFAULT_WEIGHTS = {
    "w_bs": 0.15, "w_cu": 0.25, "w_l2": 0.20,
    "w_bw": 0.25, "w_ipc": 0.10, "w_overlap": 0.05,
}
DIMS = ["bs", "cu", "l2", "bw"]


# ── 资源竞争强度向量 ──────────────────────────────────────────

def compute_aggressor_strength(entry):
    """从 MLWD 条目映射四维资源竞争强度向量 A(r)。"""
    g_launch = entry.get("g_launch", 1.0)
    ci_attn = entry.get("ci_attn", 0.01)
    ci_ffn = entry.get("ci_ffn", 0.01)
    r_attn = entry.get("r_attn", 0.0)
    r_ffn = entry.get("r_ffn", 0.0)
    l2_attn = entry.get("l2_attn", 0.5)
    l2_ffn = entry.get("l2_ffn", 0.5)

    return {
        "A_bs": 1.0 / max(g_launch, 0.1),
        "A_cu": ci_attn * r_attn + ci_ffn * r_ffn,
        "A_l2": (1 - l2_attn) * r_attn + (1 - l2_ffn) * r_ffn,
        "A_bw": r_attn / max(ci_attn, 0.01) + r_ffn / max(ci_ffn, 0.01),
    }


def compute_overlap(entry_r, entry_u):
    """时间交错因子 Ω(r, u)。"""
    active_r = entry_r.get("r_attn", 0) + entry_r.get("r_ffn", 0)
    active_u = entry_u.get("r_attn", 0) + entry_u.get("r_ffn", 0)
    return min(active_r, active_u)


# ── 逐对干扰系数 ──────────────────────────────────────────────

def estimate_alpha_d(victim, aggressor, weights=None):
    """Decode 干扰系数：aggressor (prefill r) 对 victim (decode u) 的干扰。

    α_d(u, r) = Σ w_dim · σ_dim(u) · A_dim(r) + w_ipc · IPC(r) · σ_cu(u) + w_overlap · Ω
    """
    w = weights or DEFAULT_WEIGHTS
    A = compute_aggressor_strength(aggressor)
    omega = compute_overlap(aggressor, victim)
    alpha = 0.0
    for dim in DIMS:
        alpha += w[f"w_{dim}"] * victim.get(f"sigma_{dim}", 0) * A[f"A_{dim}"]
    alpha += w["w_ipc"] * aggressor.get("ipc", 0) * victim.get("sigma_cu", 0)
    alpha += w["w_overlap"] * omega
    return alpha


def estimate_alpha_p(prefill, decode, weights=None):
    """Prefill 干扰系数：decode (u) 对 prefill (r) 的干扰。

    α_p(r, u) = Σ w_dim · σ_dim(r) · A_dim(u) + w_ipc · IPC(u) · σ_cu(r) + w_overlap · Ω
    """
    w = weights or DEFAULT_WEIGHTS
    A = compute_aggressor_strength(decode)
    omega = compute_overlap(decode, prefill)
    alpha = 0.0
    for dim in DIMS:
        alpha += w[f"w_{dim}"] * prefill.get(f"sigma_{dim}", 0) * A[f"A_{dim}"]
    alpha += w["w_ipc"] * decode.get("ipc", 0) * prefill.get("sigma_cu", 0)
    alpha += w["w_overlap"] * omega
    return alpha


# ── 节点级聚合 ──────────────────────────────────────────────

def aggregate_node_mlwd(entries, remaining_tokens=None):
    """按剩余输出长度加权聚合节点上多个 Decode 请求的 MLWD。"""
    n = len(entries)
    if n == 0:
        return {}
    if remaining_tokens is None:
        remaining_tokens = [1.0] * n
    total_weight = sum(remaining_tokens)
    if total_weight <= 0:
        total_weight = 1.0

    agg = {}
    # 聚合 σ
    for dim in DIMS:
        agg[f"sigma_{dim}"] = sum(
            (rt / total_weight) * e.get(f"sigma_{dim}", 0)
            for e, rt in zip(entries, remaining_tokens)
        )
    # 聚合 r_attn, r_ffn, ipc, l2
    for field in ["r_attn", "r_ffn", "ipc", "l2_attn", "l2_ffn",
                   "ci_attn", "ci_ffn", "g_launch"]:
        agg[field] = sum(
            (rt / total_weight) * e.get(field, 0)
            for e, rt in zip(entries, remaining_tokens)
        )
    return agg


def estimate_alpha_d_node(node_agg, new_request, weights=None):
    """节点级 Decode 干扰系数。"""
    return estimate_alpha_d(node_agg, new_request, weights)


def estimate_alpha_p_node(new_request, node_agg, weights=None):
    """节点级 Prefill 干扰系数。"""
    return estimate_alpha_p(new_request, node_agg, weights)


# ── OLS 标定 ──────────────────────────────────────────────

def build_feature_row(victim, aggressor):
    """构建单个 (victim, aggressor) 对的 6 维特征行。"""
    A = compute_aggressor_strength(aggressor)
    omega = compute_overlap(aggressor, victim)
    return [
        victim.get(f"sigma_{dim}", 0) * A[f"A_{dim}"]
        for dim in DIMS
    ] + [
        aggressor.get("ipc", 0) * victim.get("sigma_cu", 0),
        omega,
    ]


def generate_synthetic_targets(data_victim, data_aggressor):
    """从 sensitivity 数据生成合成干扰目标，用于无真实共置数据时的初始标定。

    合成 α_d ≈ 各维度敏感度与攻击强度的几何平均，模拟真实干扰的量级。
    """
    X, y = [], []
    for vk, victim in data_victim.items():
        if victim.get("phase") != "decode":
            continue
        for ak, aggressor in data_aggressor.items():
            if aggressor.get("phase") != "prefill":
                continue
            row = build_feature_row(victim, aggressor)
            # 合成目标：各维度 σ·A 的均值作为近似 α
            synthetic_alpha = np.mean([
                victim.get(f"sigma_{dim}", 0) * compute_aggressor_strength(aggressor)[f"A_{dim}"]
                for dim in DIMS
            ])
            X.append(row)
            y.append(synthetic_alpha)
    return np.array(X), np.array(y)


def calibrate_weights(X, y):
    """OLS 标定 6 个加权系数。"""
    w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    return {
        f"w_{name}": round(float(w[i]), 6)
        for i, name in enumerate(DIMS + ["ipc", "overlap"])
    }


# ── 干扰矩阵 ──────────────────────────────────────────────

def compute_interference_matrix(models, weights=None):
    """计算所有 (decode victim, prefill aggressor) 对的干扰矩阵。"""
    # 收集所有 decode 和 prefill 条目
    decode_entries, prefill_entries = [], []
    for mname, data in models.items():
        for key, entry in data.items():
            entry["_label"] = f"{mname}\n{key}"
            if entry.get("phase") == "decode":
                decode_entries.append(entry)
            elif entry.get("phase") == "prefill":
                prefill_entries.append(entry)

    n_dec, n_pre = len(decode_entries), len(prefill_entries)
    alpha_d_mat = np.zeros((n_dec, n_pre))
    alpha_p_mat = np.zeros((n_dec, n_pre))

    for i, victim in enumerate(decode_entries):
        for j, aggressor in enumerate(prefill_entries):
            alpha_d_mat[i, j] = estimate_alpha_d(victim, aggressor, weights)
            alpha_p_mat[i, j] = estimate_alpha_p(aggressor, victim, weights)

    dec_labels = [e["_label"] for e in decode_entries]
    pre_labels = [e["_label"] for e in prefill_entries]
    return alpha_d_mat, alpha_p_mat, dec_labels, pre_labels


# ── 可视化 ──────────────────────────────────────────────

def plot_interference_matrix(alpha_d_mat, alpha_p_mat, dec_labels, pre_labels, out_dir):
    """绘制干扰系数热力图。"""
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.font_manager import FontProperties, fontManager

    _FONT_DIR = Path(__file__).parent.parent / "fonts"
    for fname in ["Songti.ttc", "SimSun.ttf", "FangSong.ttf"]:
        fpath = _FONT_DIR / fname
        if fpath.exists():
            fontManager.addfont(str(fpath))
            cn = FontProperties(fname=str(fpath)).get_name()
            matplotlib.rcParams['font.sans-serif'] = [cn, 'Times New Roman', 'DejaVu Sans']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'stix'

    os.makedirs(out_dir, exist_ok=True)

    for mat, title, fname in [
        (alpha_d_mat, r"Decode 干扰系数 $\hat{\alpha}_d$", "interference_alpha_d.png"),
        (alpha_p_mat, r"Prefill 干扰系数 $\hat{\alpha}_p$", "interference_alpha_p.png"),
    ]:
        fig, ax = plt.subplots(figsize=(max(10, len(pre_labels)*0.8), max(8, len(dec_labels)*0.6)))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pre_labels)))
        ax.set_xticklabels(pre_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(dec_labels)))
        ax.set_yticklabels(dec_labels, fontsize=7)
        ax.set_xlabel("Aggressor (Prefill)")
        ax.set_ylabel("Victim (Decode)")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if mat[i,j] > mat.max()*0.7 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved {os.path.join(out_dir, fname)}")


# ── CLI ──────────────────────────────────────────────

def _load_model(path):
    name = Path(path).stem
    with open(path) as f:
        return name, json.load(f)


def cmd_estimate(args):
    vname, vdata = _load_model(args.victim)
    aname, adata = _load_model(args.aggressor)
    w = DEFAULT_WEIGHTS

    print(f"Victim: {vname}, Aggressor: {aname}")
    print(f"Weights: {w}\n")
    print(f"{'Victim (Decode)':<28s} {'Aggressor (Prefill)':<28s} {'α_d':>8s} {'α_p':>8s}")
    print("-" * 76)

    for vk, victim in vdata.items():
        if victim.get("phase") != "decode":
            continue
        for ak, aggressor in adata.items():
            if aggressor.get("phase") != "prefill":
                continue
            ad = estimate_alpha_d(victim, aggressor, w)
            ap = estimate_alpha_p(aggressor, victim, w)
            print(f"{vname}/{vk:<20s} {aname}/{ak:<20s} {ad:8.4f} {ap:8.4f}")


def cmd_matrix(args):
    models = {}
    for p in args.files:
        name, data = _load_model(p)
        models[name] = data

    ad, ap, dl, pl = compute_interference_matrix(models)
    out_dir = args.output or "output/interference_plots"
    plot_interference_matrix(ad, ap, dl, pl, out_dir)

    # 保存数值结果
    result = {
        "alpha_d": {"matrix": ad.tolist(), "rows": dl, "cols": pl},
        "alpha_p": {"matrix": ap.tolist(), "rows": dl, "cols": pl},
        "weights": DEFAULT_WEIGHTS,
    }
    json_path = os.path.join(out_dir, "interference_matrix.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved {json_path}")


def cmd_calibrate(args):
    with open(args.data) as f:
        coloc = json.load(f)

    # 期望格式: [{"victim": {...}, "aggressor": {...}, "alpha_d": float, "alpha_p": float}, ...]
    X_d, y_d, X_p, y_p = [], [], [], []
    for sample in coloc:
        row_d = build_feature_row(sample["victim"], sample["aggressor"])
        row_p = build_feature_row(sample["aggressor"], sample["victim"])
        X_d.append(row_d)
        y_d.append(sample["alpha_d"])
        X_p.append(row_p)
        y_p.append(sample["alpha_p"])

    X_d, y_d = np.array(X_d), np.array(y_d)
    X_p, y_p = np.array(X_p), np.array(y_p)

    # 合并 α_d 和 α_p 样本联合标定
    X = np.vstack([X_d, X_p])
    y = np.concatenate([y_d, y_p])
    w = calibrate_weights(X, y)

    print(f"Calibrated weights ({len(coloc)} samples):")
    for k, v in w.items():
        print(f"  {k}: {v:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(w, f, indent=2)
        print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="MLWD 干扰系数估算")
    sub = parser.add_subparsers(dest="cmd")

    p_est = sub.add_parser("estimate", help="逐对干扰系数估算")
    p_est.add_argument("--victim", required=True)
    p_est.add_argument("--aggressor", required=True)

    p_mat = sub.add_parser("matrix", help="干扰矩阵 + 热力图")
    p_mat.add_argument("--files", nargs="+", required=True)
    p_mat.add_argument("--output", default=None)

    p_cal = sub.add_parser("calibrate", help="OLS 权重标定")
    p_cal.add_argument("--data", required=True, help="共置实验数据 JSON")
    p_cal.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.cmd == "estimate":
        cmd_estimate(args)
    elif args.cmd == "matrix":
        cmd_matrix(args)
    elif args.cmd == "calibrate":
        cmd_calibrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
