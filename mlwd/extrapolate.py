"""参数化算子模型：基于现有 (b,s) 数据训练回归模型，外推到完整实验矩阵。

chap3 §3.3.4: 对 MLWD 第一层关键指标建立轻量级参数化回归模型，
以 (b,s) 为输入，算子画像指标为输出，降低离线采集成本。

Usage:
    python -m mlwd.extrapolate --input output/Qwen-2.5-7B.json --model qwen2.5-7b --output output/Qwen-2.5-7B-full.json
    python -m mlwd.extrapolate --input output/Qwen-2.5-7B.json --model qwen2.5-7b --validate
"""

import argparse, json, os
from itertools import product
from pathlib import Path
import numpy as np

from .config import (FULL_BATCH_SIZES, FULL_SEQ_LENGTHS,
                     get_model_params, V100_L2_BYTES, V100_BW_GBS)
from .estimate_missing import estimate_l2_attn, estimate_l2_ffn, estimate_ipc

# ── 回归工具 ──────────────────────────────────────────────

def _fit_power_law(xs, ys):
    """幂律回归 y = α·x^β → log(y) = log(α) + β·log(x)。"""
    mask = (np.array(xs) > 0) & (np.array(ys) > 0)
    lx, ly = np.log(np.array(xs)[mask]), np.log(np.array(ys)[mask])
    if len(lx) < 2:
        return (0.0, 1.0)
    coeffs = np.polyfit(lx, ly, 1)  # [β, log(α)]
    return (np.exp(coeffs[1]), coeffs[0])  # (α, β)


def _predict_power_law(coeffs, x):
    alpha, beta = coeffs
    return alpha * (x ** beta)


def _fit_log_linear(xs, ys):
    """对数线性 y = a + b·log(x)。"""
    lx = np.log(np.array(xs))
    coeffs = np.polyfit(lx, np.array(ys), 1)  # [b, a]
    return (coeffs[1], coeffs[0])  # (a, b)


def _predict_log_linear(coeffs, x):
    a, b = coeffs
    return a + b * np.log(x)


def _fit_linear(xs, ys):
    """线性 y = a + b·x。"""
    coeffs = np.polyfit(np.array(xs), np.array(ys), 1)
    return (coeffs[1], coeffs[0])  # (a, b)


def _predict_linear(coeffs, x):
    a, b = coeffs
    return a + b * x


def _fit_quadratic_log(xs, ys):
    """二次对数多项式 y = a + b·log(x) + c·log(x)²。"""
    lx = np.log(np.array(xs))
    coeffs = np.polyfit(lx, np.array(ys), 2)  # [c, b, a]
    return (coeffs[2], coeffs[1], coeffs[0])  # (a, b, c)


def _predict_quadratic_log(coeffs, x):
    a, b, c = coeffs
    lx = np.log(x)
    return a + b * lx + c * lx * lx


def _fit_bivar_log(xs_bs, ys):
    """双变量对数回归 y = a + b1·log(b) + b2·log(s)。xs_bs 为 (b, s) 列表。"""
    X = np.column_stack([np.ones(len(xs_bs)),
                         np.log([x[0] for x in xs_bs]),
                         np.log([x[1] for x in xs_bs])])
    coeffs, _, _, _ = np.linalg.lstsq(X, np.array(ys), rcond=None)
    return tuple(coeffs)  # (a, b1, b2)


def _predict_bivar_log(coeffs, b, s):
    a, b1, b2 = coeffs
    return a + b1 * np.log(b) + b2 * np.log(s)


def _fit_bivar_linear(xs_bs, ys):
    """双变量线性 y = a + b1·b + b2·s。"""
    X = np.column_stack([np.ones(len(xs_bs)),
                         [x[0] for x in xs_bs],
                         [x[1] for x in xs_bs]])
    coeffs, _, _, _ = np.linalg.lstsq(X, np.array(ys), rcond=None)
    return tuple(coeffs)


def _predict_bivar_linear(coeffs, b, s):
    a, b1, b2 = coeffs
    return a + b1 * b + b2 * s


# ── 数据提取 ──────────────────────────────────────────────

def _extract_phase_independent(data):
    """提取相位无关字段（nsys/ci 数据 prefill 和 decode 相同）。

    返回 {(b,s): {field: value}} 字典，每个 (b,s) 取 prefill 条目。
    """
    points = {}
    for key, entry in data.items():
        if entry.get("phase") != "prefill":
            continue
        b, s = entry["batch_size"], entry["seq_len"]
        points[(b, s)] = entry
    return points


def _extract_phase_dependent(data, phase):
    """提取指定相位的字段。"""
    points = {}
    for key, entry in data.items():
        if entry.get("phase") != phase:
            continue
        b, s = entry["batch_size"], entry["seq_len"]
        points[(b, s)] = entry
    return points


# ── 字段回归配置 ──────────────────────────────────────────

# 相位无关字段
PI_FIELDS = {
    "ci_attn":  ("power_law", "b*s"),
    "ci_ffn":   ("power_law", "b*s"),
    "t_attn":   ("bivar_log", "b,s"),
    "t_ffn":    ("bivar_log", "b,s"),
    "g_launch": ("bivar_log", "b,s"),
    "r_attn":   ("bivar_log", "b,s"),
    "r_ffn":    ("bivar_log", "b,s"),
    "f_switch": ("bivar_log", "b,s"),
}

# 相位相关字段（每个相位独立拟合）
PD_FIELDS = {
    "sigma_bs":  ("quadratic_log", "b*s"),
    "sigma_cu":  ("quadratic_log", "b*s"),
    "sigma_l2":  ("quadratic_log", "b*s"),
    "sigma_bw":  ("quadratic_log", "b*s"),
    "baseline_ms": ("bivar_linear", "b,s"),
}

FIT_FN = {
    "power_law": _fit_power_law,
    "log_linear": _fit_log_linear,
    "linear": _fit_linear,
    "quadratic_log": _fit_quadratic_log,
    "bivar_log": None,      # 特殊处理
    "bivar_linear": None,   # 特殊处理
}
PREDICT_FN = {
    "power_law": _predict_power_law,
    "log_linear": _predict_log_linear,
    "linear": _predict_linear,
    "quadratic_log": _predict_quadratic_log,
    "bivar_log": _predict_bivar_log,
    "bivar_linear": _predict_bivar_linear,
}


def _get_x(b, s, x_type):
    if x_type == "b*s":
        return b * s
    return s


# ── 训练 + 预测 ──────────────────────────────────────────

def _train_field(points, field, method, x_type):
    """在给定数据点上训练单个字段的回归模型。"""
    xs, ys, bs_pairs = [], [], []
    for (b, s), entry in sorted(points.items()):
        v = entry.get(field)
        if v is not None:
            xs.append(_get_x(b, s, x_type))
            ys.append(v)
            bs_pairs.append((b, s))
    if len(xs) < 2:
        return None
    if method in ("bivar_log", "bivar_linear"):
        if method == "bivar_log":
            return _fit_bivar_log(bs_pairs, ys)
        else:
            return _fit_bivar_linear(bs_pairs, ys)
    return FIT_FN[method](xs, ys)


def _predict_field(coeffs, method, b, s, x_type):
    if method in ("bivar_log", "bivar_linear"):
        return PREDICT_FN[method](coeffs, b, s)
    x = _get_x(b, s, x_type)
    return PREDICT_FN[method](coeffs, x)


def extrapolate_full(data, model_key,
                     batch_sizes=None, seq_lengths=None):
    """训练回归模型并外推到完整实验矩阵。"""
    if batch_sizes is None:
        batch_sizes = FULL_BATCH_SIZES
    if seq_lengths is None:
        seq_lengths = FULL_SEQ_LENGTHS

    mp = get_model_params(model_key)
    pi_points = _extract_phase_independent(data)

    # 训练相位无关模型
    pi_models = {}
    for field, (method, x_type) in PI_FIELDS.items():
        coeffs = _train_field(pi_points, field, method, x_type)
        if coeffs is not None:
            pi_models[field] = (coeffs, method, x_type)

    # 训练相位相关模型
    pd_models = {}
    for phase in ["prefill", "decode"]:
        pd_points = _extract_phase_dependent(data, phase)
        pd_models[phase] = {}
        for field, (method, x_type) in PD_FIELDS.items():
            coeffs = _train_field(pd_points, field, method, x_type)
            if coeffs is not None:
                pd_models[phase][field] = (coeffs, method, x_type)

    # 已有实测数据的 (b,s) 集合
    measured_bs = set(pi_points.keys())

    # 生成完整数据
    result = {}
    for b, s, phase in product(batch_sizes, seq_lengths, ["prefill", "decode"]):
        key = f"b{b}_s{s}_{phase}"
        is_measured = (b, s) in measured_bs

        if is_measured:
            # 使用实测数据
            entry = dict(data.get(key, {}))
            entry["source"] = "measured"
        else:
            entry = {"batch_size": b, "seq_len": s, "phase": phase, "source": "extrapolated"}
            # 相位无关字段
            for field, (coeffs, method, x_type) in pi_models.items():
                entry[field] = round(_predict_field(coeffs, method, b, s, x_type), 4)
            # 相位相关字段
            for field, (coeffs, method, x_type) in pd_models.get(phase, {}).items():
                entry[field] = round(_predict_field(coeffs, method, b, s, x_type), 4)

            # 值域约束：防止外推产生不合理值
            for f in ["ci_attn", "ci_ffn"]:
                if f in entry:
                    entry[f] = max(0.01, min(entry[f], 50.0))
            for f in ["t_attn", "t_ffn", "g_launch"]:
                if f in entry:
                    entry[f] = max(0.1, entry[f])
            for f in ["r_attn", "r_ffn"]:
                if f in entry:
                    entry[f] = max(0.0, min(entry[f], 1.0))
            for f in ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]:
                if f in entry:
                    entry[f] = max(0.0, entry[f])
            if "f_switch" in entry:
                entry["f_switch"] = max(0.0, entry["f_switch"])
            if "baseline_ms" in entry:
                entry["baseline_ms"] = max(1.0, entry["baseline_ms"])

        # 理论估算字段（实测和外推都重新计算）
        entry["l2_attn"] = estimate_l2_attn(b, s, mp)
        entry["l2_ffn"] = estimate_l2_ffn(b, s, mp)
        if entry.get("ci_attn") is not None and entry.get("r_attn") is not None:
            entry["ipc"] = estimate_ipc(
                entry.get("ci_attn", 0), entry.get("ci_ffn", 0),
                entry.get("r_attn", 0), entry.get("r_ffn", 0))

        entry["complete"] = all(
            entry.get(f) is not None
            for f in ["ci_attn", "ci_ffn", "sigma_bs", "sigma_cu",
                       "sigma_l2", "sigma_bw", "t_ffn", "l2_ffn", "ipc"])
        result[key] = entry

    return result


# ── 留一交叉验证 ──────────────────────────────────────────

def leave_one_out(data):
    """留一交叉验证，评估外推精度。"""
    pi_points = _extract_phase_independent(data)
    all_keys = sorted(pi_points.keys())

    all_fields = list(PI_FIELDS.keys()) + list(PD_FIELDS.keys())
    errors = {f: [] for f in all_fields}

    for leave_out in all_keys:
        # 构建去掉一个点的数据
        reduced = {k: v for k, v in data.items()
                   if not (v.get("batch_size") == leave_out[0]
                           and v.get("seq_len") == leave_out[1])}

        # 训练
        rpi = _extract_phase_independent(reduced)
        for field, (method, x_type) in PI_FIELDS.items():
            coeffs = _train_field(rpi, field, method, x_type)
            if coeffs is None:
                continue
            true_val = pi_points[leave_out].get(field)
            if true_val is None or true_val == 0:
                continue
            pred = _predict_field(coeffs, method, leave_out[0], leave_out[1], x_type)
            errors[field].append(abs(pred - true_val) / abs(true_val))

        for phase in ["prefill", "decode"]:
            rpd = _extract_phase_dependent(reduced, phase)
            orig_pd = _extract_phase_dependent(data, phase)
            if leave_out not in orig_pd:
                continue
            for field, (method, x_type) in PD_FIELDS.items():
                coeffs = _train_field(rpd, field, method, x_type)
                if coeffs is None:
                    continue
                true_val = orig_pd[leave_out].get(field)
                if true_val is None or true_val == 0:
                    continue
                pred = _predict_field(coeffs, method, leave_out[0], leave_out[1], x_type)
                errors[field].append(abs(pred - true_val) / abs(true_val))

    print("\n=== 留一交叉验证 MAPE ===")
    for field, errs in errors.items():
        if errs:
            mape = np.mean(errs) * 100
            print(f"  {field:15s}: {mape:6.1f}%  (n={len(errs)})")
        else:
            print(f"  {field:15s}: N/A")


# ── 趋势图 ──────────────────────────────────────────────

def plot_extrapolation(data, full_data, out_dir):
    """绘制实测点 + 外推曲线。"""
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

    fields_to_plot = ["ci_attn", "ci_ffn", "t_ffn", "g_launch",
                       "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw", "baseline_ms"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for idx, field in enumerate(fields_to_plot):
        ax = axes[idx]
        for phase, marker, color in [("prefill", "o", "#D32F2F"), ("decode", "s", "#1976D2")]:
            # 实测点
            mx, my = [], []
            for key, entry in data.items():
                if entry.get("phase") != phase:
                    continue
                v = entry.get(field)
                if v is not None:
                    mx.append(entry["batch_size"] * entry["seq_len"])
                    my.append(v)
            # 外推点
            ex, ey = [], []
            for key, entry in full_data.items():
                if entry.get("phase") != phase:
                    continue
                if entry.get("source") == "extrapolated":
                    v = entry.get(field)
                    if v is not None:
                        ex.append(entry["batch_size"] * entry["seq_len"])
                        ey.append(v)

            ax.scatter(mx, my, marker=marker, color=color, s=40, zorder=3,
                       label=f"{phase} (实测)" if idx == 0 else "")
            if ex:
                ax.scatter(ex, ey, marker=marker, color=color, s=25, alpha=0.5,
                           facecolors='none', edgecolors=color, zorder=2,
                           label=f"{phase} (外推)" if idx == 0 else "")

        ax.set_xscale("log")
        ax.set_xlabel("b × s")
        ax.set_ylabel(field)
        ax.set_title(field)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("MLWD 参数化外推：实测 vs 外推", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "extrapolation_trends.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLWD 参数化外推")
    parser.add_argument("--input", required=True, help="实测数据 JSON")
    parser.add_argument("--model", required=True, help="模型标识")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--validate", action="store_true", help="留一交叉验证")
    parser.add_argument("--plot", action="store_true", help="生成趋势图")
    parser.add_argument("--plot_dir", default="output/extrapolation_plots")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.validate:
        leave_one_out(data)
        return

    full = extrapolate_full(data, args.model)

    n_measured = sum(1 for v in full.values() if v.get("source") == "measured")
    n_extrap = sum(1 for v in full.values() if v.get("source") == "extrapolated")
    n_complete = sum(1 for v in full.values() if v.get("complete"))
    print(f"Total: {len(full)} entries ({n_measured} measured, {n_extrap} extrapolated, {n_complete} complete)")

    for key in sorted(full.keys()):
        entry = full[key]
        src = entry.get("source", "?")[:4]
        ok = "OK" if entry.get("complete") else "--"
        print(f"  {key:25s} [{src}] {ok}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(full, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    if args.plot:
        plot_extrapolation(data, full, args.plot_dir)


if __name__ == "__main__":
    main()
