"""OLS 标定：从共置实验数据标定干扰系数加权权重。

将 colocation.json 中的真实 α_d/α_p 与 MLWD 特征对齐，
通过最小二乘法拟合 6 个加权系数。

Usage:
    python -m mlwd.colocation_calibrate \\
        --colocation output/colocation.json \\
        --victim-mlwd output/Qwen-2.5-7B.json \\
        --aggressor-mlwd output/Llama-3.2-3B.json \\
        --output output/weights.json
"""

import argparse, json, os
import numpy as np
from pathlib import Path
from .interference import (build_feature_row, calibrate_weights,
                           DEFAULT_WEIGHTS, DIMS, estimate_alpha_d, estimate_alpha_p)


def _find_mlwd_entry(mlwd_data, b, s, phase):
    """从 MLWD 数据中查找匹配的条目。"""
    key = f"b{b}_s{s}_{phase}"
    return mlwd_data.get(key)


def build_calibration_data(colocation_data, victim_mlwd, aggressor_mlwd):
    """从共置实验数据构建 OLS 训练矩阵。"""
    X_d, y_d = [], []  # α_d 样本
    X_p, y_p = [], []  # α_p 样本

    for sample in colocation_data:
        if sample.get("alpha_d") is None:
            continue

        v_entry = _find_mlwd_entry(victim_mlwd,
                                    sample["victim_b"], sample["victim_s"],
                                    sample["victim_phase"])
        a_entry = _find_mlwd_entry(aggressor_mlwd,
                                    sample["aggressor_b"], sample["aggressor_s"],
                                    sample["aggressor_phase"])
        if v_entry is None or a_entry is None:
            continue

        # α_d: aggressor 对 victim 的干扰
        row_d = build_feature_row(v_entry, a_entry)
        X_d.append(row_d)
        y_d.append(sample["alpha_d"])

        # α_p: victim 对 aggressor 的干扰（角色互换）
        if sample.get("alpha_p") is not None:
            row_p = build_feature_row(a_entry, v_entry)
            X_p.append(row_p)
            y_p.append(sample["alpha_p"])

    return (np.array(X_d) if X_d else np.empty((0, 6)),
            np.array(y_d) if y_d else np.empty(0),
            np.array(X_p) if X_p else np.empty((0, 6)),
            np.array(y_p) if y_p else np.empty(0))


def evaluate_weights(weights, colocation_data, victim_mlwd, aggressor_mlwd):
    """评估标定权重的预测精度。"""
    errors_d, errors_p = [], []

    for sample in colocation_data:
        if sample.get("alpha_d") is None:
            continue
        v = _find_mlwd_entry(victim_mlwd, sample["victim_b"],
                              sample["victim_s"], sample["victim_phase"])
        a = _find_mlwd_entry(aggressor_mlwd, sample["aggressor_b"],
                              sample["aggressor_s"], sample["aggressor_phase"])
        if v is None or a is None:
            continue

        pred_d = estimate_alpha_d(v, a, weights)
        true_d = sample["alpha_d"]
        if true_d != 0:
            errors_d.append(abs(pred_d - true_d) / abs(true_d))

        if sample.get("alpha_p") is not None:
            pred_p = estimate_alpha_p(a, v, weights)
            true_p = sample["alpha_p"]
            if true_p != 0:
                errors_p.append(abs(pred_p - true_p) / abs(true_p))

    return {
        "mape_alpha_d": round(np.mean(errors_d) * 100, 2) if errors_d else None,
        "mape_alpha_p": round(np.mean(errors_p) * 100, 2) if errors_p else None,
        "mae_alpha_d": round(np.mean([abs(e) for e in errors_d]), 4) if errors_d else None,
        "n_samples_d": len(errors_d),
        "n_samples_p": len(errors_p),
    }


def main():
    parser = argparse.ArgumentParser(description="OLS 标定干扰系数权重")
    parser.add_argument("--colocation", required=True, help="共置实验数据")
    parser.add_argument("--victim-mlwd", required=True, help="Victim MLWD JSON")
    parser.add_argument("--aggressor-mlwd", required=True, help="Aggressor MLWD JSON")
    parser.add_argument("--output", default=None, help="输出权重 JSON")
    args = parser.parse_args()

    with open(args.colocation) as f:
        coloc = json.load(f)
    with open(args.victim_mlwd) as f:
        v_mlwd = json.load(f)
    with open(args.aggressor_mlwd) as f:
        a_mlwd = json.load(f)

    print(f"Loaded {len(coloc)} co-location samples")

    X_d, y_d, X_p, y_p = build_calibration_data(coloc, v_mlwd, a_mlwd)
    print(f"Feature matrix: α_d {X_d.shape}, α_p {X_p.shape}")

    if len(X_d) == 0 and len(X_p) == 0:
        print("No valid samples for calibration!")
        return

    # 联合标定
    X = np.vstack([x for x in [X_d, X_p] if len(x) > 0])
    y = np.concatenate([v for v in [y_d, y_p] if len(v) > 0])
    weights = calibrate_weights(X, y)

    print(f"\nCalibrated weights ({len(y)} samples):")
    for k, v in weights.items():
        default = DEFAULT_WEIGHTS.get(k, 0)
        print(f"  {k}: {v:+.6f}  (default: {default:.2f})")

    # 评估
    metrics_cal = evaluate_weights(weights, coloc, v_mlwd, a_mlwd)
    metrics_def = evaluate_weights(DEFAULT_WEIGHTS, coloc, v_mlwd, a_mlwd)
    print(f"\nCalibrated:  MAPE(α_d)={metrics_cal['mape_alpha_d']}%  MAPE(α_p)={metrics_cal['mape_alpha_p']}%")
    print(f"Default:     MAPE(α_d)={metrics_def['mape_alpha_d']}%  MAPE(α_p)={metrics_def['mape_alpha_p']}%")

    result = {
        "weights": weights,
        "metrics_calibrated": metrics_cal,
        "metrics_default": metrics_def,
        "n_samples": len(y),
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
