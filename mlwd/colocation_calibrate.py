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


def build_calibration_data(colocation_data, mlwd_files):
    """从共置实验数据构建 OLS 训练矩阵。

    mlwd_files: {"model_name": {mlwd_data}} 字典，自动按 victim_model/aggressor_model 匹配。
    """
    X_d, y_d = [], []

    for sample in colocation_data:
        if sample.get("alpha_d") is None:
            continue

        # 查找匹配的 MLWD 数据
        v_mlwd = mlwd_files.get(sample.get("victim_model"))
        a_mlwd = mlwd_files.get(sample.get("aggressor_model"))
        if v_mlwd is None or a_mlwd is None:
            continue

        v_entry = _find_mlwd_entry(v_mlwd,
                                    sample["victim_b"], sample["victim_s"],
                                    sample["victim_phase"])
        a_entry = _find_mlwd_entry(a_mlwd,
                                    sample["aggressor_b"], sample["aggressor_s"],
                                    sample["aggressor_phase"])
        if v_entry is None or a_entry is None:
            continue

        row_d = build_feature_row(v_entry, a_entry)
        X_d.append(row_d)
        y_d.append(sample["alpha_d"])

    return (np.array(X_d) if X_d else np.empty((0, 6)),
            np.array(y_d) if y_d else np.empty(0))


def evaluate_weights(weights, colocation_data, mlwd_files):
    """评估标定权重的预测精度。"""
    errors_d = []

    for sample in colocation_data:
        if sample.get("alpha_d") is None:
            continue
        v_mlwd = mlwd_files.get(sample.get("victim_model"))
        a_mlwd = mlwd_files.get(sample.get("aggressor_model"))
        if v_mlwd is None or a_mlwd is None:
            continue
        v = _find_mlwd_entry(v_mlwd, sample["victim_b"],
                              sample["victim_s"], sample["victim_phase"])
        a = _find_mlwd_entry(a_mlwd, sample["aggressor_b"],
                              sample["aggressor_s"], sample["aggressor_phase"])
        if v is None or a is None:
            continue

        pred_d = estimate_alpha_d(v, a, weights)
        true_d = sample["alpha_d"]
        if true_d != 0:
            errors_d.append(abs(pred_d - true_d) / abs(true_d))

    return {
        "mape_alpha_d": round(np.mean(errors_d) * 100, 2) if errors_d else None,
        "n_samples": len(errors_d),
    }


def main():
    parser = argparse.ArgumentParser(description="OLS 标定干扰系数权重")
    parser.add_argument("--colocation", required=True, help="共置实验数据 JSON")
    parser.add_argument("--mlwd", nargs="+", required=True,
                        help="MLWD JSON 文件（可多个，按模型名自动匹配）")
    parser.add_argument("--output", default=None, help="输出权重 JSON")
    args = parser.parse_args()

    with open(args.colocation) as f:
        coloc = json.load(f)

    # 加载所有 MLWD 文件，key 为文件名 stem
    mlwd_files = {}
    for p in args.mlwd:
        name = Path(p).stem
        with open(p) as f:
            mlwd_files[name] = json.load(f)
    print(f"MLWD files: {list(mlwd_files.keys())}")

    print(f"Loaded {len(coloc)} co-location samples")

    X_d, y_d = build_calibration_data(coloc, mlwd_files)
    print(f"Feature matrix: {X_d.shape}")

    if len(X_d) == 0:
        print("No valid samples for calibration!")
        return

    weights = calibrate_weights(X_d, y_d)

    print(f"\nCalibrated weights ({len(y_d)} samples):")
    for k, v in weights.items():
        default = DEFAULT_WEIGHTS.get(k, 0)
        print(f"  {k}: {v:+.6f}  (default: {default:.2f})")

    metrics_cal = evaluate_weights(weights, coloc, mlwd_files)
    metrics_def = evaluate_weights(DEFAULT_WEIGHTS, coloc, mlwd_files)
    print(f"\nCalibrated:  MAPE(α_d)={metrics_cal['mape_alpha_d']}%  (n={metrics_cal['n_samples']})")
    print(f"Default:     MAPE(α_d)={metrics_def['mape_alpha_d']}%  (n={metrics_def['n_samples']})")

    result = {
        "weights": weights,
        "metrics_calibrated": metrics_cal,
        "metrics_default": metrics_def,
        "n_samples": len(y_d),
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
