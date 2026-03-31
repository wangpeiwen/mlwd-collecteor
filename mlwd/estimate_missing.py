"""理论估算缺失的 MLWD 字段：L2 命中率、IPC。

当 Nsight Compute 不可用时，基于模型架构参数和硬件规格进行理论估算。

Usage:
    python -m mlwd.estimate_missing --input output/Qwen-2.5-7B.json --model qwen2.5-7b
"""

import argparse, json, os
from .config import V100_L2_BYTES, V100_BW_GBS, V100_NUM_SMS, get_model_params

# V100 FP16 Tensor Core 峰值 15.7 TFLOPS
V100_PEAK_TFLOPS = 15.7
# Roofline balance point: peak_flops / peak_bandwidth
BALANCE_POINT = V100_PEAK_TFLOPS * 1e12 / (V100_BW_GBS * 1e9)  # ≈17.44 FLOP/Byte
# V100 每 SM 4 个 warp scheduler
MAX_IPC = 4.0
# L2 有效容量系数（其他数据竞争缓存行）
L2_EFF = 0.85


def estimate_l2_attn(b, s, mp):
    """估算 Attention Kernel 的 L2 命中率。

    工作集主要为 KV cache：2(K+V) * kv_heads * head_dim * seq_len * batch * 2(FP16)。
    GQA 模型 kv_heads 少，工作集小，命中率高。
    """
    kv_heads = mp["kv_heads"]
    head_dim = mp["head_dim"]
    working_set = 2 * kv_heads * head_dim * s * b * 2  # bytes
    if working_set <= 0:
        return 1.0
    return round(min(1.0, L2_EFF * V100_L2_BYTES / working_set), 4)


def estimate_l2_ffn(b, s, mp):
    """估算 FFN Kernel 的 L2 命中率。

    FFN 权重矩阵远超 L2 容量（gate+up+down 三个投影），命中率极低。
    """
    weight_bytes = 3 * mp["hidden"] * mp["inter"] * 2  # FP16
    activation_bytes = b * s * mp["hidden"] * 2
    total = weight_bytes + activation_bytes
    if total <= 0:
        return 1.0
    return round(min(1.0, L2_EFF * V100_L2_BYTES / total), 4)


def estimate_ipc(ci_attn, ci_ffn, r_attn, r_ffn):
    """基于 Roofline 模型估算 IPC。

    加权 CI 反映整体计算/访存比，CI 越高 pipeline 越饱和，IPC 越接近上限。
    """
    denom = r_attn + r_ffn
    if denom <= 0:
        return round(MAX_IPC * 0.1, 4)
    ci_weighted = (ci_attn * r_attn + ci_ffn * r_ffn) / denom
    saturation = min(1.0, ci_weighted / BALANCE_POINT)
    return round(MAX_IPC * saturation, 4)


def patch_entry(entry, mp):
    """为单条 MLWD 记录补充 l2_attn, l2_ffn, ipc。"""
    b = entry["batch_size"]
    s = entry["seq_len"]
    entry["l2_attn"] = estimate_l2_attn(b, s, mp)
    entry["l2_ffn"] = estimate_l2_ffn(b, s, mp)
    entry["ipc"] = estimate_ipc(
        entry.get("ci_attn", 0), entry.get("ci_ffn", 0),
        entry.get("r_attn", 0), entry.get("r_ffn", 0),
    )
    return entry


def patch_json(json_path, model_key):
    """读取 JSON 文件，补充缺失字段后写回。"""
    mp = get_model_params(model_key)
    with open(json_path) as f:
        data = json.load(f)

    for key, entry in data.items():
        patch_entry(entry, mp)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Patched {len(data)} entries in {json_path}")
    for key, entry in data.items():
        print(f"  {key:25s} l2_attn={entry['l2_attn']:.4f}  l2_ffn={entry['l2_ffn']:.4f}  ipc={entry['ipc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="估算缺失的 MLWD 字段")
    parser.add_argument("--input", required=True, help="JSON 文件路径")
    parser.add_argument("--model", required=True, help="模型标识（如 qwen2.5-7b）")
    args = parser.parse_args()
    patch_json(args.input, args.model)


if __name__ == "__main__":
    main()
