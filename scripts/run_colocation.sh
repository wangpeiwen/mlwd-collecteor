#!/bin/bash
# 共置干扰实验运行脚本
# 双卡并行：卡0 Qwen(victim) vs Llama(aggressor)，卡1 角色互换
#
# 前置条件：
#   1. 启动 MPS daemon: nvidia-cuda-mps-control -d
#   2. 确认两张 V100-32GB 可用: nvidia-smi
#
# Usage:
#   bash scripts/run_colocation.sh [QWEN_PATH] [LLAMA_PATH]

set -e

QWEN=${1:-"/data/Qwen2.5-7B-Instruct"}
LLAMA=${2:-"/data/LLM-Research/Llama-3.2-3B-Instruct"}
NUM_RUNS=5
WARMUP=2

echo "============================================"
echo "  共置干扰实验"
echo "  Victim/Aggressor: Qwen-2.5-7B / Llama-3.2-3B"
echo "  GPU: 0, 1 (parallel)"
echo "  Runs: ${NUM_RUNS}, Warmup: ${WARMUP}"
echo "============================================"

# MPS 日志目录（避免权限 warning）
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-pipe}
mkdir -p "$CUDA_MPS_LOG_DIRECTORY" "$CUDA_MPS_PIPE_DIRECTORY"

# 检查 MPS
if ! nvidia-cuda-mps-control status 2>/dev/null; then
    echo "[INFO] 启动 MPS daemon..."
    nvidia-cuda-mps-control -d 2>/dev/null || echo "[WARN] MPS 启动失败，共置实验可能不准确"
fi

mkdir -p output

# 卡0: Qwen(decode) vs Llama(prefill)
echo ""
echo "[GPU 0] Qwen(victim/decode) vs Llama(aggressor/prefill)"
python3 -m mlwd.colocation \
    --victim "$QWEN" \
    --aggressor "$LLAMA" \
    --gpu 0 \
    --output output/colocation_qwen_victim.json \
    --num_runs $NUM_RUNS --warmup $WARMUP &
PID0=$!

# 卡1: Llama(decode) vs Qwen(prefill)
echo "[GPU 1] Llama(victim/decode) vs Qwen(aggressor/prefill)"
python3 -m mlwd.colocation \
    --victim "$LLAMA" \
    --aggressor "$QWEN" \
    --gpu 1 \
    --output output/colocation_llama_victim.json \
    --num_runs $NUM_RUNS --warmup $WARMUP &
PID1=$!

echo ""
echo "Running in parallel: PID0=$PID0 (GPU 0), PID1=$PID1 (GPU 1)"
echo "Monitor: tail -f output/colocation_*.json"

wait $PID0
echo "[GPU 0] Done."
wait $PID1
echo "[GPU 1] Done."

# 合并结果
echo ""
echo "Merging results..."
python3 -c "
import json
merged = []
for f in ['output/colocation_qwen_victim.json', 'output/colocation_llama_victim.json']:
    try:
        with open(f) as fh:
            data = json.load(fh)
            merged.extend([r for r in data if r.get('alpha_d') is not None])
    except: pass
with open('output/colocation.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(merged)} co-location pairs → output/colocation.json')
"

# OLS 标定
echo ""
echo "Running OLS calibration..."
python3 -m mlwd.colocation_calibrate \
    --colocation output/colocation.json \
    --victim-mlwd output/Qwen-2.5-7B.json \
    --aggressor-mlwd output/Llama-3.2-3B.json \
    --output output/weights.json

echo ""
echo "============================================"
echo "  Done! Results:"
echo "    output/colocation.json       - 共置实验数据"
echo "    output/weights.json          - OLS 标定权重"
echo "============================================"
