#!/bin/bash
# 共置干扰实验 v2 — 安全单卡方案
#
# 架构：
#   Phase 1: vLLM 加载 victim → 测 baseline → 卸载
#   Phase 2: PyTorch 加载 aggressor + vLLM 加载 victim(util=0.5) → 测共置 → 卸载
#   Phase 3: vLLM 加载 aggressor → 测 baseline → 卸载
#
# 显存安全：任何时刻最多一个 vLLM 实例，Phase 2 中 aggressor 用 PyTorch (~7GB)
#
# 双卡并行：卡0 和 卡1 各跑一组（角色互换），互不干扰
#
# Usage:
#   bash scripts/run_colocation.sh [QWEN_PATH] [LLAMA_PATH]

set -e

QWEN=${1:-"/data/Qwen2.5-7B-Instruct"}
LLAMA=${2:-"/data/LLM-Research/Llama-3.2-3B-Instruct"}
NUM_RUNS=5
WARMUP=2

echo "============================================"
echo "  共置干扰实验 v2 (安全单卡方案)"
echo "  GPU 0: Qwen(victim) vs Llama(aggressor)"
echo "  GPU 1: Llama(victim) vs Qwen(aggressor)"
echo "  Runs: ${NUM_RUNS}, Warmup: ${WARMUP}"
echo "============================================"

mkdir -p output

# 卡0: Qwen(victim/decode) vs Llama(aggressor/prefill)
echo ""
echo "[GPU 0] Qwen(victim) vs Llama(aggressor)"
python3 -m mlwd.colocation \
    --victim "$QWEN" \
    --aggressor "$LLAMA" \
    --gpu 0 \
    --output output/colocation_qwen_victim.json \
    --num_runs $NUM_RUNS --warmup $WARMUP &
PID0=$!

# 卡1: Llama(victim/decode) vs Qwen(aggressor/prefill)
echo "[GPU 1] Llama(victim) vs Qwen(aggressor)"
python3 -m mlwd.colocation \
    --victim "$LLAMA" \
    --aggressor "$QWEN" \
    --gpu 1 \
    --output output/colocation_llama_victim.json \
    --num_runs $NUM_RUNS --warmup $WARMUP &
PID1=$!

echo ""
echo "Running in parallel: PID0=$PID0 (GPU 0), PID1=$PID1 (GPU 1)"
echo "Monitor progress:"
echo "  tail -f output/colocation_qwen_victim.json"
echo "  tail -f output/colocation_llama_victim.json"

wait $PID0
STATUS0=$?
echo "[GPU 0] Done (exit=$STATUS0)"

wait $PID1
STATUS1=$?
echo "[GPU 1] Done (exit=$STATUS1)"

# 合并 + OLS 标定
echo ""
echo "Running OLS calibration..."
python3 -c "
import json, sys
sys.path.insert(0, '.')
from mlwd.colocation_calibrate import main as calibrate_main

# 合并 pairs
merged_pairs = []
for f in ['output/colocation_qwen_victim.json', 'output/colocation_llama_victim.json']:
    try:
        with open(f) as fh:
            data = json.load(fh)
            pairs = data.get('pairs', [])
            merged_pairs.extend([p for p in pairs if p.get('alpha_d') is not None])
    except Exception as e:
        print(f'Warning: {f}: {e}')

print(f'Merged {len(merged_pairs)} valid pairs')
with open('output/colocation_merged.json', 'w') as f:
    json.dump(merged_pairs, f, indent=2)
print('Saved output/colocation_merged.json')
"

python3 -m mlwd.colocation_calibrate \
    --colocation output/colocation_merged.json \
    --mlwd output/Qwen-2.5-7B.json output/Llama-3.2-3B.json \
    --output output/weights.json

echo ""
echo "============================================"
echo "  Done!"
echo "  output/colocation_qwen_victim.json"
echo "  output/colocation_llama_victim.json"
echo "  output/colocation_merged.json"
echo "  output/weights.json"
echo "============================================"
