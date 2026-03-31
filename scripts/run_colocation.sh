#!/bin/bash
# 同模型 PD 共置干扰实验
# 双卡并行：卡0 跑 Qwen，卡1 跑 Llama
#
# 每个模型在单个 vLLM 实例内测量 Prefill 注入对 Decode 的干扰
# 不需要 MPS，不会 OOM
#
# Usage:
#   bash scripts/run_colocation.sh [QWEN_PATH] [LLAMA_PATH]

set -e

QWEN=${1:-"/data/Qwen2.5-7B-Instruct"}
LLAMA=${2:-"/data/LLM-Research/Llama-3.2-3B-Instruct"}
NUM_RUNS=5
WARMUP=2

echo "============================================"
echo "  同模型 PD 共置干扰实验"
echo "  GPU 0: Qwen-2.5-7B"
echo "  GPU 1: Llama-3.2-3B"
echo "  Runs: ${NUM_RUNS}, Warmup: ${WARMUP}"
echo "============================================"

mkdir -p output

# 双卡并行
python3 -m mlwd.colocation \
    --model "$QWEN" --gpu 0 \
    --output output/colocation_qwen.json \
    --num_runs $NUM_RUNS --warmup $WARMUP &
PID0=$!

python3 -m mlwd.colocation \
    --model "$LLAMA" --gpu 1 \
    --output output/colocation_llama.json \
    --num_runs $NUM_RUNS --warmup $WARMUP &
PID1=$!

echo "Running: Qwen(PID=$PID0, GPU 0), Llama(PID=$PID1, GPU 1)"
wait $PID0; echo "[GPU 0] Done."
wait $PID1; echo "[GPU 1] Done."

# 合并 + OLS 标定
echo ""
python3 -c "
import json
merged = []
for f in ['output/colocation_qwen.json', 'output/colocation_llama.json']:
    try:
        with open(f) as fh:
            data = json.load(fh)
            merged.extend(data.get('pairs', []))
    except Exception as e:
        print(f'Warning: {f}: {e}')
with open('output/colocation_merged.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(merged)} pairs → output/colocation_merged.json')
"

python3 -m mlwd.colocation_calibrate \
    --colocation output/colocation_merged.json \
    --mlwd output/Qwen-2.5-7B.json output/Llama-3.2-3B.json \
    --output output/weights.json

echo ""
echo "Done! Results in output/weights.json"
