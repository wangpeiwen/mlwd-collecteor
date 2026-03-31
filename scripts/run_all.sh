#!/bin/bash
# MLWD 全流程一键采集
set -e

MODEL=${MODEL:-/data/Qwen/Qwen2.5-7B-Instruct}
export PYTHONPATH=.

# 从模型路径提取短名作为输出目录
MODEL_NAME=$(basename ${MODEL})
OUT_DIR="output/${MODEL_NAME}"
mkdir -p ${OUT_DIR}

echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}/"
echo ""

echo "=== Step 1: 编译压力核 ==="
cmake -B build && cmake --build build

echo ""
echo "=== Step 2: 干扰敏感度采集 ==="
python -m mlwd.collect_sensitivity --model ${MODEL} --output ${OUT_DIR}/sensitivity.json

echo ""
echo "=== Step 3: nsys 采集 ==="
MODEL=${MODEL} OUT_DIR=${OUT_DIR} bash scripts/run_nsys_all.sh

echo ""
echo "=== Step 4: CI 估算 ==="
python -m mlwd.collect_ci --model ${MODEL} --output ${OUT_DIR}/ci.json

echo ""
echo "=== Step 5: 合并数据 ==="
python -m mlwd.merge --dir ${OUT_DIR}

echo ""
echo "=== Step 6: 验证 ==="
python -m mlwd.validate --dir ${OUT_DIR}

echo ""
echo "=== Step 7: 可视化 ==="
python -m mlwd.visualize --data ${OUT_DIR}/mlwd_complete.json --output ${OUT_DIR}/plots

echo ""
echo "=== 完成 ==="
echo "结果: ${OUT_DIR}/mlwd_complete.json"
echo "图表: ${OUT_DIR}/plots/"
