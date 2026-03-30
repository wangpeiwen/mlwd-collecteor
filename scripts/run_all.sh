#!/bin/bash
# MLWD 全流程一键采集
set -e

MODEL=${MODEL:-/data/Qwen/Qwen2.5-7B-Instruct}
export PYTHONPATH=.

echo "=== Step 1: 编译压力核 ==="
cmake -B build && cmake --build build

echo ""
echo "=== Step 2: 干扰敏感度采集 ==="
python -m mlwd.collect_sensitivity --model ${MODEL}

echo ""
echo "=== Step 3: nsys 采集 ==="
bash scripts/run_nsys_all.sh

echo ""
echo "=== Step 4: CI 估算 ==="
python -m mlwd.collect_ci --model ${MODEL}

echo ""
echo "=== Step 5: 合并数据 ==="
python -m mlwd.merge

echo ""
echo "=== Step 6: 验证 ==="
python -m mlwd.validate

echo ""
echo "=== Step 7: 可视化 ==="
python -m mlwd.visualize

echo ""
echo "=== 完成 ==="
echo "结果: output/mlwd_complete.json"
echo "图表: output/plots/"
