#!/bin/bash
# nsys 批量采集：每个 (b,s) 单独跑一次 nsys
set -e

NSYS=${NSYS:-/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys}
MODEL=${MODEL:-/data/Qwen/Qwen2.5-7B-Instruct}
TRACE_DIR=/tmp/mlwd_nsys
BATCH_SIZES=(1 4)
SEQ_LENGTHS=(32 64 128)

mkdir -p ${TRACE_DIR} output

for b in "${BATCH_SIZES[@]}"; do
  for s in "${SEQ_LENGTHS[@]}"; do
    key="b${b}_s${s}"
    echo "=== ${key} ==="

    ${NSYS} profile -o ${TRACE_DIR}/${key} --trace cuda,nvtx \
      --trace-fork-before-exec=true --cuda-graph-trace=node \
      --sample none --cpuctxsw none --force-overwrite true \
      python -m mlwd.collect_nsys --profile --model ${MODEL} \
        --batch_sizes ${b} --seq_lengths ${s} 2>&1 | tail -5

    ${NSYS} export --type sqlite --output ${TRACE_DIR}/${key}.sqlite \
      --force-overwrite true ${TRACE_DIR}/${key}.nsys-rep 2>&1 | tail -2

    PYTHONPATH=. python -m mlwd.collect_nsys --parse ${TRACE_DIR}/${key}.sqlite --key ${key}

    rm -f ${TRACE_DIR}/${key}.nsys-rep
    echo "${key} done."
  done
done

echo "All nsys done. Results: output/nsys.json"
