# MLWD Collector

多层次工作负载描述符（Multi-Level Workload Descriptor）离线采集工具。

针对 LLM 推理场景，采集 GPU 共置干扰预测所需的 15 维 MLWD 向量。

## 环境要求

- CUDA 12.5+, CMake 3.22+
- Python 3.8+, vLLM, transformers, matplotlib, numpy
- GPU: V100 (SM 70)

## 快速开始

```bash
# 一键全流程
MODEL=/data/Qwen/Qwen2.5-7B-Instruct bash scripts/run_all.sh
```

## 分步执行

```bash
# 1. 编译压力核
cmake -B build && cmake --build build

# 2. 干扰敏感度 (σ_bs, σ_cu, σ_l2, σ_bw)
PYTHONPATH=. python -m mlwd.collect_sensitivity --model /data/Qwen/Qwen2.5-7B-Instruct

# 3. nsys 执行模式 (t_attn, t_ffn, g_launch, r_attn, r_ffn, f_switch)
bash scripts/run_nsys_all.sh

# 4. CI 估算 (CI_attn, CI_ffn)
PYTHONPATH=. python -m mlwd.collect_ci --model /data/Qwen/Qwen2.5-7B-Instruct

# 5. 合并 + 验证 + 可视化
PYTHONPATH=. python -m mlwd.merge
PYTHONPATH=. python -m mlwd.validate
PYTHONPATH=. python -m mlwd.visualize
```

## 输出

```
output/
├── sensitivity.json       # 四维干扰敏感度
├── nsys.json              # nsys 执行模式特征
├── ci.json                # 计算强度 (CI)
├── mlwd_complete.json     # 合并后的完整 15 维向量
└── plots/                 # 可视化图表
```

## MLWD 15 维向量

| 组 | 特征 | 采集方式 |
|----|------|---------|
| 资源竞争强度 | CI_attn, CI_ffn | vLLM profiler + 理论 FLOPs |
| 资源竞争强度 | L2_attn, L2_ffn | 需要 ncu 权限（暂不支持） |
| 干扰敏感度 | σ_bs, σ_cu, σ_l2, σ_bw | 合成压力核共置实验 |
| 执行模式 | t_attn, t_ffn, g_launch | nsys trace |
| 执行模式 | r_attn, r_ffn, f_switch | nsys trace |
| 执行模式 | IPC | 需要 ncu 权限（暂不支持） |
