"""实验矩阵、硬件参数、压力核参数。"""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import List

# V100 硬件参数
V100_NUM_SMS = 80
V100_L2_BYTES = 6 * 1024 * 1024
V100_BW_GBS = 900

# Qwen2.5-7B 模型参数（用于理论 FLOPs 计算）
QWEN_HIDDEN = 3584
QWEN_LAYERS = 28
QWEN_HEADS = 28
QWEN_KV_HEADS = 4
QWEN_HEAD_DIM = 128
QWEN_INTER = 18944

DEFAULT_MODEL = "/data/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BATCH_SIZES = [1, 4]
DEFAULT_SEQ_LENGTHS = [32, 64, 128]
OUTPUT_DIR = Path("output")
LIB_PATH = Path("build/cuda/libstress_interface.so")


@dataclass
class StressConfig:
    bs_tb: int = 160;  bs_threads: int = 1024; bs_itrs: int = 100000
    cu_tb: int = 80;   cu_threads: int = 128;  cu_itrs: int = 500000
    l2_tb: int = 40;   l2_threads: int = 1024; l2_bytes: int = V100_L2_BYTES; l2_itrs: int = 10000
    bw_tb: int = 80;   bw_threads: int = 1024; bw_bytes: int = 256*1024*1024; bw_itrs: int = 200


@dataclass
class Experiment:
    model: str = DEFAULT_MODEL
    batch_sizes: List[int] = field(default_factory=lambda: DEFAULT_BATCH_SIZES.copy())
    seq_lengths: List[int] = field(default_factory=lambda: DEFAULT_SEQ_LENGTHS.copy())
    phases: List[str] = field(default_factory=lambda: ["prefill", "decode"])
    num_runs: int = 5
    warmup_runs: int = 2
    max_tokens: int = 32
    stress: StressConfig = field(default_factory=StressConfig)

    def iter_points(self):
        for b, s, p in product(self.batch_sizes, self.seq_lengths, self.phases):
            yield b, s, p

    def total(self):
        return len(self.batch_sizes) * len(self.seq_lengths) * len(self.phases)
