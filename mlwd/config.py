"""实验矩阵、硬件参数、压力核参数、模型结构参数。"""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import List, Optional

# V100 硬件参数
V100_NUM_SMS = 80
V100_L2_BYTES = 6 * 1024 * 1024
V100_BW_GBS = 900

# 模型结构参数（用于理论 FLOPs 计算）
MODEL_PARAMS = {
    "qwen2.5-7b": {
        "hidden": 3584, "layers": 28, "heads": 28,
        "kv_heads": 4, "head_dim": 128, "inter": 18944,
    },
    "llama-3.2-3b": {
        "hidden": 3072, "layers": 28, "heads": 24,
        "kv_heads": 8, "head_dim": 128, "inter": 8192,
    },
    "llama-2-7b": {
        "hidden": 4096, "layers": 32, "heads": 32,
        "kv_heads": 32, "head_dim": 128, "inter": 11008,
    },
}


def get_model_params(model_path: str) -> dict:
    """从模型路径推断模型结构参数。"""
    path_lower = model_path.lower()
    for key, params in MODEL_PARAMS.items():
        if key.replace("-", "").replace(".", "") in path_lower.replace("-", "").replace(".", ""):
            return params
    # 未知模型，尝试从 config.json 读取
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        return {
            "hidden": cfg.get("hidden_size", 4096),
            "layers": cfg.get("num_hidden_layers", 32),
            "heads": cfg.get("num_attention_heads", 32),
            "kv_heads": cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 32)),
            "head_dim": cfg.get("head_dim", 128),
            "inter": cfg.get("intermediate_size", 11008),
        }
    raise ValueError(f"Unknown model: {model_path}. Add to MODEL_PARAMS or provide config.json.")


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
