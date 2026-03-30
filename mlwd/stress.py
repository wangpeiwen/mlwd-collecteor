"""ctypes 封装 4 类合成压力核。"""

import ctypes
from pathlib import Path
from .config import LIB_PATH


class StressKernels:
    def __init__(self, lib_path: str = None):
        path = lib_path or str(LIB_PATH)
        self.lib = ctypes.CDLL(path)
        for name, args in [
            ("run_bs_stress", [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_int]),
            ("run_cu_stress", [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_int]),
            ("run_l2_stress", [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]),
            ("run_bw_stress", [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]),
        ]:
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = None

    def bs(self, cfg):
        self.lib.run_bs_stress(cfg.bs_tb, cfg.bs_threads, cfg.bs_itrs, 1)

    def cu(self, cfg):
        self.lib.run_cu_stress(cfg.cu_tb, cfg.cu_threads, cfg.cu_itrs, 1)

    def l2(self, cfg):
        self.lib.run_l2_stress(cfg.l2_tb, cfg.l2_threads, cfg.l2_itrs, cfg.l2_bytes, 1)

    def bw(self, cfg):
        self.lib.run_bw_stress(cfg.bw_tb, cfg.bw_threads, cfg.bw_itrs, cfg.bw_bytes, 1)
