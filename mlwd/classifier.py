"""Kernel 名称分类：Attention / FFN / Other。"""

import re
from enum import Enum


class Cat(Enum):
    ATTN = "attention"
    FFN = "ffn"
    OTHER = "other"


_ATTN = [re.compile(p, re.I) for p in [
    r"flash_attn", r"fmha", r"paged_attention", r"unified_attention",
    r"kernel.*attention", r"attention.*kernel", r"scaled_dot_product",
    r"flash.*fwd", r"reshape_and_cache", r"reduce_segments",
]]
_FFN = [re.compile(p, re.I) for p in [
    r"gemm", r"cublas.*gemm", r"cutlass.*gemm", r"sgemm", r"hgemm",
    r"volta.*gemm", r"sm70.*gemm", r"linear", r"mlp.*kernel",
]]
_ATTN_GEMM = [re.compile(p, re.I) for p in [
    r"attn.*gemm", r"attention.*gemm", r"qkv.*gemm",
]]


def classify(name: str) -> Cat:
    for p in _ATTN_GEMM:
        if p.search(name): return Cat.ATTN
    for p in _ATTN:
        if p.search(name): return Cat.ATTN
    for p in _FFN:
        if p.search(name): return Cat.FFN
    return Cat.OTHER
