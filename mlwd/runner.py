"""vLLM 推理封装。加载模型一次，支持 prefill/decode 测量。"""

import os, time
from transformers import AutoTokenizer


def load_model(model: str, quantization: str = "fp16", tp: int = 1):
    from vllm import LLM
    llm = LLM(model=model, dtype="float16", tensor_parallel_size=tp,
              trust_remote_code=True, enforce_eager=True, max_model_len=2048)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return llm, tokenizer


def make_prompts(tokenizer, seq_len: int, batch_size: int):
    text = "hello " * (seq_len * 2)
    ids = tokenizer.encode(text)[:seq_len]
    prompt = tokenizer.decode(ids)
    return [prompt] * batch_size


def run_inference(llm, prompts, max_tokens: int, num_runs: int, warmup: int = 2):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0)
    for _ in range(warmup):
        llm.generate(prompts, sp)
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    mid = len(latencies) // 2
    median = latencies[mid] if len(latencies) % 2 else (latencies[mid-1] + latencies[mid]) / 2
    return median, latencies
