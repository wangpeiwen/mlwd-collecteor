#pragma once

__global__ void sleep_kernel(long long num_itr);
__global__ void copy_kernel(float *in, float *out, long long num_floats, long long num_itr);
__global__ void fma_fp32_ilp4(float *a, float *b, float *c, long long num_itr);
