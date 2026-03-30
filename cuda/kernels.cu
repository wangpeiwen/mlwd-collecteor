#include "kernels.cuh"

__global__ void sleep_kernel(long long num_itr) {
    for (long long i = 0; i < num_itr; ++i)
        asm volatile("nanosleep.u32 1000;");
}

__global__ void copy_kernel(float *in, float *out, long long num_floats, long long num_itr) {
    size_t start = threadIdx.x + blockDim.x * blockIdx.x;
    size_t step = gridDim.x * blockDim.x;
    for (size_t j = 0; j < num_itr; j++)
        for (size_t i = start; i < num_floats; i += step)
            out[i] = in[i];
}

__global__ void fma_fp32_ilp4(float *a, float *b, float *c, long long num_itr) {
    float op1 = a[threadIdx.x], op2 = b[threadIdx.x];
    float op3 = 0.f, op4 = 0.f, op5 = 0.f, op6 = 0.f;
    for (long long i = 0; i < num_itr; i++) {
        op3 = __fmaf_rn(op1, op2, op3);
        op4 = __fmaf_rn(op1, op2, op4);
        op5 = __fmaf_rn(op1, op2, op5);
        op6 = __fmaf_rn(op1, op2, op6);
    }
    c[threadIdx.x] = op3 + op4 + op5 + op6;
}
