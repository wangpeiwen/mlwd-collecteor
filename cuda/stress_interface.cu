#include <stdio.h>
#include <cuda_runtime.h>
#include "kernels.cuh"

#define CUDACHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// ── Block Scheduler 压力核 ──
extern "C" void run_bs_stress(int num_tb, int num_threads, long long num_itrs, int num_runs) {
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int i = 0; i < num_runs; i++)
        sleep_kernel<<<num_tb, num_threads, 0, stream>>>(num_itrs);
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
}

// ── Compute Unit 压力核 ──
extern "C" void run_cu_stress(int num_tb, int num_threads, long long num_itrs, int num_runs) {
    size_t sz = num_threads * sizeof(float);
    float *h_a = (float*)malloc(sz), *h_b = (float*)malloc(sz);
    float *d_a, *d_b, *d_c;
    CUDACHECK(cudaMalloc(&d_a, sz));
    CUDACHECK(cudaMalloc(&d_b, sz));
    CUDACHECK(cudaMalloc(&d_c, sz));
    for (int i = 0; i < num_threads; i++) { h_a[i] = 1.f; h_b[i] = 1.f; }
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDACHECK(cudaMemcpyAsync(d_a, h_a, sz, cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaMemcpyAsync(d_b, h_b, sz, cudaMemcpyHostToDevice, stream));
    for (int i = 0; i < num_runs; i++)
        fma_fp32_ilp4<<<num_tb, num_threads, 0, stream>>>(d_a, d_b, d_c, num_itrs);
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
    free(h_a); free(h_b);
    CUDACHECK(cudaFree(d_a)); CUDACHECK(cudaFree(d_b)); CUDACHECK(cudaFree(d_c));
}

// ── L2 Cache 压力核 ──
extern "C" void run_l2_stress(int num_tb, int num_threads, long long num_itrs, long long num_bytes, int num_runs) {
    float *d_in, *d_out;
    CUDACHECK(cudaMalloc(&d_in, num_bytes));
    CUDACHECK(cudaMalloc(&d_out, num_bytes));
    CUDACHECK(cudaMemset(d_in, 0, num_bytes));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int i = 0; i < num_runs; i++)
        copy_kernel<<<num_tb, num_threads, 0, stream>>>(d_in, d_out, num_bytes / (long long)sizeof(float), num_itrs);
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(d_in)); CUDACHECK(cudaFree(d_out));
}

// ── Memory Bandwidth 压力核 ──
extern "C" void run_bw_stress(int num_tb, int num_threads, long long num_itrs, long long num_bytes, int num_runs) {
    float *d_in, *d_out;
    CUDACHECK(cudaMalloc(&d_in, num_bytes));
    CUDACHECK(cudaMalloc(&d_out, num_bytes));
    CUDACHECK(cudaMemset(d_in, 0, num_bytes));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int i = 0; i < num_runs; i++)
        copy_kernel<<<num_tb, num_threads, 0, stream>>>(d_in, d_out, num_bytes / (long long)sizeof(float), num_itrs);
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(d_in)); CUDACHECK(cudaFree(d_out));
}
