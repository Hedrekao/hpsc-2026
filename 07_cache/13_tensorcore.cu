#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <chrono>

using namespace std;
using namespace nvcuda;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 64;

// Upgraded from 2 to 4 stages to fully hide H100 global memory latency
constexpr int STAGES = 4;
constexpr int SMEM_PAD = 8;

// Macro to cleanly issue asynchronous global-to-shared copies.
// _Pragma("unroll") ensures the compiler vectorizes the memory addresses perfectly.
#define LOAD_TILE(STAGE, KB, block_row, block_col) \
do { \
    half* stageA = smemA + (STAGE) * BLOCK_K * (BLOCK_M + SMEM_PAD); \
    half* stageB = smemB + (STAGE) * BLOCK_N * (BLOCK_K + SMEM_PAD); \
    _Pragma("unroll") \
    for (int idx = threadIdx.x; idx < (BLOCK_M * BLOCK_K) / 8; idx += THREADS_PER_BLOCK) { \
        int local_row = (idx * 8) % BLOCK_M; \
        int local_col = (idx * 8) / BLOCK_M; \
        __pipeline_memcpy_async(&stageA[local_col * (BLOCK_M + SMEM_PAD) + local_row], \
                                &A[((KB) + local_col) * dim_m + ((block_row) + local_row)], 16); \
    } \
    _Pragma("unroll") \
    for (int idx = threadIdx.x; idx < (BLOCK_K * BLOCK_N) / 8; idx += THREADS_PER_BLOCK) { \
        int local_row = (idx * 8) % BLOCK_K; \
        int local_col = (idx * 8) / BLOCK_K; \
        __pipeline_memcpy_async(&stageB[local_col * (BLOCK_K + SMEM_PAD) + local_row], \
                                &B[((block_col) + local_col) * dim_k + ((KB) + local_row)], 16); \
    } \
    __pipeline_commit(); \
} while(0)

// Macro to handle the Tensor Core math loop.
#define COMPUTE_TILE() \
do { \
    _Pragma("unroll") \
    for (int kk = 0; kk < BLOCK_K; kk += 16) { \
        _Pragma("unroll") \
        for (int i = 0; i < 2; i++) { \
            int a_row = warp_row * WARP_TILE_M + i * 16; \
            wmma::load_matrix_sync(a_frag[i], &computeA[kk * (BLOCK_M + SMEM_PAD) + a_row], BLOCK_M + SMEM_PAD); \
        } \
        _Pragma("unroll") \
        for (int j = 0; j < 4; j++) { \
            int b_col = warp_col * WARP_TILE_N + j * 16; \
            wmma::load_matrix_sync(b_frag[j], &computeB[b_col * (BLOCK_K + SMEM_PAD) + kk], BLOCK_K + SMEM_PAD); \
        } \
        _Pragma("unroll") \
        for (int i = 0; i < 2; i++) { \
            _Pragma("unroll") \
            for (int j = 0; j < 4; j++) { \
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]); \
            } \
        } \
    } \
} while(0)

// launch bounds force 2 blocks per SM to ensure high occupancy with the new shared memory usage
__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void kernel(
    int dim_m,
    int dim_n,
    int dim_k,
    const half *__restrict__ A,
    const half *__restrict__ B,
    float *__restrict__ C)
{
    extern __shared__ half smem[];

    half* smemA = smem;
    half* smemB = smem + STAGES * BLOCK_K * (BLOCK_M + SMEM_PAD);

    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;

    const int block_row = blockIdx.x * BLOCK_M;
    const int block_col = blockIdx.y * BLOCK_N;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];

    // --- Prologue: Fill the pipeline ---
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        LOAD_TILE(stage, stage * BLOCK_K, block_row, block_col);
    }

    int write_stage = STAGES - 1;
    int read_stage = 0;

    // --- Main Pipeline Loop ---
    for (int kb = (STAGES - 1) * BLOCK_K; kb < dim_k; kb += BLOCK_K)
    {
        // 1. Asynchronously issue hardware copies for the NEXT tile
        LOAD_TILE(write_stage, kb, block_row, block_col);

        // 2. Wait until only STAGES-1 requests are in flight (meaning the oldest one has landed)
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        // 3. Compute the CURRENT tile
        half* computeA = smemA + read_stage * BLOCK_K * (BLOCK_M + SMEM_PAD);
        half* computeB = smemB + read_stage * BLOCK_N * (BLOCK_K + SMEM_PAD);

        COMPUTE_TILE();

        write_stage = (write_stage + 1) % STAGES;
        read_stage = (read_stage + 1) % STAGES;
    }

    // --- Epilogue: Drain the pipeline ---
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        __pipeline_wait_prior(STAGES - 2 - stage);
        __syncthreads();

        half* computeA = smemA + read_stage * BLOCK_K * (BLOCK_M + SMEM_PAD);
        half* computeB = smemB + read_stage * BLOCK_N * (BLOCK_K + SMEM_PAD);

        COMPUTE_TILE();

        read_stage = (read_stage + 1) % STAGES;
    }

    // --- Store Out ---
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int c_row = block_row + warp_row * WARP_TILE_M + i * 16;
            int c_col = block_col + warp_col * WARP_TILE_N + j * 16;

            if (c_row < dim_m && c_col < dim_n)
            {
                wmma::store_matrix_sync(
                    &C[c_col * dim_m + c_row],
                    acc[i][j],
                    dim_m,
                    wmma::mem_col_major);
            }
        }
    }
}

int main(int argc, const char **argv) {
    int m = 10240;
    int k = 4096;
    int n = 8192;

    float alpha = 1.0;
    float beta = 0.0;
    int Nt = 10;

    float *A, *B, *C, *C2;
    half *A16, *B16;

    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C, m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    cudaMallocManaged(&A16, m * k * sizeof(half));
    cudaMallocManaged(&B16, k * n * sizeof(half));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A[k * i + j] = drand48();
            A16[k * i + j] = __float2half_rn(A[k * i + j]);
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B[n * i + j] = drand48();
            B16[n * i + j] = __float2half_rn(B[n * i + j]);
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[m * i + j] = 0;
            C2[m * i + j] = 0;
        }
    }

    cudaMemLocation device_loc;
    device_loc.type = cudaMemLocationTypeDevice;
    device_loc.id = 0;

    cudaMemPrefetchAsync(A, m * k * sizeof(float), device_loc, 0, 0);
    cudaMemPrefetchAsync(B, k * n * sizeof(float), device_loc, 0, 0);
    cudaMemPrefetchAsync(C, m * n * sizeof(float), device_loc, 0, 0);
    cudaMemPrefetchAsync(C2, m * n * sizeof(float), device_loc, 0, 0);
    cudaMemPrefetchAsync(A16, m * k * sizeof(half), device_loc, 0, 0);
    cudaMemPrefetchAsync(B16, k * n * sizeof(half), device_loc, 0, 0);
    cudaDeviceSynchronize();

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m, n, k,
            &alpha,
            A, CUDA_R_32F, m,
            B, CUDA_R_32F, k,
            &beta,
            C, CUDA_R_32F, m,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();

    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);

    size_t smem_size = STAGES * BLOCK_K * (BLOCK_M + SMEM_PAD) * sizeof(half) +
                       STAGES * BLOCK_N * (BLOCK_K + SMEM_PAD) * sizeof(half);

    // CRITICAL FIX: Hopper defaults to 48KB of dynamic shared memory.
    // A 4-stage pipeline requires ~75KB. You must request the higher limit explicitly.
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel<<<grid, block, smem_size>>>(m, n, k, A16, B16, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();

    double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
    double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;

    printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);

    double err = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            err += fabs(C[m * i + j] - C2[m * i + j]);
        }
    }
    printf("error: %lf\n", err / n / m);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C2);
    cudaFree(A16);
    cudaFree(B16);
    cublasDestroy(cublas_handle);
}
