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
#include <chrono>

using namespace std;
using namespace nvcuda;

constexpr int TILE_M = 128;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int ACC_ROWS = 2;
constexpr int ACC_COLS = 4;

__global__ __launch_bounds__(128) void kernel(int dim_m, int dim_n, int dim_k,
                       const half *__restrict__ d_a,
                       const half *__restrict__ d_b,
                       float *__restrict__ d_c) {
  const int offset_a_m = TILE_M * blockIdx.x;
  const int offset_b_n = TILE_N * blockIdx.y;
  const int warp_id = threadIdx.x >> 5;

  __shared__ __align__(16) half block_a[TILE_K][TILE_M];
  __shared__ __align__(16) half block_b[TILE_K][TILE_N];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[ACC_ROWS][ACC_COLS];
  #pragma unroll
  for (int r = 0; r < ACC_ROWS; ++r) {
    #pragma unroll
    for (int c = 0; c < ACC_COLS; ++c) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[ACC_ROWS];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[ACC_COLS];

  for (int k = 0; k < dim_k; k += TILE_K) {
    __syncthreads();

    if (warp_id < 2) {
      const int a_col = (warp_id * 32 + (threadIdx.x & 31)) * 2;
      #pragma unroll
      for (int j = 0; j < TILE_K; ++j) {
        *reinterpret_cast<half2 *>(&block_a[j][a_col]) =
            *reinterpret_cast<const half2 *>(&d_a[(k + j) * dim_m + offset_a_m + a_col]);
      }
    }

    if (warp_id >= 2) {
      const int b_col = (warp_id - 2) * 32 + (threadIdx.x & 31);
      #pragma unroll
      for (int j = 0; j < TILE_K; ++j) {
        block_b[j][b_col] = d_b[(offset_b_n + b_col) * dim_k + k + j];
      }
    }

    __syncthreads();

    #pragma unroll
    for (int c = 0; c < ACC_COLS; ++c) {
      wmma::load_matrix_sync(b_frag[c], &block_b[0][c * 16], TILE_N);
    }

    #pragma unroll
    for (int r = 0; r < ACC_ROWS; ++r) {
      const int row_tile = warp_id * ACC_ROWS + r;
      wmma::load_matrix_sync(a_frag[r], &block_a[0][row_tile * 16], TILE_M);
      #pragma unroll
      for (int c = 0; c < ACC_COLS; ++c) {
        wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
      }
    }
  }

  #pragma unroll
  for (int r = 0; r < ACC_ROWS; ++r) {
    #pragma unroll
    for (int c = 0; c < ACC_COLS; ++c) {
      const int c_m = offset_a_m + (warp_id * ACC_ROWS + r) * 16;
      const int c_n = offset_b_n + c * 16;
      wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
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
      C[m * i + j] = C2[m * i + j] = 0;
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
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
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

  dim3 block = dim3(128);
  dim3 grid = dim3((m + TILE_M - 1) / TILE_M, (n + TILE_N - 1) / TILE_N);
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<<grid, block>>>(m,
			    n,
			    k,
			    A16,
			    B16,
			    C2);
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
