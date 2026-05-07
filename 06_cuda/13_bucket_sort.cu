#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void histogramKernel(int* d_key, int* d_bucket, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&d_bucket[d_key[idx]], 1);
    }
}

__global__ void scanKernel(int* d_bucket, int* d_offsets, int range) {
    extern __shared__ int temp[];
    int i = threadIdx.x;

    if (i < range) {
        temp[i] = (i > 0) ? d_bucket[i - 1] : 0;
    }
    __syncthreads();

    for (int j = 1; j < range; j <<= 1) {
        int val = 0;
        if (i >= j) val = temp[i - j];
        __syncthreads();
        if (i >= j) temp[i] += val;
        __syncthreads();
    }

    if (i < range) {
        d_offsets[i] = temp[i];
    }
}

__global__ void scatterKernel(int* d_key, int* d_sorted, int* d_offsets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = d_key[idx];
        int pos = atomicAdd(&d_offsets[val], 1);
        d_sorted[pos] = val;
    }
}

int main() {
    const int n = 50;
    const int range = 5;

    int *key, *bucket, *offsets, *sorted;
    cudaMallocManaged(&key, n * sizeof(int));
    cudaMallocManaged(&bucket, range * sizeof(int));
    cudaMallocManaged(&offsets, range * sizeof(int));
    cudaMallocManaged(&sorted, n * sizeof(int));

    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    for (int i = 0; i < range; i++) bucket[i] = 0;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    histogramKernel<<<blocksPerGrid, threadsPerBlock>>>(key, bucket, n);
    cudaDeviceSynchronize();

    scanKernel<<<1, range, range * sizeof(int)>>>(bucket, offsets, range);
    cudaDeviceSynchronize();

    scatterKernel<<<blocksPerGrid, threadsPerBlock>>>(key, sorted, offsets, n);
    cudaDeviceSynchronize();

    printf("Sorted:   ");
    for (int i = 0; i < n; i++) {
        printf("%d ", sorted[i]);
    }
    printf("\n");

    cudaFree(key);
    cudaFree(bucket);
    cudaFree(offsets);
    cudaFree(sorted);

    return 0;
}
