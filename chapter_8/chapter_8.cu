#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define OUT_TILE_DIM 4
#define IN_TILE_DIM (OUT_TILE_DIM + 1)

// Stencil filters
__constant__ int c0, c1, c2, c3, c4, c5, c6 = 1;

__global__ void stencil_kernel(float *in, float *out, unsigned int N)
{
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)
    {
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] +
                                     c1 * in[i * N * N + j * N + (k - 1)] +
                                     c2 * in[i * N * N + j * N + (k + 1)] +
                                     c3 * in[i * N * N + (j - 1) * N + k] +
                                     c4 * in[i * N * N + (j + 1) * N + k] +
                                     c5 * in[(i - 1) * N * N + j * N + k] +
                                     c6 * in[(i + 1) * N * N + j * N + k];
    }
}

// 1.5 OOMs faster than naive approach through reducing global memory traffic
__global__ void stencil_kernel_register_tiling(float *in, float *out, unsigned int N)
{
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    float inPrev;

    // Register variables for register tiling optimization
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr;
    float inNext;

    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i)
    {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N)
        {
            inNext = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)
        {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1)
            {
                out[i * N * N + j * N + k] = c0 * inCurr +
                                             c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                                             c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
                                             c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                                             c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
                                             c5 * inPrev +
                                             c6 * inNext;
            }
        }
        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

int main()
{
    unsigned int N = 256; // Size of the 3D matrix
    size_t size = N * N * N * sizeof(float);

    // Allocate host memory
    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    float *h_out_ref = (float *)malloc(size);

    // Initialize input data with random values
    for (int i = 0; i < N * N * N; i++)
    {
        h_in[i] = rand() % 100 / 10.0f;
    }

    // Allocate device memory
    float *d_in, *d_out, *d_out_ref;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    cudaMalloc((void **)&d_out_ref, size);

    // Copy input data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Launch the stencil kernel
    cudaEvent_t start, stop;
    float elapsedTime;

    // Time the stencil_kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out_ref, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for stencil_kernel: %f ms\n", elapsedTime);

    // Time the stencil_kernel_register_tiling
    cudaEventRecord(start, 0);

    stencil_kernel_register_tiling<<<dimGrid, dimBlock>>>(d_in, d_out, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for stencil_kernel_register_tiling: %f ms\n", elapsedTime);

    // Copy output data back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_ref, d_out_ref, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_ref);

    // Free host memory
    free(h_in);
    free(h_out);
    free(h_out_ref);

    return 0;
}