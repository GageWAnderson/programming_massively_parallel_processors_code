#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define ARRAY_LENGTH 1000000000
#define BLOCK_SIZE 64
#define SECTION_SIZE 64

void sequential_scan(float *x, float *y, unsigned int N)
{
    y[0] = x[0];
    for (unsigned int i = 1; i < N; ++i)
    {
        y[i] = y[i - 1] + x[i];
    }
}

__global__ void kogge_stone_scan_kernel(float *X, float *Y, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        XY[threadIdx.x] = X[i];
    }
    else
    {
        XY[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        float temp; // store res in tmp to prevent race conditions
        if (threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }
    if (i < N)
    {
        Y[i] = XY[threadIdx.x];
    }
}

__global__ void brent_kung_scan_kernel(float *X, float *Y, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        XY[threadIdx.x] = X[i];
    if (i + blockDim.x < N)
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE)
        {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE)
        {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < N)
        Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N)
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

// Helper kernel for the first phase of segmented scan
__global__ void block_scan_kernel(float *X, float *Y, float *block_sums, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < N)
        XY[threadIdx.x] = X[i];
    else
        XY[threadIdx.x] = 0.0f;

    if (i + blockDim.x < N)
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    else
        XY[threadIdx.x + blockDim.x] = 0.0f;

    // Perform Brent-Kung scan on this block
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE)
        {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE)
        {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    // Write results back to global memory
    if (i < N)
        Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N)
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];

    // Store the sum of this block for the next phase
    if (threadIdx.x == 0 && block_sums != NULL)
    {
        block_sums[blockIdx.x] = XY[SECTION_SIZE - 1];
    }
}

// Helper kernel to add block sums to each element
__global__ void add_block_sums_kernel(float *Y, float *block_sums, unsigned int N)
{
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0)
    {
        float sum = block_sums[blockIdx.x - 1];
        if (i < N)
            Y[i] += sum;
        if (i + blockDim.x < N)
            Y[i + blockDim.x] += sum;
    }
}

// Remove the __global__ qualifier and make this a host function
void segmented_scan(float *X, float *Y, unsigned int N, float *aux_array, unsigned int M)
{
    // First phase: Scan each block independently and store block sums
    block_scan_kernel<<<M, BLOCK_SIZE>>>(X, Y, aux_array, N);

    // Second phase: Scan the block sums
    if (M > 1)
    {
        // Scan the auxiliary array containing block sums
        block_scan_kernel<<<1, BLOCK_SIZE>>>(aux_array, aux_array, NULL, M);

        // Third phase: Add the scanned block sums back to each block
        add_block_sums_kernel<<<M, BLOCK_SIZE>>>(Y, aux_array, N);
    }
}

int main()
{
    float *h_x, *h_y, *d_x, *d_y;

    // Allocate host memory
    h_x = (float *)malloc(ARRAY_LENGTH * sizeof(float));
    h_y = (float *)malloc(ARRAY_LENGTH * sizeof(float));

    // Initialize input array
    for (unsigned int i = 0; i < ARRAY_LENGTH; ++i)
    {
        h_x[i] = 1.0f;
    }

    // Run sequential scan and time it
    clock_t cpu_start = clock();
    sequential_scan(h_x, h_y, ARRAY_LENGTH);
    clock_t cpu_end = clock();

    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("Sequential scan took %f seconds.\n", cpu_time);

    // Allocate device memory
    cudaMalloc((void **)&d_x, ARRAY_LENGTH * sizeof(float));
    cudaMalloc((void **)&d_y, ARRAY_LENGTH * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, ARRAY_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(SECTION_SIZE);
    dim3 gridDim((ARRAY_LENGTH + SECTION_SIZE - 1) / SECTION_SIZE);

    // Create CUDA events for timing
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    float gpu_time = 0.0f;

    // Run and time the CUDA kernel
    cudaEventRecord(gpu_start);
    kogge_stone_scan_kernel<<<gridDim, blockDim>>>(d_x, d_y, ARRAY_LENGTH);
    cudaEventRecord(gpu_stop);

    // Wait for the kernel to complete
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);

    printf("Kogge-Stone scan (CUDA) took %f seconds.\n", gpu_time / 1000.0f);

    // Run and time the Brent-Kung kernel
    cudaEventRecord(gpu_start);
    brent_kung_scan_kernel<<<gridDim, blockDim>>>(d_x, d_y, ARRAY_LENGTH);
    cudaEventRecord(gpu_stop);

    // Wait for the Brent-Kung kernel to complete
    cudaEventSynchronize(gpu_stop);
    float bk_gpu_time = 0.0f;
    cudaEventElapsedTime(&bk_gpu_time, gpu_start, gpu_stop);

    printf("Brent-Kung scan (CUDA) took %f seconds.\n", bk_gpu_time / 1000.0f);

    // Wait for the segmented scan kernel for arbitrary input sizes to complete
    cudaEventRecord(gpu_start);
    unsigned int num_blocks = (ARRAY_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_aux_array;
    cudaMalloc((void **)&d_aux_array, num_blocks * sizeof(float));

    segmented_scan(d_x, d_y, ARRAY_LENGTH, d_aux_array, num_blocks);

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    float segmented_gpu_time = 0.0f;
    cudaEventElapsedTime(&segmented_gpu_time, gpu_start, gpu_stop);

    printf("Segmented scan (CUDA) took %f seconds.\n", segmented_gpu_time / 1000.0f);

    // Clean up
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_aux_array);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    free(h_x);
    free(h_y);

    return 0;
}