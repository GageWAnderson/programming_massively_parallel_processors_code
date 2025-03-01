#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define ARRAY_SIZE 1024
#define BLOCK_DIM 64
#define THREADS_PER_BLOCK (ARRAY_SIZE / 2)

__global__ void SimpleSumReductionKernel(float *input, float *output, int n)
{
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if (threadIdx.x % stride == 0 && (i + stride) < n)
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        *output = input[0];
    }
}

__global__ void ConvergedSumReductionKernel(float *input, float *output)
{
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *output = input[0];
    }
}

__global__ void SharedMemorySumReductionKernel(float *input, float *output)
{
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
        {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0)
    {
        *output = input_s[0];
    }
}

__global__ void SegmentedSumReductionKernel(float *input, float *output)
{
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
        {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0)
    {
        atomicAdd(output, input_s[0]);
    }
}

int main()
{
    float *h_input, *d_input, *d_input_copy, *h_output, *d_output;
    cudaEvent_t start, stop;
    float elapsedTimeSimple, elapsedTimeConverged, elapsedTimeShared, elapsedTimeSegmented;

    // Allocate host memory
    h_input = (float *)malloc(ARRAY_SIZE * sizeof(float));
    h_output = (float *)malloc(sizeof(float));

    // Initialize random seed
    srand(time(NULL));

    // Generate random input data
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_input, ARRAY_SIZE * sizeof(float));
    cudaMalloc((void **)&d_input_copy, ARRAY_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time SimpleSumReductionKernel
    cudaMemcpy(d_input_copy, d_input, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaEventRecord(start, 0);
    SimpleSumReductionKernel<<<1, THREADS_PER_BLOCK>>>(d_input_copy, d_output, ARRAY_SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeSimple, start, stop);

    // Time ConvergedSumReductionKernel
    cudaMemcpy(d_input_copy, d_input, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaEventRecord(start, 0);
    ConvergedSumReductionKernel<<<1, THREADS_PER_BLOCK>>>(d_input_copy, d_output);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeConverged, start, stop);

    // Time SharedMemorySumReductionKernel
    cudaMemcpy(d_input_copy, d_input, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaEventRecord(start, 0);
    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input_copy, d_output);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeShared, start, stop);

    // Time SegmentedSumReductionKernel
    cudaMemcpy(d_input_copy, d_input, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start, 0);
    SegmentedSumReductionKernel<<<ARRAY_SIZE / (2 * BLOCK_DIM), BLOCK_DIM>>>(d_input_copy, d_output);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeSegmented, start, stop);

    // Print runtimes comparison
    printf("Kernel runtimes (ms):\n");
    printf("SimpleSumReductionKernel: %f ms\n", elapsedTimeSimple);
    printf("ConvergedSumReductionKernel: %f ms\n", elapsedTimeConverged);
    printf("SharedMemorySumReductionKernel: %f ms\n", elapsedTimeShared);
    printf("SegmentedSumReductionKernel: %f ms\n", elapsedTimeSegmented);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_input_copy);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}