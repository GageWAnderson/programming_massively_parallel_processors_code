#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_BINS 26
#define COARSE_FACTOR 4

__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            atomicAdd(&(histo[blockIdx.x * NUM_BINS + alphabet_position / 4]), 1);
        }
    }
    if (blockIdx.x > 0)
    {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0)
            {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}

__global__ void histo_private_kernel_shared(char *data, unsigned int length, unsigned int *histo)
{
    // initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Histogram calculation
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            atomicAdd(&(histo[blockIdx.x * NUM_BINS + alphabet_position / 4]), 1);
        }
    }
    __syncthreads();

    // Reduction, commit to global memory
    if (blockIdx.x > 0)
    {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0)
            {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}

__global__ void histo_private_kernel_shared_coarsened(char *data, unsigned int length, unsigned int *histo)
{
    // initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Histogram calculation with coarsening
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int offset = 0; offset < COARSE_FACTOR && i + offset * blockDim.x * gridDim.x < length; offset++)
    {
        unsigned int idx = i + offset * blockDim.x * gridDim.x;
        int alphabet_position = data[idx] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            atomicAdd(&(histo[blockIdx.x * NUM_BINS + alphabet_position / 4]), 1);
        }
    }
    __syncthreads();

    // Reduction, commit to global memory
    if (blockIdx.x > 0)
    {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0)
            {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}

__global__ void histo_private_kernel_shared_aggregated(char *data, unsigned int length, unsigned int *histo)
{
    // initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Histogram calculation
    unsigned int accumulator = 0;
    int prevBinIdx = -1;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            int bin = alphabet_position / 4;
            if (bin == prevBinIdx)
            {
                ++accumulator;
            }
            else
            {
                if (accumulator > 0)
                {
                    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }
    if (accumulator > 0)
    {
        atomicAdd(&(histo_s[prevBinIdx]), accumulator);
    }
    __syncthreads();

    // Reduction, commit to global memory
    if (blockIdx.x > 0)
    {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0)
            {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}

int main()
{
    // Test with different input sizes
    const unsigned int LENGTHS[] = {100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    const unsigned int LENGTHS_SIZE = sizeof(LENGTHS) / sizeof(LENGTHS[0]);
    for (int test = 0; test < LENGTHS_SIZE; test++)
    {
        const unsigned int LENGTH = LENGTHS[test];
        printf("\nTesting with LENGTH = %u\n", LENGTH);

        // Input data setup
        char *h_data = (char *)malloc(LENGTH * sizeof(char));
        // Fill with random lowercase letters
        for (unsigned int i = 0; i < LENGTH; i++)
        {
            h_data[i] = 'a' + (rand() % 26);
        }

        // Calculate grid dimensions first
        dim3 blockDim(256);
        dim3 gridDim((LENGTH + blockDim.x - 1) / blockDim.x);
        int numBlocks = gridDim.x; // Store grid dimension

        // Allocate device memory
        char *d_data;
        unsigned int *d_histo;
        cudaMalloc((void **)&d_data, LENGTH * sizeof(char));
        cudaMalloc((void **)&d_histo, NUM_BINS * numBlocks * sizeof(unsigned int));

        // Copy data to device
        cudaMemcpy(d_data, h_data, LENGTH * sizeof(char), cudaMemcpyHostToDevice);

        // Time first kernel (histo_private_kernel)
        cudaEvent_t start1, stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);

        cudaMemset(d_histo, 0, NUM_BINS * numBlocks * sizeof(unsigned int));
        cudaEventRecord(start1);
        histo_private_kernel<<<gridDim, blockDim>>>(d_data, LENGTH, d_histo);
        cudaEventRecord(stop1);

        cudaEventSynchronize(stop1);
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start1, stop1);
        printf("histo_private_kernel execution time: %f ms\n", milliseconds1);

        // Time second kernel (histo_private_kernel_shared)
        cudaEvent_t start2, stop2;
        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);

        cudaMemset(d_histo, 0, NUM_BINS * numBlocks * sizeof(unsigned int));
        cudaEventRecord(start2);
        histo_private_kernel_shared<<<gridDim, blockDim>>>(d_data, LENGTH, d_histo);
        cudaEventRecord(stop2);

        cudaEventSynchronize(stop2);
        float milliseconds2 = 0;
        cudaEventElapsedTime(&milliseconds2, start2, stop2);
        printf("histo_private_kernel_shared execution time: %f ms (%.2fx)\n",
               milliseconds2, milliseconds2 / milliseconds1);

        // Time third kernel (histo_private_kernel_shared_coarsened)
        cudaEvent_t start3, stop3;
        cudaEventCreate(&start3);
        cudaEventCreate(&stop3);

        cudaMemset(d_histo, 0, NUM_BINS * numBlocks * sizeof(unsigned int));
        cudaEventRecord(start3);
        histo_private_kernel_shared_coarsened<<<gridDim, blockDim>>>(d_data, LENGTH, d_histo);
        cudaEventRecord(stop3);

        cudaEventSynchronize(stop3);
        float milliseconds3 = 0;
        cudaEventElapsedTime(&milliseconds3, start3, stop3);
        printf("histo_private_kernel_shared_coarsened execution time: %f ms (%.2fx)\n",
               milliseconds3, milliseconds3 / milliseconds1);

        // Time fourth kernel (histo_private_kernel_shared_aggregated)
        cudaEvent_t start4, stop4;
        cudaEventCreate(&start4);
        cudaEventCreate(&stop4);

        cudaMemset(d_histo, 0, NUM_BINS * numBlocks * sizeof(unsigned int));
        cudaEventRecord(start4);
        histo_private_kernel_shared_aggregated<<<gridDim, blockDim>>>(d_data, LENGTH, d_histo);
        cudaEventRecord(stop4);

        cudaEventSynchronize(stop4);
        float milliseconds4 = 0;
        cudaEventElapsedTime(&milliseconds4, start4, stop4);
        printf("histo_private_kernel_shared_aggregated execution time: %f ms (%.2fx)\n",
               milliseconds4, milliseconds4 / milliseconds1);

        // Cleanup
        cudaFree(d_data);
        cudaFree(d_histo);
        free(h_data);
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1);
        cudaEventDestroy(start2);
        cudaEventDestroy(stop2);
        cudaEventDestroy(start3);
        cudaEventDestroy(stop3);
        cudaEventDestroy(start4);
        cudaEventDestroy(stop4);
    }

    return 0;
}