

__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[blockIdx.x*NUM_BINS + alphabet_position/4]), 1);
        }
    }
    if (blockIdx.x > 0) {
        __syncthreads();
        for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = histo[blockIdx.x*NUM_BINS + bin];
            if (binValue > 0) {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}

int main() {
    // Input data setup
    const unsigned int LENGTH = 1000000;
    char *h_data = (char*)malloc(LENGTH * sizeof(char));
    // Fill with random lowercase letters
    for(unsigned int i = 0; i < LENGTH; i++) {
        h_data[i] = 'a' + (rand() % 26);
    }

    // Allocate device memory
    char *d_data;
    unsigned int *d_histo;
    cudaMalloc((void**)&d_data, LENGTH * sizeof(char));
    cudaMalloc((void**)&d_histo, NUM_BINS * gridDim.x * sizeof(unsigned int));
    
    // Copy data to device
    cudaMemcpy(d_data, h_data, LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, NUM_BINS * gridDim.x * sizeof(unsigned int));

    // Launch kernel and time it
    dim3 blockDim(256);
    dim3 gridDim((LENGTH + blockDim.x - 1) / blockDim.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    histo_private_kernel<<<gridDim, blockDim>>>(d_data, LENGTH, d_histo);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_histo);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}