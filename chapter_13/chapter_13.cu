#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__device__ void exclusiveScan(unsigned int *bits, unsigned int N)
{
    // Use shared memory for efficiency
    extern __shared__ unsigned int temp[];

    unsigned int thid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + thid;

    // Load input into shared memory
    if (i < N)
    {
        temp[thid] = bits[i];
    }
    else
    {
        temp[thid] = 0;
    }
    __syncthreads();

    // Store original value to shift later for exclusive scan
    unsigned int original = temp[thid];
    
    // Kogge-Stone scan
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int val = (thid >= stride) ? temp[thid - stride] : 0;
        __syncthreads();
        temp[thid] += val;
    }
    
    __syncthreads();
    
    // Convert to exclusive scan by shifting
    if (thid == 0)
    {
        bits[i] = 0;
    }
    else if (i < N)
    {
        bits[i] = temp[thid - 1];
    }
}

__global__ void radix_sort_iter(unsigned int *input, unsigned int *output, unsigned int *bits, unsigned int N, unsigned int iter)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key, bit;
    if (i < N)
    {
        key = input[i];
        bit = (key >> iter) & 1; // bit mask to get LSB of key
        bits[i] = bit;
    }
    exclusiveScan(bits, N);
    if (i < N)
    {
        unsigned int numOnesBefore = bits[i];
        unsigned int numOnesTotal = bits[N];
        unsigned int dst = (bit == 0) ? (i - numOnesBefore) : (N - numOnesTotal - numOnesBefore);
        output[dst] = key;
    }
}

// Sequential merge sort implementation
void merge(unsigned int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    unsigned int* L = new unsigned int[n1];
    unsigned int* R = new unsigned int[n2];
    
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
        
    i = 0;
    j = 0;
    k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
    
    delete[] L;
    delete[] R;
}

void mergeSort(unsigned int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main()
{
    const unsigned int N = 1000000;
    const unsigned int NUM_BITS = 32; // 32 bits for unsigned int
    const unsigned int BLOCK_SIZE = 256;
    
    // Allocate host memory
    unsigned int *h_input = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int *h_output = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int *sequential_input = (unsigned int*)malloc(N * sizeof(unsigned int));
    
    // Initialize random input
    srand(time(NULL));
    for (unsigned int i = 0; i < N; i++) {
        h_input[i] = rand() % UINT_MAX;
        sequential_input[i] = h_input[i];  // Copy same data for fair comparison
    }
    
    // Allocate device memory
    unsigned int *d_input, *d_output, *d_bits;
    cudaMalloc((void**)&d_input, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bits, (N + 1) * sizeof(unsigned int));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // Calculate grid size
    unsigned int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Time GPU Radix Sort
    // NOTE: CPU and GPU run on different clocks - hence the different time types
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Perform radix sort (LSB to MSB)
    unsigned int *d_in = d_input;
    unsigned int *d_out = d_output;
    
    for (unsigned int bit = 0; bit < NUM_BITS; bit++) {
        radix_sort_iter<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(
            d_in, d_out, d_bits, N, bit);
        
        unsigned int *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }
    
    if (NUM_BITS % 2 == 1) {
        cudaMemcpy(h_output, d_out, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_output, d_in, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    
    // Time CPU Merge Sort
    clock_t cpu_start = clock();
    mergeSort(sequential_input, 0, N - 1);
    clock_t cpu_end = clock();
    float cpu_milliseconds = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    printf("Sorting %u elements:\n", N);
    printf("GPU Radix sort time: %.3f ms\n", gpu_milliseconds);
    printf("CPU Merge sort time: %.3f ms\n", cpu_milliseconds);
    printf("Speedup: %.2fx\n", cpu_milliseconds / gpu_milliseconds);
    
    // Verify both sorts produced correct results
    bool correct = true;
    for (unsigned int i = 1; i < N; i++) {
        if (h_output[i] < h_output[i-1] || sequential_input[i] < sequential_input[i-1]) {
            correct = false;
            break;
        }
    }
    printf("Sort results are %s\n", correct ? "correct" : "incorrect");
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bits);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_input);
    free(h_output);
    free(sequential_input);
    
    return 0;
}