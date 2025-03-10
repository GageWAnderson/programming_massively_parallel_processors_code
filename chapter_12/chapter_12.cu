#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define ARRAY_LENGTH 1000000000

void h_merge_sequential(int *A, int *B, int *C, int m, int n)
{
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < m) && (j < n))
    {
        if (A[i] < B[j])
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }
    if (i == m)
    {
        while (j < n)
        {
            C[k++] = B[j++];
        }
    }
    else
    {
        while (i < m)
        {
            C[k++] = A[i++];
        }
    }
}

void h_merge_sort_sequential(int *arr, int *temp, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        h_merge_sort_sequential(arr, temp, left, mid);
        h_merge_sort_sequential(arr, temp, mid + 1, right);

        // Merge the sorted halves
        int n1 = mid - left + 1;
        int n2 = right - mid;

        // Copy data to temp arrays
        for (int i = 0; i < n1; i++)
            temp[i] = arr[left + i];
        for (int j = 0; j < n2; j++)
            temp[n1 + j] = arr[mid + 1 + j];

        // Merge temp arrays back into arr
        h_merge_sequential(temp, temp + n1, arr + left, n1, n2);
    }
}

__device__ void d_merge_sequential(int *A, int *B, int *C, int m, int n)
{
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < m) && (j < n))
    {
        if (A[i] < B[j])
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }
    if (i == m)
    {
        while (j < n)
        {
            C[k++] = B[j++];
        }
    }
    else
    {
        while (i < m)
        {
            C[k++] = A[i++];
        }
    }
}

__device__ void d_merge_sort_sequential(int *arr, int *temp, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        d_merge_sort_sequential(arr, temp, left, mid);
        d_merge_sort_sequential(arr, temp, mid + 1, right);

        // Merge the sorted halves
        int n1 = mid - left + 1;
        int n2 = right - mid;

        // Copy data to temp arrays
        for (int i = 0; i < n1; i++)
            temp[i] = arr[left + i];
        for (int j = 0; j < n2; j++)
            temp[n1 + j] = arr[mid + 1 + j];

        // Merge temp arrays back into arr
        d_merge_sequential(temp, temp + n1, arr + left, n1, n2);
    }
}

// NOTE: In-place merge sort isn't performant on GPUs since speed is memory bound

// NOTE: Co-rank is a merge function helper from a 2012 paper that assists with
// parallelizing the merge function by identifying the segments of A,B that are
// needed to construct a given segment of the output array C
__device__ int co_rank(int k, int *A, int m, int *B, int n)
{
    int i = k < m ? k : m; // i = min(k,m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : n - n; // i_low = max(0, k-n)
    int j_low = 0 > (k - m) ? 0 : k - m; // j_low = max(0, k-m)
    int delta;
    bool active = true;
    while (active)
    {
        if (i > 0 && j < n && A[i - 1] > B[j])
        {
            delta = ((i - i_low + 1) >> 1); // ceil(i-i_low/2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i])
        {
            delta = ((j - j_low + 1) >> 1); // ceil(j-j_low/2)
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else
        {
            active = false;
        }
    }
    return i;
}
__global__ void merge_basic_kernel(int *A, int m, int *B, int n, int *C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = (int)ceilf((float)(m + n) / (blockDim.x * gridDim.x));
    int k_curr = tid * elementsPerThread;                   // start output index
    int k_next = min((tid + 1) * elementsPerThread, m + n); // end output index
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    d_merge_sequential(&A[i_curr], &B[j_curr], &C[k_curr], i_next - i_curr, j_next - j_curr);
}

__global__ void merge_tiled_kernel(int *A, int m, int *B, int n, int *C, int tile_size)
{
    extern __shared__ int shareAB[]; // Section of input arrays in shared mem in this SM
    int *A_S = &shareAB[0];
    int *B_S = &shareAB[tile_size];
    int C_curr = blockIdx.x * (int)ceilf((float)(m + n) / gridDim.x);                            // start point of block C subarray
    int C_next = min((int)((blockIdx.x + 1) * ceilf((float)(m + n) / gridDim.x)), (int)(m + n)); // ending point

    if (threadIdx.x == 0)
    {
        A_S[0] = co_rank(C_curr, A, m, B, n); // Make block-level co-rank values visible
        A_S[1] = co_rank(C_next, A, m, B, n); // to all other threads in this block
    }
    __syncthreads(); // Needed since other threads need to see the block-level co-rank value

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceilf((C_length) / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    while (counter < total_iteration)
    {
        // loading tile size A and B elements into shared memory
        for (int i = 0; i < tile_size; i += blockDim.x)
        {
            if (i + threadIdx.x < A_length - A_consumed)
            {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < tile_size; i += blockDim.x)
        {
            if (i + threadIdx.x < B_length - B_consumed)
            {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
    }
    __syncthreads();

    int c_curr = threadIdx.x * (tile_size / blockDim.x);
    int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
    c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
    // Find co-rank for c_curr and c_next
    int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
    int b_curr = c_curr - a_curr;

    int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
    int b_next = c_next - a_next;

    // All threads call the sequential merge function
    d_merge_sequential(A_S + a_curr, B_S + b_curr, C + C_curr + C_completed + c_curr, a_next - a_curr, b_next - b_curr);
    // Update the number of A and B elements that have been consumed thus far
    counter++;
    C_completed += tile_size;
    A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
    B_consumed = C_completed - A_consumed;
    __syncthreads();
}

int main()
{
    // Allocate memory for the array
    int *arr = (int *)malloc(ARRAY_LENGTH * sizeof(int));
    int *temp = (int *)malloc(ARRAY_LENGTH * sizeof(int));

    // Device arrays
    int *d_A, *d_B, *d_C;

    if (!arr || !temp)
    {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize random seed
    srand(time(NULL));

    // Generate random array
    for (int i = 0; i < ARRAY_LENGTH; i++)
    {
        arr[i] = rand() % 10000;
    }

    // Record start time for sequential sort
    clock_t start = clock();

    // Perform merge sort
    h_merge_sort_sequential(arr, temp, 0, ARRAY_LENGTH - 1);

    // Record end time
    clock_t end = clock();

    // Calculate and print time taken
    double time_taken = (1000 * (double)(end - start)) / CLOCKS_PER_SEC;
    printf("Sequential merge sort took %f ms to sort %d elements\n", time_taken, ARRAY_LENGTH);

    // Prepare data for GPU comparison
    int half_size = ARRAY_LENGTH / 2;

    // Allocate device memory
    cudaMalloc((void **)&d_A, half_size * sizeof(int));
    cudaMalloc((void **)&d_B, (ARRAY_LENGTH - half_size) * sizeof(int));
    cudaMalloc((void **)&d_C, ARRAY_LENGTH * sizeof(int));

    // Copy sorted subarrays to device
    cudaMemcpy(d_A, arr, half_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, arr + half_size, (ARRAY_LENGTH - half_size) * sizeof(int), cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    int blockSize = 256;
    int numBlocks = (ARRAY_LENGTH + blockSize - 1) / blockSize;

    // Test basic merge kernel
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    merge_basic_kernel<<<numBlocks, blockSize>>>(d_A, half_size, d_B, ARRAY_LENGTH - half_size, d_C);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float basic_time = 0;
    cudaEventElapsedTime(&basic_time, start_event, stop_event);
    printf("Basic merge kernel took %f ms\n", basic_time);

    // Test tiled merge kernel
    int tile_size = 1024;
    int shared_mem_size = 2 * tile_size * sizeof(int);

    cudaEventRecord(start_event);
    merge_tiled_kernel<<<numBlocks, blockSize, shared_mem_size>>>(d_A, half_size, d_B, ARRAY_LENGTH - half_size, d_C, tile_size);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float tiled_time = 0;
    cudaEventElapsedTime(&tiled_time, start_event, stop_event);
    printf("Tiled merge kernel took %f ms\n", tiled_time);

    printf("Speedup of tiled vs basic: %f\n", basic_time / tiled_time);
    printf("Speedup of tiled vs sequential: %f\n", time_taken / tiled_time);
    printf("Speedup of basic vs sequential: %f\n", time_taken / basic_time);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // Verify the array is sorted (for small arrays or debugging)
    if (ARRAY_LENGTH <= 100)
    {
        printf("Sorted array: ");
        for (int i = 0; i < ARRAY_LENGTH; i++)
        {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(arr);
    free(temp);

    return 0;
}