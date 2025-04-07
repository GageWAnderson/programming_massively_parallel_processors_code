#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// define constants are available at compile-time
// Fixed array sizes can use these
#define NUM_ROWS 100000
#define NUM_COLS 100000

// NOTE: CUDA launch parameters are needed at compile-time

// COO (Coordinate list) Matrix data structure for sparse matricies
typedef struct
{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int *rowIdx;
    unsigned int *colIdx;
    float *value;
} COOMatrix;

// CSR Matrix structure for sparse matrices
// Stores only non-zero elements to save memory
typedef struct
{
    unsigned int numRows;  // Number of rows in the matrix
    unsigned int numCols;  // Number of columns in the matrix
    unsigned int nnz;      // Number of non-zero elements
    unsigned int *rowPtrs; // Array of size numRows+1, points to start of each row in colIdx and value
    unsigned int *colIdx;  // Array of size nnz, stores column indices of non-zero elements
    float *value;          // Array of size nnz, stores values of non-zero elements
} CSRMatrix;

// ELL format: padded compressed sparse matrix in column-major order
typedef struct
{
    unsigned int numRows;
    unsigned int *colIdx;
    unsigned int *nnzPerRow;
    float *value;
} ELLMatrix;

__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float *x, float *y)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cooMatrix.numNonzeros)
    {
        unsigned int row = cooMatrix.rowIdx[i];
        unsigned int col = cooMatrix.colIdx[i];
        float value = cooMatrix.value[i];
        atomicAdd(&y[row], x[col] * value);
    }
}

__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float *x, float *y)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < csrMatrix.numRows)
    {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i)
        {
            unsigned int col = csrMatrix.colIdx[i];
            float value = csrMatrix.value[i];
            sum += x[col] * value;
        }
        y[row] += sum;
    }
}

__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float *x, float *y)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows)
    {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.nnzPerRow[row]; ++t)
        {
            unsigned int i = t * ellMatrix.numRows + row;
            unsigned int col = ellMatrix.colIdx[i];
            float value = ellMatrix.value[i];
            sum += x[col] * value;
        }
        y[row] = sum;
    }
}

// TODO: write a kernel that transforms COO format to CSR format

int main()
{
    // Define matrix dimensions
    const float SPARSITY = 0.01f; // 1% non-zero elements
    const unsigned int BLOCK_SIZE = 256;

    // Calculate number of non-zero elements
    unsigned int nnz = (unsigned int)(NUM_ROWS * NUM_COLS * SPARSITY);
    printf("Creating sparse matrix with %u rows, %u columns, and %u non-zero elements\n",
           NUM_ROWS, NUM_COLS, nnz);

    // Allocate host memory for CSR matrix
    CSRMatrix h_csrMatrix;
    h_csrMatrix.numRows = NUM_ROWS;
    h_csrMatrix.numCols = NUM_COLS;
    h_csrMatrix.nnz = nnz;

    // Allocate host memory for COO matrix
    COOMatrix h_cooMatrix;
    h_cooMatrix.numRows = NUM_ROWS;
    h_cooMatrix.numCols = NUM_COLS;
    h_cooMatrix.numNonzeros = nnz;
    h_cooMatrix.rowIdx = (unsigned int *)malloc(nnz * sizeof(unsigned int));
    h_cooMatrix.colIdx = (unsigned int *)malloc(nnz * sizeof(unsigned int));
    h_cooMatrix.value = (float *)malloc(nnz * sizeof(float));

    // Allocate host memory for ELL matrix
    ELLMatrix h_ellMatrix;
    h_ellMatrix.numRows = NUM_ROWS;
    unsigned int maxNnzPerRow = (nnz + NUM_ROWS - 1) / NUM_ROWS + 1;  // Ceiling of average + 1 for safety
    h_ellMatrix.colIdx = (unsigned int *)malloc(NUM_ROWS * maxNnzPerRow * sizeof(unsigned int));
    h_ellMatrix.value = (float *)malloc(NUM_ROWS * maxNnzPerRow * sizeof(float));
    h_ellMatrix.nnzPerRow = (unsigned int *)malloc(NUM_ROWS * sizeof(unsigned int));

    h_csrMatrix.rowPtrs = (unsigned int *)malloc((NUM_ROWS + 1) * sizeof(unsigned int));
    h_csrMatrix.colIdx = (unsigned int *)malloc(nnz * sizeof(unsigned int));
    h_csrMatrix.value = (float *)malloc(nnz * sizeof(float));

    // Allocate host memory for vector x and result y
    float *h_x = (float *)malloc(NUM_COLS * sizeof(float));
    float *h_y_csr = (float *)malloc(NUM_ROWS * sizeof(float));
    float *h_y_coo = (float *)malloc(NUM_ROWS * sizeof(float));
    float *h_y_ell = (float *)malloc(NUM_ROWS * sizeof(float));

    // Initialize random seed
    srand(time(NULL));

    // Initialize vector x with random values
    for (unsigned int i = 0; i < NUM_COLS; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
    }

    // Initialize result vectors with zeros
    for (unsigned int i = 0; i < NUM_ROWS; i++) {
        h_y_csr[i] = 0.0f;
        h_y_coo[i] = 0.0f;
        h_y_ell[i] = 0.0f;
    }

    // Create a random sparse matrix in all formats
    // First, initialize CSR rowPtrs with approximate distribution
    unsigned int elemsPerRow = nnz / NUM_ROWS;
    h_csrMatrix.rowPtrs[0] = 0;
    for (unsigned int i = 1; i <= NUM_ROWS; i++) {
        h_csrMatrix.rowPtrs[i] = h_csrMatrix.rowPtrs[i - 1] + elemsPerRow;
    }
    h_csrMatrix.rowPtrs[NUM_ROWS] = nnz;

    // Generate random column indices and values
    for (unsigned int i = 0; i < nnz; i++) {
        unsigned int row = i / elemsPerRow;
        
        // CSR format
        h_csrMatrix.colIdx[i] = rand() % NUM_COLS;
        h_csrMatrix.value[i] = (float)rand() / RAND_MAX * 10.0f;
        
        // COO format (same data, different layout)
        h_cooMatrix.rowIdx[i] = row;
        h_cooMatrix.colIdx[i] = h_csrMatrix.colIdx[i];
        h_cooMatrix.value[i] = h_csrMatrix.value[i];
    }

    // Convert to ELL format
    for (unsigned int i = 0; i < NUM_ROWS; i++) {
        h_ellMatrix.nnzPerRow[i] = elemsPerRow;
        for (unsigned int j = 0; j < elemsPerRow; j++) {
            unsigned int srcIdx = i * elemsPerRow + j;
            unsigned int dstIdx = j * NUM_ROWS + i;  // Column-major order
            h_ellMatrix.colIdx[dstIdx] = h_csrMatrix.colIdx[srcIdx];
            h_ellMatrix.value[dstIdx] = h_csrMatrix.value[srcIdx];
        }
    }

    // Allocate device memory for all formats
    CSRMatrix d_csrMatrix;
    d_csrMatrix.numRows = NUM_ROWS;
    d_csrMatrix.numCols = NUM_COLS;
    d_csrMatrix.nnz = nnz;

    COOMatrix d_cooMatrix;
    d_cooMatrix.numRows = NUM_ROWS;
    d_cooMatrix.numCols = NUM_COLS;
    d_cooMatrix.numNonzeros = nnz;

    ELLMatrix d_ellMatrix;
    d_ellMatrix.numRows = NUM_ROWS;

    cudaMalloc((void **)&d_csrMatrix.rowPtrs, (NUM_ROWS + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&d_csrMatrix.colIdx, nnz * sizeof(unsigned int));
    cudaMalloc((void **)&d_csrMatrix.value, nnz * sizeof(float));

    cudaMalloc((void **)&d_cooMatrix.rowIdx, nnz * sizeof(unsigned int));
    cudaMalloc((void **)&d_cooMatrix.colIdx, nnz * sizeof(unsigned int));
    cudaMalloc((void **)&d_cooMatrix.value, nnz * sizeof(float));

    cudaMalloc((void **)&d_ellMatrix.colIdx, NUM_ROWS * maxNnzPerRow * sizeof(unsigned int));
    cudaMalloc((void **)&d_ellMatrix.value, NUM_ROWS * maxNnzPerRow * sizeof(float));
    cudaMalloc((void **)&d_ellMatrix.nnzPerRow, NUM_ROWS * sizeof(unsigned int));

    float *d_x, *d_y_csr, *d_y_coo, *d_y_ell;
    cudaMalloc((void **)&d_x, NUM_COLS * sizeof(float));
    cudaMalloc((void **)&d_y_csr, NUM_ROWS * sizeof(float));
    cudaMalloc((void **)&d_y_coo, NUM_ROWS * sizeof(float));
    cudaMalloc((void **)&d_y_ell, NUM_ROWS * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_csrMatrix.rowPtrs, h_csrMatrix.rowPtrs, (NUM_ROWS + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrMatrix.colIdx, h_csrMatrix.colIdx, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrMatrix.value, h_csrMatrix.value, nnz * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_cooMatrix.rowIdx, h_cooMatrix.rowIdx, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooMatrix.colIdx, h_cooMatrix.colIdx, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooMatrix.value, h_cooMatrix.value, nnz * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_ellMatrix.colIdx, h_ellMatrix.colIdx, NUM_ROWS * maxNnzPerRow * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ellMatrix.value, h_ellMatrix.value, NUM_ROWS * maxNnzPerRow * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ellMatrix.nnzPerRow, h_ellMatrix.nnzPerRow, NUM_ROWS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x, NUM_COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_csr, h_y_csr, NUM_ROWS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_coo, h_y_coo, NUM_ROWS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_ell, h_y_ell, NUM_ROWS * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds_csr = 0, milliseconds_coo = 0, milliseconds_ell = 0;

    // Launch kernels and measure time
    unsigned int numBlocks = (NUM_ROWS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("\nLaunching kernels with %u blocks of %u threads\n", numBlocks, BLOCK_SIZE);

    // Time CSR kernel
    cudaEventRecord(start);
    spmv_csr_kernel<<<numBlocks, BLOCK_SIZE>>>(d_csrMatrix, d_x, d_y_csr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_csr, start, stop);

    // Time COO kernel
    cudaEventRecord(start);
    spmv_coo_kernel<<<numBlocks, BLOCK_SIZE>>>(d_cooMatrix, d_x, d_y_coo);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_coo, start, stop);

    // Time ELL kernel
    cudaEventRecord(start);
    spmv_ell_kernel<<<numBlocks, BLOCK_SIZE>>>(d_ellMatrix, d_x, d_y_ell);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_ell, start, stop);

    // Print timing results and ratios
    printf("\nKernel execution times:\n");
    printf("CSR format: %.3f ms\n", milliseconds_csr);
    printf("COO format: %.3f ms\n", milliseconds_coo);
    printf("ELL format: %.3f ms\n", milliseconds_ell);
    
    printf("\nPerformance ratios:\n");
    printf("COO/CSR: %.3f\n", milliseconds_coo / milliseconds_csr);
    printf("ELL/CSR: %.3f\n", milliseconds_ell / milliseconds_csr);
    printf("COO/ELL: %.3f\n", milliseconds_coo / milliseconds_ell);

    // Clean up
    free(h_csrMatrix.rowPtrs);
    free(h_csrMatrix.colIdx);
    free(h_csrMatrix.value);
    free(h_cooMatrix.rowIdx);
    free(h_cooMatrix.colIdx);
    free(h_cooMatrix.value);
    free(h_ellMatrix.colIdx);
    free(h_ellMatrix.value);
    free(h_ellMatrix.nnzPerRow);
    free(h_x);
    free(h_y_csr);
    free(h_y_coo);
    free(h_y_ell);

    cudaFree(d_csrMatrix.rowPtrs);
    cudaFree(d_csrMatrix.colIdx);
    cudaFree(d_csrMatrix.value);
    cudaFree(d_cooMatrix.rowIdx);
    cudaFree(d_cooMatrix.colIdx);
    cudaFree(d_cooMatrix.value);
    cudaFree(d_ellMatrix.colIdx);
    cudaFree(d_ellMatrix.value);
    cudaFree(d_ellMatrix.nnzPerRow);
    cudaFree(d_x);
    cudaFree(d_y_csr);
    cudaFree(d_y_coo);
    cudaFree(d_y_ell);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}