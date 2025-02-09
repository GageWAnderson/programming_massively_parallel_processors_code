#define TILE_SIZE 16

// Import the CUDA runtime and the standard C++ libraries
#include <cuda_runtime.h>
#include <iostream>

// Define the matrixMulKernel function
__global__ void matrixMulKernel(float *M, float *N, float *P, int Width)
{
    // Tile input matricies into shared memory so multiple threads within a block can use
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row, col of the element to work on
    int ROW = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    // Loop over the M and N tiles required to compute P
    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_SIZE; ++ph)
    {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row * Width + ph * TILE_SIZE + tx];
        Nds[ty][tx] = N[(ph * TILE_SIZE + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = PValue;
}

__global__ void matrixMulKernelBoundaryChecked(float *M, float *N, float *P, int Width)
{
    // Kernel does the TILE_SIZE (nice) part then the boundary part
    
    // Tile input matricies into shared memory so multiple threads within a block can use
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row, col of the element to work on
    int ROW = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    // Loop over the M and N tiles required to compute P
    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_SIZE; ++ph)
    {
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph * TILE_SIZE + tx) < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_SIZE + tx];
        else
            Mds[ty][tx] = 0.0f;
        if ((ph * TILE_SIZE + ty) < Width && Col < Width)
            Nds[ty][tx] = N[(ph * TILE_SIZE + ty) * Width + Col];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if ((Row < Width) && (Col < Width))
        P[Row * Width + Col] = PValue;
}

int main(int argc, char **argv)
{
    // Initialize the input matrices
    int Width = 1024;
    int Height = 1024;
    float *h_M = (float *)malloc(Width * Height * sizeof(float));
    float *h_N = (float *)malloc(Width * Height * sizeof(float));
    float *h_P = (float *)malloc(Width * Height * sizeof(float));

    // Initialize the input matrices with random values
    for (int i = 0; i < Width * Height; ++i)
    {
        h_M[i] = rand() / (float)RAND_MAX;
        h_N[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the output matrix on the host
    float *h_P = (float *)malloc(Width * Height * sizeof(float));

    // Allocate the input matrices on the device
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, Width * Height * sizeof(float));
    cudaMalloc(&d_N, Width * Height * sizeof(float));
    cudaMalloc(&d_P, Width * Height * sizeof(float));

    // Copy the input matrices to the device
    cudaMemcpy(d_M, h_M, Width * Height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, Width * Height * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel on the device
    matrixMulKernelBoundaryChecked<<<dim3(Width / TILE_SIZE, Height / TILE_SIZE), dim3(TILE_SIZE, TILE_SIZE)>>>(d_M, d_N, d_P, Width);

    // Copy the output matrix from the device to the host
    cudaMemcpy(h_P, d_P, Width * Height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}