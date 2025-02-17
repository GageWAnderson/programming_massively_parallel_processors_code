#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define IN_TILE_DIM 32
#define FILTER_RADIUS 1
#define FILTER_DIM (2 * (FILTER_RADIUS) + 1)
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height)
{
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // load the input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    // calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++)
            {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++)
                {
                    Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

__global__ void convolution_tiled_2D_const_mem_kernel_block_size_p2(float *N, float *P, int width, int height)
{
    // Each thread computes one output element
    int out_col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int out_row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;

    // Load the input tile into shared memory
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    
    // Calculate starting position for input tile
    int in_row_start = blockIdx.y * OUT_TILE_DIM - FILTER_RADIUS;
    int in_col_start = blockIdx.x * OUT_TILE_DIM - FILTER_RADIUS;

    // Iterate to load all input elements into shared memory
    for (int i = threadIdx.y; i < IN_TILE_DIM; i += blockDim.y)
    {
        for (int j = threadIdx.x; j < IN_TILE_DIM; j += blockDim.x)
        {
            int in_row = in_row_start + i;
            int in_col = in_col_start + j;
            
            if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
            {
                N_s[i][j] = N[in_row * width + in_col];
            }
            else
            {
                N_s[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Calculate output elements - only for threads within output dimensions
    if (out_row < height && out_col < width)
    {
        float Pvalue = 0.0f;
        for (int i = 0; i < FILTER_DIM; i++)
        {
            for (int j = 0; j < FILTER_DIM; j++)
            {
                Pvalue += F_c[i][j] * N_s[threadIdx.y + i][threadIdx.x + j];
            }
        }
        P[out_row * width + out_col] = Pvalue;
    }
}

int main()
{
    // Add this before your existing main() code
    // Create a host-side filter for edge detection
    float h_Filter[(2 * FILTER_RADIUS + 1)][(2 * FILTER_RADIUS + 1)] = {0}; // Initialize all to 0

    // Create a Laplacian edge detection filter
    // This is a simple edge detection kernel that will detect edges in all directions
    int center = FILTER_RADIUS;
    h_Filter[center][center] = 8.0f;          // Center pixel
    h_Filter[center - 1][center] = -1.0f;     // Top
    h_Filter[center + 1][center] = -1.0f;     // Bottom
    h_Filter[center][center - 1] = -1.0f;     // Left
    h_Filter[center][center + 1] = -1.0f;     // Right
    h_Filter[center - 1][center - 1] = -1.0f; // Top-Left
    h_Filter[center - 1][center + 1] = -1.0f; // Top-Right
    h_Filter[center + 1][center - 1] = -1.0f; // Bottom-Left
    h_Filter[center + 1][center + 1] = -1.0f; // Bottom-Right

    // Copy filter to constant memory
    cudaMemcpyToSymbol(F_c, h_Filter, sizeof(float) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1));

    // generate random data to test the convolutional kernels
    // int width = 65536;
    // int height = 65536;
    // int width = 64;
    // int height = 64;
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);

    float *h_N = (float *)malloc(size);
    float *h_P = (float *)malloc(size);
    float *h_P_ref = (float *)malloc(size); // Reference output

    // Initialize input data with random values
    for (int i = 0; i < width * height; i++)
    {
        h_N[i] = rand() % 100 / 10.0f;
    }

    // Print input data
    // printf("Input Data:\n");
    // for (int i = 0; i < width * height; i++)
    // {
    //     printf("%0.1f ", h_N[i]);
    //     if ((i + 1) % width == 0)
    //         printf("\n");
    // }
    // printf("\n");

    float *d_N, *d_P;
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);

    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Launch blocks with dimensions matching the output tiles
    dim3 dimBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
                 (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    convolution_tiled_2D_const_mem_kernel_block_size_p2<<<dimGrid, dimBlock>>>(d_N, d_P, width, height);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Run the reference kernel
    dim3 dimBlockRef(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGridRef((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    convolution_tiled_2D_const_mem_kernel<<<dimGridRef, dimBlockRef>>>(d_N, d_P, width, height);

    cudaMemcpy(h_P_ref, d_P, size, cudaMemcpyDeviceToHost);

    // // Print output data
    // printf("Output Data (Block Size P2):\n");
    // for (int i = 0; i < width * height; i++)
    // {
    //     printf("%0.1f ", h_P[i]);
    //     if ((i + 1) % width == 0)
    //         printf("\n");
    // }
    // printf("\n");

    // printf("Output Data (Reference):\n");
    // for (int i = 0; i < width * height; i++)
    // {
    //     printf("%0.1f ", h_P_ref[i]);
    //     if ((i + 1) % width == 0)
    //         printf("\n");
    // }
    // printf("\n");

    // // Validate the results
    // bool valid = true;
    // for (int i = 0; i < width * height; i++)
    // {
    //     if (fabs(h_P[i] - h_P_ref[i]) > 1e-5)
    //     {
    //         valid = false;
    //         break;
    //     }
    // }

    // if (valid)
    // {
    //     printf("Validation PASSED\n");
    // }
    // else
    // {
    //     printf("Validation FAILED\n");
    // }

    // Free memory
    free(h_N);
    free(h_P);
    free(h_P_ref);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}