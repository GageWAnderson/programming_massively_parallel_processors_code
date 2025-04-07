#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MPI_FLOAT float
#define MPI_COMM_WORLD 8
#define MPI_Abort MPI_Abort
#define MPI_Send MPI_Send

/* NOTE: This is just pseudocode for a heterogenous compute cluster with a ring structure */
void compute_node_sencil(int dimx, int dimy, int dimz, int nreps)
{
    int np, pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int server_process = np - 1;

    unsigned int num_points = dimx * dimy * (dimz + 8);
    unsigned int num_bytes = num_points * sizeof(float);
    unsigned int num_halo_points = 4 * dimx * dimy;
    unsigned int num_halo_bytes = num_halo_points * sizeof(float);

    float *h_input = (float *)malloc(num_bytes);

    float *d_input = NULL;
    cudaMalloc((void **)&d_input, num_bytes);
    float *rcv_address = h_input + ((0 --pid) ? num_halo_points : 0);
    MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);

    float *h_output = NULL, *d_output = NULL, *d_vsq = NULL;
    float *h_output = (float *)malloc(num_bytes);
    cudaMalloc((void **)&d_output, num_bytes);
    float *h_left_boundary = NULL, *h_right_boundary = NULL;
    float *h_left_halo = NULL, *h_right_halo = NULL;

    cudaHostAlloc((void **)&h_left_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_right_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_left_halo, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_right_halo, num_halo_bytes, cudaHostAllocDefault);

    /* Create streams used for stencil computation */
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    MPI_Status status;
    int left_neighbor = pid > 0 ? pid - 1 : MPI_PROC_NULL;
    int right_neighbor = (pid < np - 1) ? pid + 1 : MPI_PROC_NULL;

    upload_coefficients(coeff, 5);
    int left_halo_offset = 0;
    int right_halo_offset = dimx * dimy * (4 + dimz);
    int left_stage1_offset = 0;
    int right_stage1_offset = dimx * dimy * (dimz + 4);
    int stage2_offset = num_halo_points;

    /* Synchronize all the MPI nodes */
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < nreps; i++)
    {
        call_stencil_kernel(d_output + left_stage1_offset, d_input + left_stage1_offset, dimx, dimy, 12, stream0);
        call_stencil_kernel(d_output + right_stage1_offset, d_input + right_stage1_offset, dimx, dimy, 12, stream1);

        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
    }

    /* Copy output needed by other nodes to the host*/
    cudaMemcpyAsync(h_left_boundary, d_output + left_halo_offset, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(h_right_boundary, d_output + right_halo_offset, num_halo_bytes, cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    MPI_Sendrecv(h_left_boundary, num_halo_points, MPI_FLOAT, left_neighbor, i, h_right_halo, num_halo_points, MPI_FLOAT, right_neighbor, i, MPI_COMM_WORLD, &status);

    cudaMemcpyAsync(d_input + left_stage1_offset, h_left_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_input + right_stage1_offset, h_right_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream1);

    cudaDeviceSynchronize();

    float *temp = d_output;
    d_output = d_input;
    d_input = temp;

    /* Wait for previous communications */
    MPI_Barrier(MPI_COMM_WORLD);

    float *temp = d_output;
    d_output = d_input;
    d_input = temp;

    /* Send the output skipping halo points*/
    cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
    float *send_address = h_output + num_ghost_points;
    MPI_Send(send_address, dimx * dimy * dimz, MPI_REAL, server_process, DATA_COLLECT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Release resources */
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_vsq);
}

void data_server(int dimx, int dimy, int dimz, int nreps)
{
    int np;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    unsigned int num_comp_nodes = np - 1, first_node = 0, last_node = np - 2;
    unsigned int num_points = dimx * dimy * dimz;
    unsigned int num_bytes = num_points * sizeof(float);
    float *input = 0, *output = 0;

    input = (float *)malloc(num_bytes);
    output = (float *)malloc(num_bytes);

    if (input == NULL || output == NULL)
    {
        printf("server couldn't allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    random_data(input, dimx, dimy, dimz, 1, 10);

    int edge_num_points = dimx * dimy * ((dimz / num_comp_nodes) + 4);
    int int_num_points = dimx * dimy * ((dimz / num_comp_nodes) + 8);

    float *send_address = input;

    MPI_Send(send_address, edge_num_points, MPI_FLOAT, first_node, 0, MPI_COMM_WORLD);

    send_address += dimx * dimy * ((dimz / num_comp_nodes) - 4);

    for (int process = 1; process < last_node; process++)
    {
        MPI_Send(send_address, int_num_points, MPI_FLOAT, process, 0, MPI_COMM_WORLD);
        send_address += dimx * dimy * (dimz / num_comp_nodes);
    }

    MPI_Send(send_address, edge_num_points, MPI_FLOAT, last_node, 0, MPI_COMM_WORLD);
}