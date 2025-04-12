#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector_types.h>
#include <math.h>
#include <float.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err)                                                  \
    do                                                                   \
    {                                                                    \
        cudaError_t err_ = (err);                                        \
        if (err_ != cudaSuccess)                                         \
        {                                                                \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n",              \
                    err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

struct Bounding_box
{
    float2 p_min, p_max;

    __device__ void set(float min_x, float min_y, float max_x, float max_y)
    {
        p_min = make_float2(min_x, min_y);
        p_max = make_float2(max_x, max_y);
    }
    __host__ void h_set(float min_x, float min_y, float max_x, float max_y)
    {
        p_min = make_float2(min_x, min_y);
        p_max = make_float2(max_x, max_y);
    }
    __device__ const float2 &get_min() const { return p_min; }
    __host__ const float2 &h_get_min() const { return p_min; }
    __device__ const float2 &get_max() const { return p_max; }
    __host__ const float2 &h_get_max() const { return p_max; }
    __device__ void compute_center(float2 &center) const
    {
        center.x = (p_min.x + p_max.x) * 0.5f;
        center.y = (p_min.y + p_max.y) * 0.5f;
    }
    __host__ void h_compute_center(float2 &center) const
    {
        center.x = (p_min.x + p_max.x) * 0.5f;
        center.y = (p_min.y + p_max.y) * 0.5f;
    }
};

struct Quadtree_node
{
    float x, y;
    int level;
    int parent;
    int child[4];
    int _id;
    int _points_begin;
    int _points_end;
    Bounding_box _bbox;

    __device__ int id() const { return _id; }
    __host__ int h_id() const { return _id; }
    __device__ void set_id(int id) { _id = id; }
    __host__ void h_set_id(int id) { _id = id; }
    __device__ int points_begin() const { return _points_begin; }
    __host__ int h_points_begin() const { return _points_begin; }
    __device__ int points_end() const { return _points_end; }
    __host__ int h_points_end() const { return _points_end; }
    __device__ int num_points() const { return _points_end - _points_begin; }
    __host__ int h_num_points() const { return _points_end - _points_begin; }
    __device__ void set_range(int begin, int end)
    {
        _points_begin = begin;
        _points_end = end;
    }
    __host__ void h_set_range(int begin, int end)
    {
        _points_begin = begin;
        _points_end = end;
    }
    __device__ const Bounding_box &bounding_box() const { return _bbox; }
    __host__ const Bounding_box &h_bounding_box() const { return _bbox; }
    __device__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y) { _bbox.set(min_x, min_y, max_x, max_y); }
    __host__ void h_set_bounding_box(float min_x, float min_y, float max_x, float max_y) { _bbox.h_set(min_x, min_y, max_x, max_y); }
};

struct Points
{
    float *x, *y;
    __device__ float2 get_point(int index) const { return make_float2(x[index], y[index]); }
    __host__ float2 h_get_point(int index) const { return make_float2(x[index], y[index]); }
    __device__ void set_point(int index, float2 p)
    {
        x[index] = p.x;
        y[index] = p.y;
    }
    __host__ void h_set_point(int index, float2 p)
    {
        x[index] = p.x;
        y[index] = p.y;
    }
};

struct Parameters
{
    int max_points;
    int depth;
    int max_depth;
    int min_points_per_node;
    int point_selector;
    int num_nodes_at_this_level;

    __host__ Parameters(int mp, int md, int mppn) : max_points(mp), depth(0), max_depth(md), min_points_per_node(mppn), point_selector(0), num_nodes_at_this_level(1) {}

    Parameters() = default;
};

__device__ bool check_num_points_and_depth(Quadtree_node &node, Points *points, int num_points, Parameters params)
{
    if (params.depth >= params.max_depth || num_points <= params.min_points_per_node)
    {
        // Base case: check that points[0] contains all the points
        if (params.point_selector == 1)
        {
            int it = node.points_begin(), end = node.points_end();
            for (it += threadIdx.x; it < end; it += blockDim.x)
                if (it < end)
                    points[0].set_point(it, points[1].get_point(it));
        }
        return true;
    }
    return false;
}

__device__ void count_points_in_children(const Points &in_points, int *smem, int range_begin, int range_end, float2 center)
{
    if (threadIdx.x < 4)
        smem[threadIdx.x] = 0;
    __syncthreads();

    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x)
    {
        float2 p = in_points.get_point(iter);
        if (p.x < center.x && p.y >= center.y)
            atomicAdd(&smem[0], 1); // Top left point
        else if (p.x >= center.x && p.y >= center.y)
            atomicAdd(&smem[1], 1); // Top right point
        else if (p.x < center.x && p.y < center.y)
            atomicAdd(&smem[2], 1); // Bottom left point
        else
            atomicAdd(&smem[3], 1); // Bottom right point
    }
    __syncthreads();
}

__device__ void scan_for_offsets(int node_points_begin, int *smem)
{
    int *smem2 = &smem[4];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < 4; i++)
            smem2[i] = i == 0 ? 0 : smem2[i - 1] + smem[i - 1]; // Sequential scan
        for (int i = 0; i < 4; i++)
            smem2[i] += node_points_begin;
    }
    __syncthreads();
}

__device__ void reorder_points(Points &out_points, const Points &in_points, int *smem, int range_begin, int range_end, float2 center)
{
    int *smem2 = &smem[4];

    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x)
    {
        int dest;
        float2 p = in_points.get_point(iter);
        if (p.x < center.x && p.y >= center.y)
            dest = atomicAdd(&smem2[0], 1); // Top left point
        else if (p.x >= center.x && p.y >= center.y)
            dest = atomicAdd(&smem2[1], 1); // Top right point
        else if (p.x < center.x && p.y < center.y)
            dest = atomicAdd(&smem2[2], 1); // Bottom left point
        else
            dest = atomicAdd(&smem2[3], 1); // Bottom right point

        out_points.set_point(dest, p);
    }
    __syncthreads();
}

__device__ void prepare_children(Quadtree_node *children, Quadtree_node &node, const Bounding_box &bbox, int *smem)
{
    int child_offset = 4 * node.id(); // The offsets of the children at their level
    float2 center;
    bbox.compute_center(center);

    children[child_offset + 0].set_id(4 * node.id() + 0);
    children[child_offset + 1].set_id(4 * node.id() + 1);
    children[child_offset + 2].set_id(4 * node.id() + 2);
    children[child_offset + 3].set_id(4 * node.id() + 3);

    const float2 &p_min = bbox.get_min();
    const float2 &p_max = bbox.get_max();

    // Set the bounding box for each child
    children[child_offset + 0].set_bounding_box(p_min.x, center.y, center.x, p_max.y);
    children[child_offset + 1].set_bounding_box(center.x, center.y, p_max.x, p_max.y);
    children[child_offset + 2].set_bounding_box(p_min.x, p_min.y, center.x, center.y);
    children[child_offset + 3].set_bounding_box(center.x, p_min.y, p_max.x, center.y);

    children[child_offset + 0].set_range(node.points_begin(), smem[4 + 0]);
    children[child_offset + 1].set_range(smem[4 + 0], smem[4 + 1]);
    children[child_offset + 2].set_range(smem[4 + 1], smem[4 + 2]);
    children[child_offset + 3].set_range(smem[4 + 2], smem[4 + 3]);
}

__global__ void build_quadtree_kernel(Quadtree_node *nodes, Points *points, Parameters params)
{
    __shared__ int smem[8]; // Stores the number of points in each quadrant

    // The current node in the quadtree
    Quadtree_node &node = nodes[blockIdx.x];
    node.set_id(node.id() + blockIdx.x);
    int num_points = node.num_points();

    // Base case: check the number of points and its depth
    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if (exit)
        return;

    const Bounding_box &bbox = node.bounding_box();
    float2 center;
    bbox.compute_center(center);

    // Range of points
    int range_begin = node.points_begin();
    int range_end = node.points_end();
    const Points &in_points = points[params.point_selector];      // Input points
    Points &out_points = points[(params.point_selector + 1) % 2]; // Output points

    // Count the number of points in each child
    count_points_in_children(in_points, smem, range_begin, range_end, center);

    // Scan the quadrants' results to know the reordering offset
    scan_for_offsets(node.points_begin(), smem);

    // Move points
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);

    // Launch new blocks
    if (threadIdx.x == blockDim.x - 1)
    {
        Quadtree_node *children = &nodes[params.num_nodes_at_this_level];

        prepare_children(children, node, bbox, smem);

        Parameters next_level_params = params;
        next_level_params.depth++;
        next_level_params.point_selector = (params.point_selector + 1) % 2;
        next_level_params.num_nodes_at_this_level = params.num_nodes_at_this_level + gridDim.x * 4;

        build_quadtree_kernel<<<4, blockDim.x, 8 * sizeof(int)>>>(children, points, next_level_params);
    }
}
int main()
{
    // --- Parameters ---
    const int NUM_POINTS = 100000;      // Example: 100k points
    const int MAX_DEPTH = 8;            // Example: Max depth of 8 levels
    const int MIN_POINTS_PER_NODE = 10; // Example: Stop subdividing if node has <= 10 points
    const int BLOCK_SIZE = 256;

    // --- Host Data ---
    float *h_x, *h_y;
    Quadtree_node *h_nodes;
    Points h_points_buffers[2]; // Host struct containing device pointers

    // Estimate max nodes needed for allocation
    size_t max_nodes = (size_t)((pow(4, MAX_DEPTH + 1) - 1) / 3.0);
    printf("Max potential nodes (depth %d): %zu\n", MAX_DEPTH, max_nodes);

    // Allocate host memory
    h_x = (float *)malloc(NUM_POINTS * sizeof(float));
    h_y = (float *)malloc(NUM_POINTS * sizeof(float));
    h_nodes = (Quadtree_node *)malloc(max_nodes * sizeof(Quadtree_node)); // Allocate for worst case

    if (!h_x || !h_y || !h_nodes)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize random points and find bounds
    srand(time(NULL));
    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = FLT_MIN, max_y = FLT_MIN;
    printf("Generating %d random points...\n", NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        h_x[i] = (float)rand() / RAND_MAX; // Points between 0.0 and 1.0
        h_y[i] = (float)rand() / RAND_MAX;
        if (h_x[i] < min_x)
            min_x = h_x[i];
        if (h_x[i] > max_x)
            max_x = h_x[i];
        if (h_y[i] < min_y)
            min_y = h_y[i];
        if (h_y[i] > max_y)
            max_y = h_y[i];
    }
    printf("Points generated. Bounds: X(%.3f, %.3f), Y(%.3f, %.3f)\n", min_x, max_x, min_y, max_y);

    // --- Device Data ---
    float *d_x0, *d_y0, *d_x1, *d_y1; // Ping-pong buffers for points
    Quadtree_node *d_nodes;
    Points *d_points_buffers; // Device pointer to array of 2 Points structs

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_x0, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y0, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x1, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nodes, max_nodes * sizeof(Quadtree_node)));
    CUDA_CHECK(cudaMalloc(&d_points_buffers, 2 * sizeof(Points)));

    // --- Initialization ---
    printf("Initializing data structures...\n");
    // Initialize host Points structs to point to device buffers
    h_points_buffers[0].x = d_x0;
    h_points_buffers[0].y = d_y0;
    h_points_buffers[1].x = d_x1;
    h_points_buffers[1].y = d_y1;

    // Initialize root node (node 0) on host
    h_nodes[0]._id = 0;
    h_nodes[0].level = 0;
    h_nodes[0].parent = -1;                               // Root has no parent
    h_nodes[0].h_set_range(0, NUM_POINTS);                // Root initially contains all points
    h_nodes[0]._bbox.h_set(min_x, min_y, max_x, max_y);   // Set overall bounding box
    for (int i = 0; i < 4; ++i)
        h_nodes[0].child[i] = -1; // Initialize children

    // Initialize kernel parameters
    Parameters params(NUM_POINTS, MAX_DEPTH, MIN_POINTS_PER_NODE);

    // --- Data Transfer Host -> Device ---
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_x0, h_x, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y0, h_y, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_buffers, h_points_buffers, 2 * sizeof(Points), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, &h_nodes[0], sizeof(Quadtree_node), cudaMemcpyHostToDevice)); // Copy only root node initially

    // --- Kernel Launch ---
    printf("Launching build_quadtree_kernel...\n");
    int sharedMemSize = 8 * sizeof(int);
    build_quadtree_kernel<<<1, BLOCK_SIZE, sharedMemSize>>>(d_nodes, d_points_buffers, params);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // --- Synchronization ---
    printf("Synchronizing device...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Quadtree build complete.\n");

    // --- Copy results back ---
    printf("Copying nodes back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_nodes, d_nodes, max_nodes * sizeof(Quadtree_node), cudaMemcpyDeviceToHost));

    // --- Print a subsection of the output nodes ---
    printf("\n--- Quadtree Node Information ---\n");
    const int MAX_NODES_TO_PRINT = 10; // Limit the number of nodes to print
    int nodes_to_print = 0;
    for (int i = 0; i < max_nodes && nodes_to_print < MAX_NODES_TO_PRINT; i++)
    {
        if (h_nodes[i].h_num_points() > 0)
        { // Only print nodes that contain points
            printf("Node %d: Level=%d, Points=%d, BBox=(%.2f,%.2f)-(%.2f,%.2f)\n",
                   h_nodes[i]._id, h_nodes[i].level, h_nodes[i].h_num_points(),
                   h_nodes[i]._bbox.p_min.x, h_nodes[i]._bbox.p_min.y,
                   h_nodes[i]._bbox.p_max.x, h_nodes[i]._bbox.p_max.y);
            nodes_to_print++;
        }
    }
    printf("Printed %d of %zu possible nodes\n\n", nodes_to_print, max_nodes);

    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    CUDA_CHECK(cudaFree(d_x0));
    CUDA_CHECK(cudaFree(d_y0));
    CUDA_CHECK(cudaFree(d_x1));
    CUDA_CHECK(cudaFree(d_y1));
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_points_buffers));

    free(h_x);
    free(h_y);
    free(h_nodes);

    printf("Finished.\n");
    return EXIT_SUCCESS;
}