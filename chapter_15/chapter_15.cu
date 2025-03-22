#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <limits.h>

// NOTE: I'm getting instincts for what (silent) CUDA memory bugs are
// Need to make sure that local frontiers can fit into shared memory in SMs
#define LOCAL_FRONTIER_CAPACITY 10

// BFS traversal methods
typedef enum
{
    BFS_VERTEX_PUSH,
    BFS_VERTEX_PULL,
    BFS_EDGE,
    BFS_FRONTIER,
    BFS_FRONTIER_PRIVATIZED
} BFSMethod;

// COO (Coordinate list) Matrix data structure for sparse matricies (graphs)
typedef struct
{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numEdges;
    unsigned int *src;
    unsigned int *dst;
} COOGraph;

// CSR (Compressed sparse row) Graph structure for sparse matrices (graphs)
// Stores only non-zero elements to save memory
// Has easy access to the OUTGOING edges
typedef struct
{
    unsigned int numRows;      // Number of rows in the matrix
    unsigned int numCols;      // Number of columns in the matrix
    unsigned int numVerticies; // Number of non-zero elements
    unsigned int *srcPtrs;     // Array of size numRows+1, points to start of each row in colIdx and value
    unsigned int *dst;         // Array of size nnz, stores column indices of non-zero elements
} CSRGraph;

// CSC (Compressed sparse col) Graph structure for sparse matrices (graphs)
// Stores only non-zero elements to save memory
// Has easy access to the INCOMING edges
typedef struct
{
    unsigned int numCols;      // Number of columns in the matrix
    unsigned int numRows;      // Number of rows in the matrix
    unsigned int numVerticies; // Number of non-zero elements
    unsigned int *dstPtrs;     // Array of size numCols+1, points to start of each col in rowIdx and value
    unsigned int *src;         // Array of size nnz, stores row indices of non-zero elements
} CSCGraph;

__global__ void vertex_centric_push_bfs_kernel(CSRGraph csrGraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel)
{
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < csrGraph.numVerticies)
    {
        if (level[vertex] == currLevel - 1)
        {
            for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
            {
                unsigned int neighbor = csrGraph.dst[edge];
                if (level[neighbor] == UINT_MAX)
                {
                    // Visit the neighbor
                    level[neighbor] = currLevel;
                    *newVertexVisited = 1;
                }
            }
        }
    }
}

__global__ void vertex_centric_pull_bfs_kernel(CSCGraph cscGraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel)
{
    // NOTE: A pull-centric BFS is faster than a push-centric approach if the input data is in column-major layout
    // Speed depends on whether it's faster to access the OUTBOUND or INBOUND edges for a given vertex
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < cscGraph.numVerticies)
    {
        if (level[vertex] == UINT_MAX)
        {
            for (unsigned int edge = cscGraph.dstPtrs[vertex]; edge < cscGraph.dstPtrs[vertex + 1]; ++edge)
            {
                unsigned int neighbor = cscGraph.src[edge];
                // If a thread finds an index belonging to a previous level, the thread labels this vertex as the current level
                // PULL-BASED implementation
                if (level[neighbor] == currLevel - 1)
                {
                    level[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break;
                }
            }
        }
    }
    return;
}

__global__ void edge_centric_push_bfs_kernel(COOGraph cooGraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel)
{
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < cooGraph.numEdges)
    {
        unsigned int vertex = cooGraph.src[edge];
        if (level[vertex] == currLevel - 1)
        {
            unsigned int neighbor = cooGraph.dst[edge];
            if (level[neighbor] == UINT_MAX)
            {
                // Visit the neighbor
                level[neighbor] = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}

__global__ void frontier_bfs_kernel(CSRGraph csrGraph, unsigned int *level, unsigned int *prevFrontier, unsigned int *currFrontier, unsigned int numPrevFrontier, unsigned int *numCurrFrontier, unsigned int currLevel)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier)
    {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
        {
            unsigned int neighbor = csrGraph.dst[edge];
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX)
            {
                unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                currFrontier[currFrontierIdx] = neighbor;
            }
        }
    }
}

__global__ void privatized_frontier_bfs_kernel(CSRGraph csrGraph, unsigned int *level, unsigned int *prevFrontier, unsigned int *currFrontier, unsigned int numPrevFrontier, unsigned int *numCurrFrontier, unsigned int currLevel)
{
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0)
    {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier)
    {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
        {
            unsigned int neighbor = csrGraph.dst[edge];
            if(atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if(currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                }
            }
        }
    }
    __syncthreads(); // NOTE: lots of thread divergence in warps here, no garuntee about adjacent verticies having the same behavior

    // Allocate the global frontier
    __shared__ unsigned int currFrontierStartIdx;
    if(threadIdx.x == 0) {
        // 1 thread in the block gets the start index in global mem for the current frontier
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    for (unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s; currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

// Helper function to construct and allocate the COO graph
COOGraph *construct_coo_graph(unsigned int numVertices, float edgeProbability)
{
    // Allocate host memory for COO graph
    COOGraph *h_graph = (COOGraph *)malloc(sizeof(COOGraph));
    h_graph->numRows = numVertices;
    h_graph->numCols = numVertices;

    // First pass: count total edges
    unsigned int totalEdges = 0;
    for (unsigned int src = 0; src < numVertices; src++)
    {
        for (unsigned int dst = 0; dst < numVertices; dst++)
        {
            if (src != dst && ((float)rand() / RAND_MAX) < edgeProbability)
            {
                totalEdges++;
            }
        }
    }

    h_graph->numEdges = totalEdges;
    printf("Generated %u actual edges for COO graph\n", totalEdges);

    // Allocate arrays for edges
    h_graph->src = (unsigned int *)malloc(totalEdges * sizeof(unsigned int));
    h_graph->dst = (unsigned int *)malloc(totalEdges * sizeof(unsigned int));

    // Second pass: fill edge arrays
    unsigned int edgeIndex = 0;
    srand(time(NULL)); // Reset seed to get the same edges
    for (unsigned int src = 0; src < numVertices; src++)
    {
        for (unsigned int dst = 0; dst < numVertices; dst++)
        {
            if (src != dst && ((float)rand() / RAND_MAX) < edgeProbability)
            {
                h_graph->src[edgeIndex] = src;
                h_graph->dst[edgeIndex] = dst;
                edgeIndex++;
            }
        }
    }

    return h_graph;
}

// Helper function to construct and allocate the CSC graph
CSCGraph *construct_csc_graph(unsigned int numVertices, float edgeProbability)
{
    // Allocate host memory for CSC graph
    CSCGraph *h_graph = (CSCGraph *)malloc(sizeof(CSCGraph));
    h_graph->numRows = numVertices;
    h_graph->numCols = numVertices;

    // First pass: count edges per column
    unsigned int *edgesPerColumn = (unsigned int *)calloc(numVertices, sizeof(unsigned int));
    unsigned int totalEdges = 0;

    for (unsigned int src = 0; src < numVertices; src++)
    {
        for (unsigned int dst = 0; dst < numVertices; dst++)
        {
            if (src != dst && ((float)rand() / RAND_MAX) < edgeProbability)
            {
                edgesPerColumn[dst]++;
                totalEdges++;
            }
        }
    }

    h_graph->numVerticies = numVertices;
    printf("Generated %u actual edges for CSC graph\n", totalEdges);

    // Allocate arrays
    h_graph->dstPtrs = (unsigned int *)malloc((numVertices + 1) * sizeof(unsigned int));
    h_graph->src = (unsigned int *)malloc(totalEdges * sizeof(unsigned int));

    // Build dstPtrs array
    h_graph->dstPtrs[0] = 0;
    for (unsigned int i = 0; i < numVertices; i++)
    {
        h_graph->dstPtrs[i + 1] = h_graph->dstPtrs[i] + edgesPerColumn[i];
    }

    // Reset edge counters for second pass
    unsigned int *currentEdge = (unsigned int *)calloc(numVertices, sizeof(unsigned int));

    // Second pass: fill source array
    srand(time(NULL)); // Reset seed to get the same edges
    for (unsigned int src = 0; src < numVertices; src++)
    {
        for (unsigned int dst = 0; dst < numVertices; dst++)
        {
            if (src != dst && ((float)rand() / RAND_MAX) < edgeProbability)
            {
                unsigned int index = h_graph->dstPtrs[dst] + currentEdge[dst];
                h_graph->src[index] = src;
                currentEdge[dst]++;
            }
        }
    }

    // Free temporary arrays
    free(edgesPerColumn);
    free(currentEdge);

    return h_graph;
}

// Helper function to free COO graph memory
void free_coo_graph(COOGraph *graph)
{
    if (graph)
    {
        free(graph->src);
        free(graph->dst);
        free(graph);
    }
}

// Helper function to free CSC graph memory
void free_csc_graph(CSCGraph *graph)
{
    if (graph)
    {
        free(graph->dstPtrs);
        free(graph->src);
        free(graph);
    }
}

// Helper function to perform BFS traversal
unsigned int perform_bfs(CSRGraph &h_graph, COOGraph &h_coo_graph, CSCGraph &h_csc_graph,
                         unsigned int *h_level, unsigned int startVertex,
                         unsigned int numVertices, unsigned int blockSize,
                         bool useVertexCentric = true, bool usePush = true, bool useFrontier = false,
                         bool usePrivatizedFrontier = false)
{
    // Allocate device memory
    CSRGraph d_graph;
    d_graph.numRows = h_graph.numRows;
    d_graph.numCols = h_graph.numCols;
    d_graph.numVerticies = h_graph.numVerticies;

    cudaMalloc(&d_graph.srcPtrs, (numVertices + 1) * sizeof(unsigned int));
    cudaMalloc(&d_graph.dst, h_graph.srcPtrs[numVertices] * sizeof(unsigned int));

    // For pull-based approach, we need CSC format
    CSCGraph d_cscGraph;
    if (useVertexCentric && !usePush)
    {
        d_cscGraph.numCols = h_csc_graph.numCols;
        d_cscGraph.numRows = h_csc_graph.numRows;
        d_cscGraph.numVerticies = h_csc_graph.numVerticies;

        cudaMalloc(&d_cscGraph.dstPtrs, (numVertices + 1) * sizeof(unsigned int));
        cudaMalloc(&d_cscGraph.src, h_csc_graph.dstPtrs[numVertices] * sizeof(unsigned int));

        cudaMemcpy(d_cscGraph.dstPtrs, h_csc_graph.dstPtrs, (numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cscGraph.src, h_csc_graph.src, h_csc_graph.dstPtrs[numVertices] * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    // For edge-centric approach, we need COO format
    COOGraph d_cooGraph;
    if (!useVertexCentric && !useFrontier)
    {
        d_cooGraph.numRows = h_coo_graph.numRows;
        d_cooGraph.numCols = h_coo_graph.numCols;
        d_cooGraph.numEdges = h_coo_graph.numEdges;

        cudaMalloc(&d_cooGraph.src, h_coo_graph.numEdges * sizeof(unsigned int));
        cudaMalloc(&d_cooGraph.dst, h_coo_graph.numEdges * sizeof(unsigned int));

        cudaMemcpy(d_cooGraph.src, h_coo_graph.src, h_coo_graph.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cooGraph.dst, h_coo_graph.dst, h_coo_graph.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    unsigned int *d_level;
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));

    unsigned int *d_newVertexVisited;
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

    // Copy data to device
    cudaMemcpy(d_graph.srcPtrs, h_graph.srcPtrs, (numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.dst, h_graph.dst, h_graph.srcPtrs[numVertices] * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, h_level, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // BFS traversal
    unsigned int currLevel = 1;
    unsigned int h_newVertexVisited = 1;

    // Calculate grid dimensions
    unsigned int numBlocks;
    if (useVertexCentric || useFrontier)
    {
        numBlocks = (numVertices + blockSize - 1) / blockSize;
    }
    else
    {
        // For edge-centric, we need blocks based on number of edges
        numBlocks = (h_coo_graph.numEdges + blockSize - 1) / blockSize;
    }

    // For frontier-based approach
    unsigned int *d_prevFrontier = nullptr;
    unsigned int *d_currFrontier = nullptr;
    unsigned int *d_numCurrFrontier = nullptr;
    unsigned int h_numPrevFrontier = 0;
    unsigned int h_numCurrFrontier = 0;

    if (useFrontier)
    {
        // Allocate memory for frontier arrays
        cudaMalloc(&d_prevFrontier, numVertices * sizeof(unsigned int));
        cudaMalloc(&d_currFrontier, numVertices * sizeof(unsigned int));
        cudaMalloc(&d_numCurrFrontier, sizeof(unsigned int));

        // Initialize first frontier with start vertex
        h_numPrevFrontier = 1;
        unsigned int h_startVertex = startVertex;
        cudaMemcpy(d_prevFrontier, &h_startVertex, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    printf("Starting BFS traversal from vertex %u using %s approach\n",
           startVertex,
           useFrontier ? "frontier-based" : (useVertexCentric ? (usePush ? "vertex-centric push" : "vertex-centric pull") : "edge-centric"));

    while (h_newVertexVisited || (useFrontier && h_numPrevFrontier > 0))
    {
        h_newVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &h_newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);

        if (useFrontier)
        {
            // Reset current frontier counter
            h_numCurrFrontier = 0;
            cudaMemcpy(d_numCurrFrontier, &h_numCurrFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);

            // Calculate blocks based on previous frontier size
            unsigned int frontierBlocks = (h_numPrevFrontier + blockSize - 1) / blockSize;

            // Launch frontier-based kernel
            if (usePrivatizedFrontier) {
                privatized_frontier_bfs_kernel<<<frontierBlocks, blockSize>>>(
                    d_graph, d_level, d_prevFrontier, d_currFrontier,
                    h_numPrevFrontier, d_numCurrFrontier, currLevel);
            } else {
                frontier_bfs_kernel<<<frontierBlocks, blockSize>>>(
                    d_graph, d_level, d_prevFrontier, d_currFrontier,
                    h_numPrevFrontier, d_numCurrFrontier, currLevel);
            }

            // Get size of new frontier
            cudaMemcpy(&h_numCurrFrontier, d_numCurrFrontier, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            // Swap frontiers for next iteration
            unsigned int *temp = d_prevFrontier;
            d_prevFrontier = d_currFrontier;
            d_currFrontier = temp;
            h_numPrevFrontier = h_numCurrFrontier;

            if (h_numCurrFrontier > 0)
            {
                h_newVertexVisited = 1;
            }
        }
        else if (useVertexCentric)
        {
            if (usePush)
            {
                vertex_centric_push_bfs_kernel<<<numBlocks, blockSize>>>(d_graph, d_level, d_newVertexVisited, currLevel);
            }
            else
            {
                vertex_centric_pull_bfs_kernel<<<numBlocks, blockSize>>>(d_cscGraph, d_level, d_newVertexVisited, currLevel);
            }
        }
        else
        {
            edge_centric_push_bfs_kernel<<<numBlocks, blockSize>>>(d_cooGraph, d_level, d_newVertexVisited, currLevel);
        }

        if (!useFrontier)
        {
            cudaMemcpy(&h_newVertexVisited, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        }

        if (h_newVertexVisited)
        {
            printf("Level %u: found new vertices\n", currLevel);
            currLevel++;
        }
    }

    // Copy results back to host
    cudaMemcpy(h_level, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_graph.srcPtrs);
    cudaFree(d_graph.dst);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);

    // Free frontier-based memory if used
    if (useFrontier)
    {
        cudaFree(d_prevFrontier);
        cudaFree(d_currFrontier);
        cudaFree(d_numCurrFrontier);
    }

    // Free CSC graph memory if used
    if (useVertexCentric && !usePush)
    {
        cudaFree(d_cscGraph.dstPtrs);
        cudaFree(d_cscGraph.src);
    }

    // Free COO graph memory if used
    if (!useVertexCentric && !useFrontier)
    {
        cudaFree(d_cooGraph.src);
        cudaFree(d_cooGraph.dst);
    }

    return currLevel;
}

// Helper function to construct and allocate the graph
CSRGraph *construct_csr_graph(unsigned int numVertices, float edgeProbability)
{
    // Allocate host memory for CSR graph
    CSRGraph *h_graph = (CSRGraph *)malloc(sizeof(CSRGraph));
    h_graph->numRows = numVertices;
    h_graph->numCols = numVertices;
    h_graph->numVerticies = numVertices;

    // Allocate arrays for graph construction
    h_graph->srcPtrs = (unsigned int *)malloc((numVertices + 1) * sizeof(unsigned int));

    // First pass: count edges per vertex
    unsigned int *edgesPerVertex = (unsigned int *)calloc(numVertices, sizeof(unsigned int));

    // Initialize random seed
    srand(time(NULL));

    // Generate random edges
    unsigned int totalEdges = 0;
    for (unsigned int src = 0; src < numVertices; src++)
    {
        for (unsigned int dst = 0; dst < numVertices; dst++)
        {
            if (src != dst && ((float)rand() / RAND_MAX) < edgeProbability)
            {
                edgesPerVertex[src]++;
                totalEdges++;
            }
        }
    }

    printf("Generated %u actual edges\n", totalEdges);

    // Allocate destination array based on actual edge count
    h_graph->dst = (unsigned int *)malloc(totalEdges * sizeof(unsigned int));

    // Build srcPtrs array
    h_graph->srcPtrs[0] = 0;
    for (unsigned int i = 0; i < numVertices; i++)
    {
        h_graph->srcPtrs[i + 1] = h_graph->srcPtrs[i] + edgesPerVertex[i];
    }

    // Reset edge counters for second pass
    unsigned int *currentEdge = (unsigned int *)calloc(numVertices, sizeof(unsigned int));

    // Second pass: fill destination array
    srand(time(NULL)); // Reset seed to get the same edges
    for (unsigned int src = 0; src < numVertices; src++)
    {
        for (unsigned int dst = 0; dst < numVertices; dst++)
        {
            if (src != dst && ((float)rand() / RAND_MAX) < edgeProbability)
            {
                unsigned int index = h_graph->srcPtrs[src] + currentEdge[src];
                h_graph->dst[index] = dst;
                currentEdge[src]++;
            }
        }
    }

    // Free temporary arrays
    free(edgesPerVertex);
    free(currentEdge);

    return h_graph;
}

// Helper function to free graph memory
void free_graph(CSRGraph *graph)
{
    if (graph)
    {
        free(graph->srcPtrs);
        free(graph->dst);
        free(graph);
    }
}

// Helper function to initialize level array for BFS
unsigned int *initialize_level_array(unsigned int numVertices)
{
    unsigned int *h_level = (unsigned int *)malloc(numVertices * sizeof(unsigned int));
    for (unsigned int i = 0; i < numVertices; i++)
    {
        h_level[i] = UINT_MAX; // Unvisited
    }
    return h_level;
}

// Helper function to count vertices at each level
unsigned int *count_vertices_per_level(unsigned int *h_level, unsigned int numVertices,
                                       unsigned int numLevels, unsigned int *visitedVertices)
{
    unsigned int *verticesPerLevel = (unsigned int *)calloc(numLevels, sizeof(unsigned int));
    *visitedVertices = 0;

    for (unsigned int i = 0; i < numVertices; i++)
    {
        if (h_level[i] != UINT_MAX)
        {
            verticesPerLevel[h_level[i]]++;
            (*visitedVertices)++;
        }
    }

    return verticesPerLevel;
}

// Helper function to process command line arguments
void process_arguments(int argc, char *argv[], unsigned int *numVertices,
                       float *edgeProbability, unsigned int *blockSize)
{
    // Set default values
    *numVertices = (argc > 1) ? atoi(argv[1]) : 10000;
    *edgeProbability = (argc > 2) ? atof(argv[2]) : 0.001f;
    *blockSize = (argc > 3) ? atoi(argv[3]) : 256;

    // Validate inputs
    if (*blockSize > 1024)
    {
        fprintf(stderr, "Error: BLOCK_SIZE must be <= 1024\n");
        exit(1);
    }
    if (*edgeProbability <= 0.0f || *edgeProbability >= 1.0f)
    {
        fprintf(stderr, "Error: EDGE_PROBABILITY must be between 0 and 1\n");
        exit(1);
    }
}

// Helper function to run a BFS approach and measure its performance
float run_bfs_approach(CSRGraph *h_graph, COOGraph *h_coo_graph, CSCGraph *h_csc_graph,
                       unsigned int NUM_VERTICES, unsigned int BLOCK_SIZE,
                       bool useVertexCentric, bool usePush, bool useFrontier, const char *approachName,
                       unsigned int *numLevelsOut, unsigned int *visitedVerticesOut,
                       bool usePrivatizedFrontier = false)
{
    // Skip edge-centric pull as it's not implemented
    if (!useVertexCentric && !usePush && !useFrontier)
    {
        printf("Skipping %s (not implemented)\n", approachName);
        *numLevelsOut = 0;
        *visitedVerticesOut = 0;
        return -1.0f;
    }

    // Initialize level array for BFS
    unsigned int *h_level = initialize_level_array(NUM_VERTICES);
    h_level[0] = 0; // Start BFS from vertex 0

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime = 0.0f;

    // Start timing
    cudaEventRecord(start);

    // Perform BFS traversal with current approach
    unsigned int numLevels = perform_bfs(*h_graph, *h_coo_graph, *h_csc_graph, h_level, 0, NUM_VERTICES, BLOCK_SIZE,
                                         useVertexCentric, usePush, useFrontier, usePrivatizedFrontier);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Count vertices at each level
    unsigned int visitedVertices = 0;
    unsigned int *verticesPerLevel = count_vertices_per_level(h_level, NUM_VERTICES, numLevels, &visitedVertices);

    printf("\n%s approach:\n", approachName);
    printf("BFS traversal completed in %u levels\n", numLevels - 1);
    printf("Visited %u out of %u vertices\n", visitedVertices, NUM_VERTICES);
    printf("Execution time: %.3f ms\n", elapsedTime);

    // Set output parameters
    *numLevelsOut = numLevels;
    *visitedVerticesOut = visitedVertices;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_level);
    free(verticesPerLevel);

    return elapsedTime;
}

// Helper function to run all BFS approaches and print performance comparison
void run_and_compare_bfs_approaches(CSRGraph *h_graph, COOGraph *h_coo_graph, CSCGraph *h_csc_graph,
                                    unsigned int NUM_VERTICES, unsigned int BLOCK_SIZE)
{
    // Define the different BFS approaches to test
    // TODO: Convert graph approaches to enums
    const int NUM_APPROACHES = 6;  // Increased by 1 for privatized frontier
    bool useVertexCentric[NUM_APPROACHES] = {true, true, false, false, true, true};
    bool usePush[NUM_APPROACHES] = {true, false, true, false, true, true};
    bool useFrontier[NUM_APPROACHES] = {false, false, false, false, true, true};
    bool usePrivatizedFrontier[NUM_APPROACHES] = {false, false, false, false, false, true};
    const char *approachNames[NUM_APPROACHES] = {
        "Vertex-Centric Push",
        "Vertex-Centric Pull",
        "Edge-Centric Push",
        "Edge-Centric Pull",
        "Frontier-Based",
        "Privatized Frontier-Based"};

    // Arrays to store results
    float elapsedTime[NUM_APPROACHES];
    unsigned int numLevels[NUM_APPROACHES];
    unsigned int visitedVertices[NUM_APPROACHES];

    // Run each approach and measure performance
    for (int i = 0; i < NUM_APPROACHES; i++)
    {
        elapsedTime[i] = run_bfs_approach(h_graph, h_coo_graph, h_csc_graph, NUM_VERTICES, BLOCK_SIZE,
                                          useVertexCentric[i], usePush[i], useFrontier[i],
                                          approachNames[i], &numLevels[i], &visitedVertices[i],
                                          usePrivatizedFrontier[i]);
    }

    // Print performance comparison
    printf("\nPerformance Comparison:\n");
    printf("--------------------------------------------------\n");
    printf("Approach               | Execution Time (ms)\n");
    printf("--------------------------------------------------\n");
    for (int i = 0; i < NUM_APPROACHES; i++)
    {
        if (elapsedTime[i] >= 0)
        {
            printf("%-22s | %.3f\n", approachNames[i], elapsedTime[i]);
        }
        else
        {
            printf("%-22s | Not implemented\n", approachNames[i]);
        }
    }
    printf("--------------------------------------------------\n");
}

int main(int argc, char *argv[])
{
    unsigned int NUM_VERTICES;
    float EDGE_PROBABILITY;
    unsigned int BLOCK_SIZE;

    process_arguments(argc, argv, &NUM_VERTICES, &EDGE_PROBABILITY, &BLOCK_SIZE);

    // Construct all graph formats
    CSRGraph *h_graph = construct_csr_graph(NUM_VERTICES, EDGE_PROBABILITY);
    COOGraph *h_coo_graph = construct_coo_graph(NUM_VERTICES, EDGE_PROBABILITY);
    CSCGraph *h_csc_graph = construct_csc_graph(NUM_VERTICES, EDGE_PROBABILITY);

    run_and_compare_bfs_approaches(h_graph, h_coo_graph, h_csc_graph, NUM_VERTICES, BLOCK_SIZE);

    // Free all graph memory
    free_graph(h_graph);
    free_coo_graph(h_coo_graph);
    free_csc_graph(h_csc_graph);

    return 0;
}