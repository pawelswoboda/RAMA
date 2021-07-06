#include <iostream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "../time_measure_util.h"
#include "../utils.h"

#define numThreads 256

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void pick_best_edge(const int num_edges, 
                            const int* const __restrict__ row_ids, 
                            const int* const __restrict__ col_ids, 
                            const float* const __restrict__ costs, 
                            int* __restrict__ highest_edge_index, 
                            float* __restrict__ highest_edge_value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid; edge < num_edges; edge += num_threads) 
    {
        //TODO: can possibly do better by using shared memory and max reduction as input is coo sorted.
        int r = row_ids[edge];
        int c = col_ids[edge];
        float w = costs[edge];
        atomicMax(&highest_edge_value[r], w);
        if (highest_edge_value[r] == w)
            highest_edge_index[r] = edge;
        // other direction:
        atomicMax(&highest_edge_value[c], w);
        if (highest_edge_value[c] == w)
            highest_edge_index[c] = edge;
        __syncthreads(); 
    }
}

__global__ void mark_matches(const int num_edges, 
                            const int* const __restrict__ row_ids, 
                            const int* const __restrict__ col_ids, 
                            const int* const __restrict__ highest_edge_index,
                            int* __restrict__ matched_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid; edge < num_edges; edge += num_threads) 
    {
        int r = row_ids[edge];
        int c = col_ids[edge];
        if (highest_edge_index[r] == highest_edge_index[c] && highest_edge_index[c] >= 0)
            matched_edges[edge] = 1;
        __syncthreads(); 
    }
}

struct remove_reverse_edges_func {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,float> e)
        {
            return thrust::get<0>(e) > thrust::get<1>(e);
        }
};

struct is_unmatched {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,int> e)
        {
            return thrust::get<2>(e) == 0;
        }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> maximum_matching_edge_based(const int numVertices, const int num_edges, const thrust::device_vector<int>& row_ids, const thrust::device_vector<int>& col_ids, const thrust::device_vector<float>& costs)
{
    thrust::device_vector<int> row_ids_directed = row_ids;
    thrust::device_vector<int> col_ids_directed = col_ids;
    thrust::device_vector<float> costs_directed = costs;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids_directed.begin(), col_ids_directed.begin(), costs_directed.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids_directed.end(), col_ids_directed.end(), costs_directed.end()));
    auto directed_last = thrust::remove_if(first, last, remove_reverse_edges_func());
    row_ids_directed.resize(std::distance(first, directed_last));
    col_ids_directed.resize(std::distance(first, directed_last));
    costs_directed.resize(std::distance(first, directed_last)); 

    const int num_dir_edges = row_ids_directed.size();
    //TODO: Permute the edge indices so that consecutive threads have less chance of writing onto the same vertex due to atomicMax.
    thrust::device_vector<int> highest_edge_index(numVertices, -1);
    thrust::device_vector<float> highest_edge_value(numVertices, 0.0f);
    int numBlocks = ceil(num_dir_edges / (float) numThreads);

    pick_best_edge<<<numBlocks, numThreads>>>(num_dir_edges, 
                                                thrust::raw_pointer_cast(row_ids_directed.data()), 
                                                thrust::raw_pointer_cast(col_ids_directed.data()), 
                                                thrust::raw_pointer_cast(costs_directed.data()), 
                                                thrust::raw_pointer_cast(highest_edge_index.data()),
                                                thrust::raw_pointer_cast(highest_edge_value.data()));

    thrust::device_vector<int> matched_edges(num_dir_edges, 0);
    mark_matches<<<numBlocks, numThreads>>>(num_dir_edges, 
        thrust::raw_pointer_cast(row_ids_directed.data()), 
        thrust::raw_pointer_cast(col_ids_directed.data()), 
        thrust::raw_pointer_cast(highest_edge_index.data()),
        thrust::raw_pointer_cast(matched_edges.data()));

    std::cout << "matched sum = " << thrust::reduce(matched_edges.begin(), matched_edges.end(), 0) << "\n";

    auto first_m = thrust::make_zip_iterator(thrust::make_tuple(row_ids_directed.begin(), col_ids_directed.begin(), matched_edges.begin()));
    auto last_m = thrust::make_zip_iterator(thrust::make_tuple(row_ids_directed.end(), col_ids_directed.end(), matched_edges.end()));
    auto matched_last = thrust::remove_if(first_m, last_m, is_unmatched());
    const int nr_matched_edges = std::distance(first_m, matched_last);
    row_ids_directed.resize(std::distance(first_m, matched_last));
    col_ids_directed.resize(std::distance(first_m, matched_last));
    // costs_directed.resize(std::distance(first, new_last)); 
    
    //TODO: Iterate and find more edges?

    std::cout << "# vertices = " << numVertices << "\n";
    std::cout << "matched = " << nr_matched_edges << " / "<< num_dir_edges<< "\n";

    return to_undirected(row_ids_directed, col_ids_directed);
}