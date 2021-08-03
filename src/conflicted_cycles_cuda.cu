#include "conflicted_cycles_cuda.h"
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/binary_search.h>
#include "time_measure_util.h"
#include <thrust/partition.h>
#include "parallel_gaec_utils.h"
#include <thrust/execution_policy.h>

// Assumes a symmetric CSR matrix.
// Initialize v1_mid_edge_index by row_offsets[v1] and v2_mid_edge_index by row_offsets[v2].
__device__ int compute_lowest_common_neighbour(const int v1, const int v2, 
                                            const int* const __restrict__ row_offsets, 
                                            const int* const __restrict__ col_ids, 
                                            const float* const __restrict__ data,
                                            int& v1_mid_edge_index, int& v2_mid_edge_index)
{
    while(v1_mid_edge_index < row_offsets[v1 + 1] && v2_mid_edge_index < row_offsets[v2 + 1])
    {
        const int v1_n = col_ids[v1_mid_edge_index];
        const int v2_n = col_ids[v2_mid_edge_index];
        if (v1_n == v2_n)
        {
            v1_mid_edge_index++;
            v2_mid_edge_index++;
            return v1_n;
        }
        if (v1_n < v2_n)
            ++v1_mid_edge_index;
        if (v1_n > v2_n)
            ++v2_mid_edge_index;        
    }
    return -1;
}

__device__ bool write_triangle(int* const __restrict__ tri_v1, 
                            int* const __restrict__ tri_v2, 
                            int* const __restrict__ tri_v3, 
                            int* __restrict__ empty_tri_index, 
                            const int max_triangles,
                            const int v1, const int v2, const int v3)
{
    const int old_index = atomicAdd(empty_tri_index, 1);
    if (old_index >= max_triangles)
        return true;
    const int min_v = min(v1, min(v2, v3));
    const int max_v = max(v1, max(v2, v3));
    tri_v1[old_index] = min_v;
    tri_v2[old_index] = max(min(v1, v2), min(max(v1, v2), v3));
    tri_v3[old_index] = max_v;
    return false;
}

__global__ void find_triangles_parallel(const int num_rep_edges,
                                    const int* const __restrict__ row_ids_rep, 
                                    const int* const __restrict__ col_ids_rep, 
                                    const int* const __restrict__ A_symm_row_offsets,
                                    const int* const __restrict__ A_symm_col_ids,
                                    const float* const __restrict__ A_symm_data,
                                    int* __restrict__ triangle_v1,
                                    int* __restrict__ triangle_v2,
                                    int* __restrict__ triangle_v3,
                                    int* __restrict__ empty_tri_index,
                                    const int max_triangles)
{
    const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    for (int edge = start_index; edge < num_rep_edges && empty_tri_index[0] < max_triangles; edge += num_threads) 
    {
        const int v1 = row_ids_rep[edge];
        const int v2 = col_ids_rep[edge];
        int v1_mid_edge_index = A_symm_row_offsets[v1];
        int v2_mid_edge_index = A_symm_row_offsets[v2];
        bool filled = false;
        while(!filled)
        {
            const int mid = compute_lowest_common_neighbour(v1, v2, A_symm_row_offsets, A_symm_col_ids, A_symm_data, v1_mid_edge_index, v2_mid_edge_index);
            if (mid == -1)
                break;
            
            filled = write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, max_triangles, v1, v2, mid);
        }
    }
}

__global__ void find_quadrangles_parallel(const long num_expansions, const int num_rep_edges,
                                        const int* const __restrict__ row_ids_rep,
                                        const int* const __restrict__ col_ids_rep,
                                        const long* const __restrict__ rep_row_offsets,
                                        const int* const __restrict__ A_symm_row_offsets,
                                        const int* const __restrict__ A_symm_col_ids,
                                        const float* const __restrict__ A_symm_data,
                                        int* __restrict__ triangle_v1,
                                        int* __restrict__ triangle_v2,
                                        int* __restrict__ triangle_v3,
                                        int* __restrict__ empty_tri_index,
                                        const int max_triangles)
{
    const long start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const long num_threads = blockDim.x * gridDim.x;
    for (long c_index = start_index; c_index < num_expansions && empty_tri_index[0] < max_triangles; c_index += num_threads) 
    {
        const long* next_rep_row_location = thrust::upper_bound(thrust::seq, rep_row_offsets, rep_row_offsets + num_rep_edges + 1, c_index);
        const long rep_edge_index = thrust::distance(rep_row_offsets, next_rep_row_location) - 1;
        assert(rep_edge_index < num_rep_edges && rep_edge_index >= 0);
        const long local_offset = c_index - rep_row_offsets[rep_edge_index];
        assert(local_offset >= 0);
        const int v1 = row_ids_rep[rep_edge_index];
        const int v2 = col_ids_rep[rep_edge_index];
        const int v1_n1 = A_symm_col_ids[A_symm_row_offsets[v1] + local_offset];
        int v1_n1_mid_edge_index = A_symm_row_offsets[v1_n1];
        int v2_mid_edge_index = A_symm_row_offsets[v2];
        bool filled = false;
        while(!filled)
        {
            const int mid = compute_lowest_common_neighbour(v1_n1, v2, A_symm_row_offsets, A_symm_col_ids, A_symm_data, v1_n1_mid_edge_index, v2_mid_edge_index);
            if (mid == -1)
                break;

            write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, max_triangles, v1, v2, v1_n1);
            filled = write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, max_triangles, v2, v1_n1, mid);
        }
    }
}

__global__ void find_pentagons_parallel(const int num_expansions, const int num_rep_edges,
                                        const int* const __restrict__ row_ids_rep,
                                        const int* const __restrict__ col_ids_rep,
                                        const int* const __restrict__ rep_edge_offsets,
                                        const int* const __restrict__ A_symm_row_offsets,
                                        const int* const __restrict__ A_symm_col_ids,
                                        const float* const __restrict__ A_symm_data,
                                        int* __restrict__ triangle_v1,
                                        int* __restrict__ triangle_v2,
                                        int* __restrict__ triangle_v3,
                                        int* __restrict__ empty_tri_index,
                                        const int max_triangles)
{
    const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    for (int c_index = start_index; c_index < num_expansions && empty_tri_index[0] < max_triangles; c_index += num_threads) 
    {
        const int* next_rep_edge_location = thrust::upper_bound(thrust::seq, rep_edge_offsets, rep_edge_offsets + num_rep_edges + 1, c_index);
        const int rep_edge_index = thrust::distance(rep_edge_offsets, next_rep_edge_location) - 1;
        assert(rep_edge_index < num_rep_edges);
        const int local_offset = c_index - rep_edge_offsets[rep_edge_index];
        assert(local_offset >= 0);
        const int v1 = row_ids_rep[rep_edge_index];
        const int v2 = col_ids_rep[rep_edge_index];
        const int v1_degree = A_symm_row_offsets[v1 + 1] - A_symm_row_offsets[v1];
        const int l1 = local_offset % v1_degree;
        const int l2 = local_offset / v1_degree;
        const int v1_n1 = A_symm_col_ids[A_symm_row_offsets[v1] + l1];
        const int v2_n1 = A_symm_col_ids[A_symm_row_offsets[v2] + l2];
        if (v1_n1 == v2_n1)
            continue;
        int v1_n1_mid_edge_index = A_symm_row_offsets[v1_n1];
        int v2_n1_mid_edge_index = A_symm_row_offsets[v2_n1];
        bool filled = false;
        while(!filled)
        {
            const int mid = compute_lowest_common_neighbour(v1_n1, v2_n1, A_symm_row_offsets, A_symm_col_ids, A_symm_data, v1_n1_mid_edge_index, v2_n1_mid_edge_index);
            if (mid == -1)
                break;

            write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, max_triangles, v1, v2, v1_n1);
            write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, max_triangles, v2, v1_n1, mid);
            filled = write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, max_triangles, v2, mid, v2_n1);
        }
    }
}

struct is_positive_edge
{
    const float tol;
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) > tol;
    }
};

struct is_neg_edge
{
    const float tol;
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) < tol;
    }
};

struct discard_edge
{
    const bool* v_present;
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        const int i = thrust::get<0>(t);
        const int j = thrust::get<1>(t);
        return !(v_present[i] && v_present[j]);
    }
};

float determine_threshold(const dCOO& A, const float tol_ratio)
{
    const float min_edge_cost = A.min();
    return min_edge_cost * tol_ratio;
}

std::tuple<dCOO, thrust::device_vector<int>, thrust::device_vector<int>> 
    create_matrices(const dCOO& A, const float thresh)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    
    // Partition edges into positive and negative.
    thrust::device_vector<int> row_ids = A.get_row_ids();
    thrust::device_vector<int> col_ids = A.get_col_ids();
    thrust::device_vector<float> costs = A.get_data();

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin(), costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end(), costs.end()));

    thrust::device_vector<int> row_ids_pos(row_ids.size());
    thrust::device_vector<int> col_ids_pos(row_ids.size());
    thrust::device_vector<float> costs_pos(row_ids.size());
    auto first_pos = thrust::make_zip_iterator(thrust::make_tuple(row_ids_pos.begin(), col_ids_pos.begin(), costs_pos.begin()));
    auto last_pos = thrust::copy_if(first, last, first_pos, is_positive_edge({-1.0f * thresh}));
    const int num_positive_edges = std::distance(first_pos, last_pos);
    row_ids_pos.resize(num_positive_edges);
    col_ids_pos.resize(num_positive_edges);
    costs_pos.resize(num_positive_edges);

    thrust::device_vector<int> row_ids_neg(row_ids.size());
    thrust::device_vector<int> col_ids_neg(row_ids.size());
    thrust::device_vector<float> costs_neg(row_ids.size());
    auto first_neg = thrust::make_zip_iterator(thrust::make_tuple(row_ids_neg.begin(), col_ids_neg.begin(), costs_neg.begin()));
    auto last_neg = thrust::copy_if(first, last, first_neg, is_neg_edge({thresh}));
    const int nr_neg_edges = std::distance(first_neg, last_neg);
    row_ids_neg.resize(nr_neg_edges);
    col_ids_neg.resize(nr_neg_edges);

    // Create symmetric adjacency matrix of positive edges.
    dCOO A_pos_symm;
    if (num_positive_edges > 0)
    {
        std::tie(row_ids_pos, col_ids_pos, costs_pos) = to_undirected(row_ids_pos, col_ids_pos, costs_pos);
        A_pos_symm = dCOO(A.max_dim(), A.max_dim(),
                        std::move(col_ids_pos),
                        std::move(row_ids_pos), 
                        std::move(costs_pos), false);
    }
    return {A_pos_symm, row_ids_neg, col_ids_neg};
}

// A should be directed thus containing same number of elements as in original problem.
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> conflicted_cycles_cuda(const dCOO& A, const int max_cycle_length, const float tri_memory_factor, const float tol_ratio)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    if (max_cycle_length > 5)
        throw std::runtime_error("max_cycle_length should be <= 5.");
    if (max_cycle_length < 3)
        return {thrust::device_vector<int>(0), thrust::device_vector<int>(0), thrust::device_vector<int>(0)};

    // Make adjacency matrix and BFS search starting matrix.
    dCOO A_pos;
    thrust::device_vector<int> row_ids_rep, col_ids_rep;
    const float min_rep_thresh = determine_threshold(A, tol_ratio);
    std::tie(A_pos, row_ids_rep, col_ids_rep) = create_matrices(A, min_rep_thresh);
    int num_rep_edges = row_ids_rep.size();
    if (A_pos.nnz() == 0 || num_rep_edges == 0)
        return {thrust::device_vector<int>(0), thrust::device_vector<int>(0), thrust::device_vector<int>(0)};

    thrust::device_vector<int> A_pos_row_offsets = A_pos.compute_row_offsets();
    assert(A_pos_row_offsets.size() == A.max_dim() + 1);

    int threadCount = 256;
    int blockCount = ceil(num_rep_edges / (float) threadCount);

    const int max_num_tri = tri_memory_factor * num_rep_edges; // For memory pre-allocation.
    thrust::device_vector<int> triangles_v1(max_num_tri);
    thrust::device_vector<int> triangles_v2(max_num_tri); 
    thrust::device_vector<int> triangles_v3(max_num_tri);
    thrust::device_vector<int> empty_tri_index(1, 0);

    find_triangles_parallel<<<blockCount, threadCount>>>(num_rep_edges, 
        thrust::raw_pointer_cast(row_ids_rep.data()), 
        thrust::raw_pointer_cast(col_ids_rep.data()), 
        thrust::raw_pointer_cast(A_pos_row_offsets.data()),
        A_pos.get_col_ids_ptr(),
        A_pos.get_data_ptr(),
        thrust::raw_pointer_cast(triangles_v1.data()),
        thrust::raw_pointer_cast(triangles_v2.data()),
        thrust::raw_pointer_cast(triangles_v3.data()),
        thrust::raw_pointer_cast(empty_tri_index.data()),
        triangles_v1.size());
    
    std::cout<<"3-cycles: found # of triangles: "<<empty_tri_index[0]<<", budget: "<<triangles_v1.size()<<std::endl;

    if (max_cycle_length >= 4 && empty_tri_index[0] < triangles_v1.size())
    {
        // Move valid triangles to starting indices to increase the budget.
        empty_tri_index[0] = rearrange_triangles(triangles_v1, triangles_v2, triangles_v3, empty_tri_index[0]); 
        thrust::device_vector<long> rep_row_offsets(num_rep_edges + 1);
        {
            const thrust::device_vector<int> vertex_degrees = offsets_to_degrees(A_pos_row_offsets);
            thrust::gather(row_ids_rep.begin(), row_ids_rep.end(), vertex_degrees.begin(), rep_row_offsets.begin());

            rep_row_offsets.back() = 0;
            thrust::exclusive_scan(rep_row_offsets.begin(), rep_row_offsets.end(), rep_row_offsets.begin());
        }
        const long num_expansions = rep_row_offsets.back();
        blockCount = ceil(num_expansions / (float) threadCount);
        std::cout<<"4-cycles: number of expansions: "<<num_expansions<<"\n";

        find_quadrangles_parallel<<<blockCount, threadCount>>>(num_expansions, num_rep_edges,
            thrust::raw_pointer_cast(row_ids_rep.data()),
            thrust::raw_pointer_cast(col_ids_rep.data()),
            thrust::raw_pointer_cast(rep_row_offsets.data()), 
            thrust::raw_pointer_cast(A_pos_row_offsets.data()),
            A_pos.get_col_ids_ptr(),
            A_pos.get_data_ptr(),
            thrust::raw_pointer_cast(triangles_v1.data()),
            thrust::raw_pointer_cast(triangles_v2.data()),
            thrust::raw_pointer_cast(triangles_v3.data()),
            thrust::raw_pointer_cast(empty_tri_index.data()),
            triangles_v1.size());
        
        std::cout<<"4-cycles: found # of triangles: "<<empty_tri_index[0]<<", budget: "<<triangles_v1.size()<<std::endl;
    }

    if (max_cycle_length >= 5 && empty_tri_index[0] < triangles_v1.size())
    {
        empty_tri_index[0] = rearrange_triangles(triangles_v1, triangles_v2, triangles_v3, empty_tri_index[0]);
        thrust::device_vector<int> rep_edge_offsets(num_rep_edges + 1);
        {
            const thrust::device_vector<int> vertex_degrees = offsets_to_degrees(A_pos_row_offsets);
            thrust::device_vector<int> row_ids_degrees(num_rep_edges);
            thrust::gather(row_ids_rep.begin(), row_ids_rep.end(), vertex_degrees.begin(), row_ids_degrees.begin());
            thrust::device_vector<int> col_ids_degrees(num_rep_edges);
            thrust::gather(col_ids_rep.begin(), col_ids_rep.end(), vertex_degrees.begin(), col_ids_degrees.begin());

            thrust::transform(row_ids_degrees.begin(), row_ids_degrees.end(), col_ids_degrees.begin(), rep_edge_offsets.begin(), thrust::multiplies<int>());
            rep_edge_offsets.back() = 0;
            thrust::exclusive_scan(rep_edge_offsets.begin(), rep_edge_offsets.end(), rep_edge_offsets.begin());
        }
        const int num_expansions = rep_edge_offsets.back();
        blockCount = ceil(num_expansions / (float) threadCount);
        std::cout<<"5-cycles: number of expansions: "<<num_expansions<<"\n";
        find_pentagons_parallel<<<blockCount, threadCount>>>(num_expansions, num_rep_edges,
            thrust::raw_pointer_cast(row_ids_rep.data()),
            thrust::raw_pointer_cast(col_ids_rep.data()),
            thrust::raw_pointer_cast(rep_edge_offsets.data()), 
            thrust::raw_pointer_cast(A_pos_row_offsets.data()),
            A_pos.get_col_ids_ptr(),
            A_pos.get_data_ptr(),
            thrust::raw_pointer_cast(triangles_v1.data()),
            thrust::raw_pointer_cast(triangles_v2.data()),
            thrust::raw_pointer_cast(triangles_v3.data()),
            thrust::raw_pointer_cast(empty_tri_index.data()),
            triangles_v1.size());
        
        std::cout<<"5-cycles: found # of triangles: "<<empty_tri_index[0]<<", budget: "<<triangles_v1.size()<<std::endl;
    }

    int nr_triangles = empty_tri_index[0];
    triangles_v1.resize(nr_triangles);
    triangles_v2.resize(nr_triangles);
    triangles_v3.resize(nr_triangles);
    return {triangles_v1, triangles_v2, triangles_v3};
}