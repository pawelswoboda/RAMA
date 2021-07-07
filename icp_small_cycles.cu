#include "icp_small_cycles.h"
#include "dCSR.h"
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include "time_measure_util.h"
#include <thrust/partition.h>
#include "utils.h"

static const float tol = 1e-6;

__device__ float get_CSR_value(const int row_index,
                                const int col_id,
                                const int* const __restrict__ row_offsets,
                                const int* const __restrict__ col_ids, 
                                const float* const __restrict__ data, 
                                int& found_index)
{
    for(int l = row_offsets[row_index]; l < row_offsets[row_index + 1]; ++l)
    {
        int current_col_id = col_ids[l]; 
        // TODO: Binary search
        // TODO: By finding collision between two sorted arrays.
        if (current_col_id > col_id) // col_ids are sorted.
            return 0.0f;

        if (current_col_id == col_id)
        {
            found_index = l;
            return data[l];
        }
    }
    return 0.0f;
}

__device__ float get_CSR_value_both_dir_geq_tol(const int row_index,
                                                const int col_id,
                                                const int* const __restrict__ row_offsets,
                                                const int* const __restrict__ col_ids, 
                                                const float* const __restrict__ data, 
                                                int& found_index)
{
    float val = get_CSR_value(row_index, col_id, row_offsets, col_ids, data, found_index);
    if (val < tol) // try other direction.
        val = get_CSR_value(col_id, row_index, row_offsets, col_ids, data, found_index);
    
    return val;
}

__device__ bool are_connected_by(const int v1, const int v2, const int mid, 
                                const int* const __restrict__ row_offsets, 
                                const int* const __restrict__ col_ids, 
                                const float* const __restrict__ data,
                                int& v1_mid_edge_index, int& v2_mid_edge_index,
                                float& v1_mid_edge_val, float& v2_mid_edge_val)
{
    v1_mid_edge_val = get_CSR_value_both_dir_geq_tol(v1, mid, row_offsets, col_ids, data, v1_mid_edge_index);
    if (v1_mid_edge_val < tol)
        return false;

    v2_mid_edge_val = get_CSR_value_both_dir_geq_tol(v2, mid, row_offsets, col_ids, data, v2_mid_edge_index);
    if (v2_mid_edge_val < tol)
        return false;

    return true;
}

__global__ void pack_triangles_parallel(const int num_rep_edges, 
                                    const int* const __restrict__ row_ids_p, 
                                    const int* const __restrict__ col_ids_p, 
                                    const int* const __restrict__ A_symm_row_offsets,
                                    const int* const __restrict__ A_symm_col_ids,
                                    const int* const __restrict__ A_dir_row_offsets, // adjacency matrix of original directed graph.
                                    const int* const __restrict__ A_dir_col_ids,
                                    float* __restrict__ A_dir_data,
                                    const int first_rep_edge_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid + first_rep_edge_index; edge < first_rep_edge_index + num_rep_edges; edge += num_threads) 
    {
        int rep_edge_row = row_ids_p[edge];
        int rep_edge_col = col_ids_p[edge];
        int rep_edge_index = -1;
        float rep_edge_cost = get_CSR_value(rep_edge_row, rep_edge_col, A_dir_row_offsets, A_dir_col_ids, A_dir_data, rep_edge_index);
        assert(rep_edge_cost < tol);
        assert(rep_edge_index >= 0); // The repulsive edge must also be present in A_dir.(row -> col).
        
        for(int l = A_symm_row_offsets[rep_edge_row]; l < A_symm_row_offsets[rep_edge_row + 1] && rep_edge_cost < -tol; ++l)
        {
            int current_col_id = A_symm_col_ids[l];
            int found_upper_index, found_lower_index;
            float upper_cost, lower_cost;
            bool connected = are_connected_by(rep_edge_row, rep_edge_col, current_col_id, 
                                            A_dir_row_offsets, A_dir_col_ids, A_dir_data,
                                            found_upper_index, found_lower_index,
                                            upper_cost, lower_cost);

            if (connected)
            {
                float packing_value = min(-rep_edge_cost, min(lower_cost, upper_cost));
                rep_edge_cost += packing_value;
                atomicAdd(&A_dir_data[found_upper_index], -packing_value);
                atomicAdd(&A_dir_data[found_lower_index], -packing_value);
                if (A_dir_data[found_upper_index] < 0 || A_dir_data[found_lower_index] < 0)
                {   // Undo:
                    rep_edge_cost -= packing_value;
                    atomicAdd(&A_dir_data[found_upper_index], packing_value);
                    atomicAdd(&A_dir_data[found_lower_index], packing_value);
                }
            }
        }
        A_dir_data[rep_edge_index] = rep_edge_cost;
        __syncthreads();
    }
}

__global__ void pack_quadrangles_parallel(const int num_rep_edges, 
    const int* const __restrict__ row_ids_p, 
    const int* const __restrict__ col_ids_p, 
    const int* const __restrict__ A_symm_row_offsets,
    const int* const __restrict__ A_symm_col_ids,
    const int* const __restrict__ A_dir_row_offsets, // adjacency matrix of original directed graph.
    const int* const __restrict__ A_dir_col_ids,
    float* __restrict__ A_dir_data,
    const int first_rep_edge_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid + first_rep_edge_index; edge < first_rep_edge_index + num_rep_edges; edge += num_threads) 
    {
        int v1 = row_ids_p[edge];
        int v2 = col_ids_p[edge];
        int rep_edge_index = -1;
        float rep_edge_cost = get_CSR_value(v1, v2, A_dir_row_offsets, A_dir_col_ids, A_dir_data, rep_edge_index);
        assert(rep_edge_cost < tol);
        assert(rep_edge_index >= 0); // The repulsive edge must also be present in A_dir.(row -> col).

        // Searching for a path like: v1 -(v1_n1_edge_index)- v1_n1 -(v1_n2_edge_index)- v1_n2 -(v2_edge_index)- v2.
        for(int l1 = A_symm_row_offsets[v1]; l1 < A_symm_row_offsets[v1 + 1] && rep_edge_cost < -tol; ++l1)
        {
            int v1_n1 = A_symm_col_ids[l1];
            int v1_n1_edge_index, v1_n2_edge_index, v2_edge_index; 
            float v1_n1_edge_cost = get_CSR_value_both_dir_geq_tol(v1, v1_n1, A_dir_row_offsets, A_dir_col_ids, A_dir_data, v1_n1_edge_index);
            int v1_n2;
            float v1_n2_edge_cost, v2_edge_cost;
            if (v1_n1_edge_cost > tol)
            {
                for(int l2 = A_symm_row_offsets[v1_n1]; l2 < A_symm_row_offsets[v1_n1 + 1] && rep_edge_cost < -tol; ++l2)
                {
                    v1_n2 = A_symm_col_ids[l2];
                    bool connected = are_connected_by(v1_n1, v2, v1_n2, 
                                                    A_dir_row_offsets, A_dir_col_ids, A_dir_data,
                                                    v1_n2_edge_index, v2_edge_index,
                                                    v1_n2_edge_cost, v2_edge_cost);

                    if (connected)
                    {
                        float packing_value = min(-rep_edge_cost, 
                                                min(v1_n1_edge_cost, 
                                                    min(v1_n2_edge_cost, v2_edge_cost)));

                        rep_edge_cost += packing_value;
                        atomicAdd(&A_dir_data[v1_n1_edge_index], -packing_value);
                        atomicAdd(&A_dir_data[v1_n2_edge_index], -packing_value);
                        atomicAdd(&A_dir_data[v2_edge_index], -packing_value);
                        if (A_dir_data[v1_n1_edge_index] < 0 || A_dir_data[v1_n2_edge_index] < 0 || A_dir_data[v2_edge_index] < 0)
                        {// Undo:
                            rep_edge_cost -= packing_value;
                            atomicAdd(&A_dir_data[v1_n1_edge_index], packing_value);
                            atomicAdd(&A_dir_data[v1_n2_edge_index], packing_value);
                            atomicAdd(&A_dir_data[v2_edge_index], packing_value);    
                        }
                    }
                }
            }
        }
        A_dir_data[rep_edge_index] = rep_edge_cost;
        __syncthreads();
    }
}

__global__ void pack_pentagons_parallel(const int num_rep_edges, 
    const int* const __restrict__ row_ids_p, 
    const int* const __restrict__ col_ids_p, 
    const int* const __restrict__ A_symm_row_offsets,
    const int* const __restrict__ A_symm_col_ids,
    const int* const __restrict__ A_dir_row_offsets, // adjacency matrix of original directed graph.
    const int* const __restrict__ A_dir_col_ids,
    float* __restrict__ A_dir_data,
    const int first_rep_edge_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid + first_rep_edge_index; edge < first_rep_edge_index + num_rep_edges; edge += num_threads) 
    {
        int v1 = row_ids_p[edge];
        int v2 = col_ids_p[edge];
        int rep_edge_index = -1;
        float rep_edge_cost = get_CSR_value(v1, v2, A_dir_row_offsets, A_dir_col_ids, A_dir_data, rep_edge_index);
        assert(rep_edge_cost < tol);
        assert(rep_edge_index >= 0); // The repulsive edge must also be present in A_dir.(row -> col).

        int v1_n1_edge_index, v1_n2_edge_index, v2_n1_edge_index, v2_edge_index; 
        // Searching for a path like: v1 -(v1_n1_edge_index)- v1_n1 -(v1_n2_edge_index)- v1_n2 -(v2_n1_edge_index)- v2_n1 -(v2_edge_index)-  v2.
        for(int l1 = A_symm_row_offsets[v1]; l1 < A_symm_row_offsets[v1 + 1] && rep_edge_cost < -tol; ++l1)
        {
            int v1_n1 = A_symm_col_ids[l1];
            float v1_n1_edge_cost = get_CSR_value_both_dir_geq_tol(v1, v1_n1, A_dir_row_offsets, A_dir_col_ids, A_dir_data, v1_n1_edge_index);
            if (v1_n1_edge_cost < tol)
                continue; 

            for(int l2 = A_symm_row_offsets[v2]; l2 < A_symm_row_offsets[v2 + 1] && rep_edge_cost < -tol; ++l2)
            {
                int v2_n1 = A_symm_col_ids[l2];
                float v2_edge_cost = get_CSR_value_both_dir_geq_tol(v2, v2_n1, A_dir_row_offsets, A_dir_col_ids, A_dir_data, v2_edge_index);
                if (v2_edge_cost < tol)
                    continue;

                for(int l3 = A_symm_row_offsets[v1_n1]; l3 < A_symm_row_offsets[v1_n1 + 1] && rep_edge_cost < -tol; ++l3)
                {
                    int v1_n2 = A_symm_col_ids[l3];
                    float v1_n2_edge_cost, v2_n1_edge_cost;
                    bool connected = are_connected_by(v1_n1, v2_n1, v1_n2, 
                                                    A_dir_row_offsets, A_dir_col_ids, A_dir_data,
                                                    v1_n2_edge_index, v2_n1_edge_index,
                                                    v1_n2_edge_cost, v2_n1_edge_cost);
                    
                    if (!connected)
                        continue;

                    float packing_value = min(-rep_edge_cost, 
                                            min(v1_n1_edge_cost, 
                                                min(v1_n2_edge_cost,
                                                    min(v2_n1_edge_cost, v2_edge_cost))));

                    rep_edge_cost += packing_value;
                    atomicAdd(&A_dir_data[v1_n1_edge_index], -packing_value);
                    atomicAdd(&A_dir_data[v1_n2_edge_index], -packing_value);
                    atomicAdd(&A_dir_data[v2_n1_edge_index], -packing_value);
                    atomicAdd(&A_dir_data[v2_edge_index], -packing_value);
                    if (A_dir_data[v1_n1_edge_index] < 0 || A_dir_data[v1_n2_edge_index] < 0 || A_dir_data[v2_edge_index] < 0 || A_dir_data[v2_n1_edge_index] < 0)
                    {// Undo:
                        rep_edge_cost -= packing_value;
                        atomicAdd(&A_dir_data[v1_n1_edge_index], packing_value);
                        atomicAdd(&A_dir_data[v1_n2_edge_index], packing_value);
                        atomicAdd(&A_dir_data[v2_n1_edge_index], packing_value);
                        atomicAdd(&A_dir_data[v2_edge_index], packing_value);    
                    }
                }
            }
        }
        A_dir_data[rep_edge_index] = rep_edge_cost;
        __syncthreads();
    }
}

struct is_positive_edge
{
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        if(thrust::get<2>(t) > 0.0f)
            return true;
        else
            return false;
    }
};

std::tuple<dCSR, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>, int> 
    create_matrices(cusparseHandle_t handle, const int num_nodes, const thrust::device_vector<int>& row_ids, const thrust::device_vector<int>& col_ids, const thrust::device_vector<float>& costs)
{
    MEASURE_FUNCTION_EXECUTION_TIME
    
    // Partition edges into positive and negative.
    thrust::device_vector<int> row_ids_p = row_ids;
    thrust::device_vector<int> col_ids_p = col_ids;
    thrust::device_vector<float> costs_p = costs;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids_p.begin(), col_ids_p.begin(), costs_p.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids_p.end(), col_ids_p.end(), costs_p.end()));

    auto first_negative = thrust::partition(first, last, is_positive_edge());
    const size_t nr_positive_edges = std::distance(first, first_negative);

    // Create symmetric adjacency matrix of positive edges.
    thrust::device_vector<int> pos_row_ids_symm, pos_col_ids_symm;
    thrust::device_vector<float> pos_costs_symm;

    std::tie(pos_row_ids_symm, pos_col_ids_symm, pos_costs_symm) = to_undirected(row_ids_p.begin(), row_ids_p.begin() + nr_positive_edges,
                                                                                col_ids_p.begin(), col_ids_p.begin() + nr_positive_edges,
                                                                                costs_p.begin(), costs_p.begin() + nr_positive_edges);
    dCSR A_pos = dCSR(handle, num_nodes, num_nodes, 
                    pos_col_ids_symm.begin(), pos_col_ids_symm.end(),
                    pos_row_ids_symm.begin(), pos_row_ids_symm.end(), 
                    pos_costs_symm.begin(), pos_costs_symm.end());

    return {A_pos, row_ids_p, col_ids_p, costs_p, nr_positive_edges};
}

// row_ids, col_ids, values should be directed thus containing same number of elements as in original problem.
std::tuple<double, dCSR> parallel_small_cycle_packing_cuda(cusparseHandle_t handle, 
    const thrust::device_vector<int>& row_ids, const thrust::device_vector<int>& col_ids, const thrust::device_vector<float>& costs, const int max_tries)
{
    MEASURE_FUNCTION_EXECUTION_TIME;

    int num_nodes = std::max(*thrust::max_element(row_ids.begin(), row_ids.end()), *thrust::max_element(col_ids.begin(), col_ids.end())) + 1;
    int num_edges = row_ids.size();

    // Make adjacency matrix and BFS search starting matrix.
    dCSR A_pos;
    thrust::device_vector<int> row_ids_p, col_ids_p;
    thrust::device_vector<float> costs_p;
    int nr_positive_edges;
    std::tie(A_pos, row_ids_p, col_ids_p, costs_p, nr_positive_edges) = create_matrices(handle, num_nodes, row_ids, col_ids, costs);

    int num_rep_edges = num_edges - nr_positive_edges;
 
    dCSR A_dir = dCSR(handle, num_nodes, num_nodes, 
                    col_ids.begin(), col_ids.end(), 
                    row_ids.begin(), row_ids.end(), 
                    costs.begin(), costs.end());
    
    int threadCount = 256;
    int blockCount = ceil(num_rep_edges / (float) threadCount);
    double lb = get_lb(A_dir.get_data());
    std::cout<<"Initial lb: "<<lb<<std::endl;

    for (int t = 0; t < max_tries; t++)
    {
        pack_triangles_parallel<<<blockCount, threadCount>>>(num_rep_edges, 
            thrust::raw_pointer_cast(row_ids_p.data()), 
            thrust::raw_pointer_cast(col_ids_p.data()), 
            A_pos.get_row_offsets_ptr(),
            A_pos.get_col_ids_ptr(),
            A_dir.get_row_offsets_ptr(),
            A_dir.get_col_ids_ptr(),
            A_dir.get_writeable_data_ptr(),
            nr_positive_edges);
        
        lb = get_lb(A_dir.get_data());
        std::cout<<"packing triangles, itr: "<<t<<", lb: "<<lb<<std::endl;
    }
    for (int t = 0; t < max_tries; t++)
    {
        pack_quadrangles_parallel<<<blockCount, threadCount>>>(num_rep_edges, 
            thrust::raw_pointer_cast(row_ids_p.data()), 
            thrust::raw_pointer_cast(col_ids_p.data()), 
            A_pos.get_row_offsets_ptr(),
            A_pos.get_col_ids_ptr(),
            A_dir.get_row_offsets_ptr(),
            A_dir.get_col_ids_ptr(),
            A_dir.get_writeable_data_ptr(),
            nr_positive_edges);
        
        lb = get_lb(A_dir.get_data());
        std::cout<<"packing quadrangles, itr: "<<t<<", lb: "<<lb<<std::endl;
    }

    return {lb, A_dir};
}

double compute_lower_bound(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const int max_tries)
{
    const int cuda_device = get_cuda_device();
    cudaSetDevice(cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);
    std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << ", device number " << cuda_device << "\n";
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");
    
    const thrust::device_vector<int> i_d = i;
    const thrust::device_vector<int> j_d = j;
    const thrust::device_vector<float> costs_d = costs;

    double lb;
    dCSR A_new;

    std::tie(lb, A_new) = parallel_small_cycle_packing_cuda(handle, i_d, j_d, costs_d, max_tries);

    return lb;
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> parallel_small_cycle_packing_costs(cusparseHandle_t handle, 
                    const thrust::device_vector<int>& row_ids, const thrust::device_vector<int>& col_ids, const thrust::device_vector<float>& costs, const int max_tries)
{
    double lb;
    dCSR A_new;

    std::tie(lb, A_new) = parallel_small_cycle_packing_cuda(handle, row_ids, col_ids, costs, max_tries);

    return A_new.export_coo(handle);
}