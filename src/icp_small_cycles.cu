#include "icp_small_cycles.h"
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include "time_measure_util.h"
#include <thrust/partition.h>
#include "parallel_gaec_utils.h"

#define tol 1e-6 

__device__ float get_CSR_value(const int row_index,
                                const int col_id,
                                const int A_num_rows,
                                const int* const __restrict__ row_offsets,
                                const int* const __restrict__ col_ids, 
                                const float* const __restrict__ data, 
                                int& found_index)
{
    if (row_index >= A_num_rows)
        return 0.0;

    for(int l = row_offsets[row_index]; l < row_offsets[row_index + 1]; ++l)
    {
        const int current_col_id = col_ids[l]; 
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
                                                const int A_num_rows,
                                                const int* const __restrict__ row_offsets,
                                                const int* const __restrict__ col_ids, 
                                                const float* const __restrict__ data, 
                                                int& found_index)
{
    float val = get_CSR_value(row_index, col_id, A_num_rows, row_offsets, col_ids, data, found_index);
    if (val < tol) // try other direction.
        val = get_CSR_value(col_id, row_index, A_num_rows, row_offsets, col_ids, data, found_index);
    
    return val;
}

__device__ bool are_connected_by(const int v1, const int v2, const int mid, 
                                const int* const __restrict__ row_offsets, 
                                const int* const __restrict__ col_ids, 
                                const float* const __restrict__ data, const int A_num_rows,
                                int& v1_mid_edge_index, int& v2_mid_edge_index,
                                float& v1_mid_edge_val, float& v2_mid_edge_val)
{
    v1_mid_edge_val = get_CSR_value_both_dir_geq_tol(v1, mid, A_num_rows, row_offsets, col_ids, data, v1_mid_edge_index);
    if (v1_mid_edge_val < tol)
        return false;

    v2_mid_edge_val = get_CSR_value_both_dir_geq_tol(v2, mid, A_num_rows, row_offsets, col_ids, data, v2_mid_edge_index);
    if (v2_mid_edge_val < tol)
        return false;

    return true;
}

// TODO: would be useful if we only want to enumerate triangles and not packing.
// Assumes a symmetric CSR matrix.
// Initialize v1_mid_edge_index by row_offsets[v1] and v2_mid_edge_index by row_offsets[v2].
__device__ int compute_lowest_common_neighbour(const int v1, const int v2, 
                                            const int* const __restrict__ row_offsets, 
                                            const int* const __restrict__ col_ids, 
                                            const float* const __restrict__ data, const int A_num_rows,
                                            int& v1_mid_edge_index, int& v2_mid_edge_index,
                                            float& v1_mid_edge_val, float& v2_mid_edge_val)
{
    while(v1_mid_edge_index < row_offsets[v1 + 1] && v2_mid_edge_index < row_offsets[v2 + 1])
    {
        int v1_n = col_ids[v1_mid_edge_index];
        int v2_n = col_ids[v2_mid_edge_index];
        if (v1_n == v2_n)
        {
            v1_mid_edge_val = data[v1_mid_edge_index++];
            v2_mid_edge_val = data[v2_mid_edge_index++];
            return v1_n;
        }
        else if (v1_n < v2_n)
        {
            ++v1_mid_edge_index;
        }
        else
        {
            ++v2_mid_edge_index;
        }
    }
    return -1;
}

__device__ void write_triangle(int* const __restrict__ tri_v1, 
                            int* const __restrict__ tri_v2, 
                            int* const __restrict__ tri_v3, 
                            int* __restrict__ empty_tri_index, 
                            const int v1, const int v2, const int v3)
{
    const int old_index = atomicAdd(empty_tri_index, 1);
    const int min_v = min(v1, min(v2, v3));
    const int max_v = max(v1, max(v2, v3));
    tri_v1[old_index] = min_v;
    tri_v2[old_index] = v1 - min_v + v2 - max_v + v3;
    tri_v3[old_index] = max_v;
}

__global__ void pack_triangles_parallel(const int num_rep_edges,
                                    const int* const __restrict__ row_ids_rep, 
                                    const int* const __restrict__ col_ids_rep, 
                                    const int* const __restrict__ A_symm_row_offsets,
                                    const int* const __restrict__ A_symm_col_ids,
                                    const int* const __restrict__ A_row_offsets, // adjacency matrix of original directed graph.
                                    const int* const __restrict__ A_col_ids,
                                    float* __restrict__ A_data,
                                    const int first_rep_edge_index,
                                    const int A_num_rows,
                                    int* __restrict__ triangle_v1,
                                    int* __restrict__ triangle_v2,
                                    int* __restrict__ triangle_v3,
                                    int* __restrict__ empty_tri_index,
                                    const int max_triangles)
{
    const int start_index = blockIdx.x * blockDim.x + threadIdx.x + first_rep_edge_index;
    const int num_threads = blockDim.x * gridDim.x;

    for (int edge = start_index; edge < first_rep_edge_index + num_rep_edges; edge += num_threads) 
    {
        const int rep_edge_row = row_ids_rep[edge];
        const int rep_edge_col = col_ids_rep[edge];
        int rep_edge_index = -1;
        float rep_edge_cost = get_CSR_value(rep_edge_row, rep_edge_col, A_num_rows, A_row_offsets, A_col_ids, A_data, rep_edge_index);
        assert(rep_edge_cost < tol);
        assert(rep_edge_index >= 0); // The repulsive edge must also be present in A.(row -> col).
        
        for(int l = A_symm_row_offsets[rep_edge_row]; l < A_symm_row_offsets[rep_edge_row + 1] && rep_edge_cost < -tol; ++l)
        {
            const int current_col_id = A_symm_col_ids[l];
            int found_upper_index, found_lower_index;
            float upper_cost, lower_cost;
            const bool connected = are_connected_by(rep_edge_row, rep_edge_col, current_col_id, 
                                            A_row_offsets, A_col_ids, A_data, A_num_rows,
                                            found_upper_index, found_lower_index,
                                            upper_cost, lower_cost);

            if (connected)
            {
                const float packing_value = min(-rep_edge_cost, min(lower_cost, upper_cost));
                rep_edge_cost += packing_value;
                atomicAdd(&A_data[found_upper_index], -packing_value);
                atomicAdd(&A_data[found_lower_index], -packing_value);
                if (A_data[found_upper_index] < 0 || A_data[found_lower_index] < 0)
                {   // Undo:
                    rep_edge_cost -= packing_value;
                    atomicAdd(&A_data[found_upper_index], packing_value);
                    atomicAdd(&A_data[found_lower_index], packing_value);
                }
                if (empty_tri_index[0] < max_triangles)
                    write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, rep_edge_row, current_col_id, rep_edge_col);
            }
        }
        A_data[rep_edge_index] = rep_edge_cost;
        __syncthreads();
    }
}

__global__ void pack_quadrangles_parallel(const int num_rep_edges, 
                                        const int* const __restrict__ row_ids_rep, 
                                        const int* const __restrict__ col_ids_rep, 
                                        const int* const __restrict__ A_symm_row_offsets,
                                        const int* const __restrict__ A_symm_col_ids,
                                        const int* const __restrict__ A_row_offsets, // adjacency matrix of original directed graph.
                                        const int* const __restrict__ A_col_ids,
                                        float* __restrict__ A_data,
                                        const int first_rep_edge_index,
                                        const int A_num_rows,
                                        int* __restrict__ triangle_v1,
                                        int* __restrict__ triangle_v2,
                                        int* __restrict__ triangle_v3,
                                        int* __restrict__ empty_tri_index,
                                        const int max_triangles)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid + first_rep_edge_index; edge < first_rep_edge_index + num_rep_edges; edge += num_threads) 
    {
        const int v1 = row_ids_rep[edge];
        const int v2 = col_ids_rep[edge];
        int rep_edge_index = -1;
        float rep_edge_cost = get_CSR_value(v1, v2, A_num_rows, A_row_offsets, A_col_ids, A_data, rep_edge_index);
        assert(rep_edge_cost < tol);
        assert(rep_edge_index >= 0); // The repulsive edge must also be present in A.(row -> col).

        // Searching for a path like: v1 -(v1_n1_edge_index)- v1_n1 -(v1_n2_edge_index)- v1_n2 -(v2_edge_index)- v2.
        for(int l1 = A_symm_row_offsets[v1]; l1 < A_symm_row_offsets[v1 + 1] && rep_edge_cost < -tol; ++l1)
        {
            const int v1_n1 = A_symm_col_ids[l1];
            int v1_n1_edge_index, v1_n2_edge_index, v2_edge_index; 
            const float v1_n1_edge_cost = get_CSR_value_both_dir_geq_tol(v1, v1_n1, A_num_rows, A_row_offsets, A_col_ids, A_data, v1_n1_edge_index);
            int v1_n2;
            float v1_n2_edge_cost, v2_edge_cost;
            if (v1_n1_edge_cost > tol)
            {
                for(int l2 = A_symm_row_offsets[v1_n1]; l2 < A_symm_row_offsets[v1_n1 + 1] && rep_edge_cost < -tol; ++l2)
                {
                    v1_n2 = A_symm_col_ids[l2];
                    const bool connected = are_connected_by(v1_n1, v2, v1_n2, 
                                                    A_row_offsets, A_col_ids, A_data, A_num_rows,
                                                    v1_n2_edge_index, v2_edge_index,
                                                    v1_n2_edge_cost, v2_edge_cost);

                    if (connected)
                    {
                        const float packing_value = min(-rep_edge_cost, 
                                                min(v1_n1_edge_cost, 
                                                    min(v1_n2_edge_cost, v2_edge_cost)));

                        rep_edge_cost += packing_value;
                        atomicAdd(&A_data[v1_n1_edge_index], -packing_value);
                        atomicAdd(&A_data[v1_n2_edge_index], -packing_value);
                        atomicAdd(&A_data[v2_edge_index], -packing_value);
                        if (A_data[v1_n1_edge_index] < 0 || A_data[v1_n2_edge_index] < 0 || A_data[v2_edge_index] < 0)
                        {// Undo:
                            rep_edge_cost -= packing_value;
                            atomicAdd(&A_data[v1_n1_edge_index], packing_value);
                            atomicAdd(&A_data[v1_n2_edge_index], packing_value);
                            atomicAdd(&A_data[v2_edge_index], packing_value);    
                        }

                        if (empty_tri_index[0] < max_triangles)
                            write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, v1, v1_n1, v2);
                        if (empty_tri_index[0] < max_triangles)
                            write_triangle(triangle_v1, triangle_v2, triangle_v3, empty_tri_index, v1_n1, v1_n2, v2);
                    }
                }
            }
        }
        A_data[rep_edge_index] = rep_edge_cost;
        __syncthreads();
    }
}

__global__ void pack_pentagons_parallel(const int num_rep_edges, 
    const int* const __restrict__ row_ids_rep, 
    const int* const __restrict__ col_ids_rep, 
    const int* const __restrict__ A_symm_row_offsets,
    const int* const __restrict__ A_symm_col_ids,
    const int* const __restrict__ A_row_offsets, // adjacency matrix of original directed graph.
    const int* const __restrict__ A_col_ids,
    float* __restrict__ A_data,
    const int first_rep_edge_index,
    const int A_num_rows)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid + first_rep_edge_index; edge < first_rep_edge_index + num_rep_edges; edge += num_threads) 
    {
        const int v1 = row_ids_rep[edge];
        const int v2 = col_ids_rep[edge];
        int rep_edge_index = -1;
        float rep_edge_cost = get_CSR_value(v1, v2, A_num_rows, A_row_offsets, A_col_ids, A_data, rep_edge_index);
        assert(rep_edge_cost < tol);
        assert(rep_edge_index >= 0); // The repulsive edge must also be present in A.(row -> col).

        int v1_n1_edge_index, v1_n2_edge_index, v2_n1_edge_index, v2_edge_index; 
        // Searching for a path like: v1 -(v1_n1_edge_index)- v1_n1 -(v1_n2_edge_index)- v1_n2 -(v2_n1_edge_index)- v2_n1 -(v2_edge_index)-  v2.
        for(int l1 = A_symm_row_offsets[v1]; l1 < A_symm_row_offsets[v1 + 1] && rep_edge_cost < -tol; ++l1)
        {
            const int v1_n1 = A_symm_col_ids[l1];
            const float v1_n1_edge_cost = get_CSR_value_both_dir_geq_tol(v1, v1_n1, A_num_rows, A_row_offsets, A_col_ids, A_data, v1_n1_edge_index);
            if (v1_n1_edge_cost < tol)
                continue; 

            for(int l2 = A_symm_row_offsets[v2]; l2 < A_symm_row_offsets[v2 + 1] && rep_edge_cost < -tol; ++l2)
            {
                const int v2_n1 = A_symm_col_ids[l2];
                const float v2_edge_cost = get_CSR_value_both_dir_geq_tol(v2, v2_n1, A_num_rows, A_row_offsets, A_col_ids, A_data, v2_edge_index);
                if (v2_edge_cost < tol)
                    continue;

                for(int l3 = A_symm_row_offsets[v1_n1]; l3 < A_symm_row_offsets[v1_n1 + 1] && rep_edge_cost < -tol; ++l3)
                {
                    const int v1_n2 = A_symm_col_ids[l3];
                    float v1_n2_edge_cost, v2_n1_edge_cost;
                    const bool connected = are_connected_by(v1_n1, v2_n1, v1_n2, 
                                                    A_row_offsets, A_col_ids, A_data, A_num_rows,
                                                    v1_n2_edge_index, v2_n1_edge_index,
                                                    v1_n2_edge_cost, v2_n1_edge_cost);
                    
                    if (!connected)
                        continue;

                    const float packing_value = min(-rep_edge_cost, 
                                            min(v1_n1_edge_cost, 
                                                min(v1_n2_edge_cost,
                                                    min(v2_n1_edge_cost, v2_edge_cost))));

                    rep_edge_cost += packing_value;
                    atomicAdd(&A_data[v1_n1_edge_index], -packing_value);
                    atomicAdd(&A_data[v1_n2_edge_index], -packing_value);
                    atomicAdd(&A_data[v2_n1_edge_index], -packing_value);
                    atomicAdd(&A_data[v2_edge_index], -packing_value);
                    if (A_data[v1_n1_edge_index] < 0 || A_data[v1_n2_edge_index] < 0 || A_data[v2_edge_index] < 0 || A_data[v2_n1_edge_index] < 0)
                    {// Undo:
                        rep_edge_cost -= packing_value;
                        atomicAdd(&A_data[v1_n1_edge_index], packing_value);
                        atomicAdd(&A_data[v1_n2_edge_index], packing_value);
                        atomicAdd(&A_data[v2_n1_edge_index], packing_value);
                        atomicAdd(&A_data[v2_edge_index], packing_value);    
                    }
                }
            }
        }
        A_data[rep_edge_index] = rep_edge_cost;
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

std::tuple<dCOO, thrust::device_vector<int>, thrust::device_vector<int>, int> create_matrices(const dCOO& A)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    
    // Partition edges into positive and negative.
    thrust::device_vector<int> row_ids_rep = A.get_row_ids();
    thrust::device_vector<int> col_ids_rep = A.get_col_ids();
    thrust::device_vector<float> costs = A.get_data();

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids_rep.begin(), col_ids_rep.begin(), costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids_rep.end(), col_ids_rep.end(), costs.end()));

    auto first_negative = thrust::partition(first, last, is_positive_edge());
    const size_t nr_positive_edges = std::distance(first, first_negative);

    // Create symmetric adjacency matrix of positive edges.
    thrust::device_vector<int> pos_row_ids_symm, pos_col_ids_symm;
    thrust::device_vector<float> pos_costs_symm;
    dCOO A_pos;
    if (nr_positive_edges > 0)
    {
        std::tie(pos_row_ids_symm, pos_col_ids_symm, pos_costs_symm) = to_undirected(row_ids_rep.begin(), row_ids_rep.begin() + nr_positive_edges,
                                                                                    col_ids_rep.begin(), col_ids_rep.begin() + nr_positive_edges,
                                                                                    costs.begin(), costs.begin() + nr_positive_edges);
        A_pos = dCOO(A.max_dim(), A.max_dim(),
                    pos_col_ids_symm.begin(), pos_col_ids_symm.end(),
                    pos_row_ids_symm.begin(), pos_row_ids_symm.end(), 
                    pos_costs_symm.begin(), pos_costs_symm.end(), false);
    }
    return {A_pos, row_ids_rep, col_ids_rep, nr_positive_edges};
}

// A should be directed thus containing same number of elements as in original problem. Does packing in-place on A.
std::tuple<double, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> parallel_small_cycle_packing_cuda(dCOO& A, const int max_tries_triangles, const int max_tries_quads)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    int num_nodes = A.rows();
    int num_edges = A.nnz();
    double lb = get_lb(A.get_data());
    std::cout<<"Initial lb: "<<lb<<std::endl;

    // Make adjacency matrix and BFS search starting matrix.
    dCOO A_pos;
    thrust::device_vector<int> row_ids_rep, col_ids_rep;
    int nr_positive_edges;
    std::tie(A_pos, row_ids_rep, col_ids_rep, nr_positive_edges) = create_matrices(A);
    if (nr_positive_edges == 0)
        return {lb, thrust::device_vector<int>(0), thrust::device_vector<int>(0), thrust::device_vector<int>(0)}; 

    int num_rep_edges = num_edges - nr_positive_edges;
 
    thrust::device_vector<int> A_row_offsets = A.compute_row_offsets();
    thrust::device_vector<int> A_pos_row_offsets = A_pos.compute_row_offsets();

    int threadCount = 256;
    int blockCount = ceil(num_rep_edges / (float) threadCount);
    thrust::device_vector<int> triangles_v1(num_rep_edges * 10); //TODO
    thrust::device_vector<int> triangles_v2(num_rep_edges * 10); //TODO
    thrust::device_vector<int> triangles_v3(num_rep_edges * 10); //TODO
    thrust::device_vector<int> empty_tri_index(1, 0);

    for (int t = 0; t < max_tries_triangles; t++)
    {
        pack_triangles_parallel<<<blockCount, threadCount>>>(num_rep_edges, 
            thrust::raw_pointer_cast(row_ids_rep.data()), 
            thrust::raw_pointer_cast(col_ids_rep.data()), 
            thrust::raw_pointer_cast(A_pos_row_offsets.data()),
            A_pos.get_col_ids_ptr(),
            thrust::raw_pointer_cast(A_row_offsets.data()),
            A.get_col_ids_ptr(),
            A.get_writeable_data_ptr(),
            nr_positive_edges,
            A.rows(), 
            thrust::raw_pointer_cast(triangles_v1.data()),
            thrust::raw_pointer_cast(triangles_v2.data()),
            thrust::raw_pointer_cast(triangles_v3.data()),
            thrust::raw_pointer_cast(empty_tri_index.data()),
            triangles_v1.size());
        
        lb = get_lb(A.get_data());
        std::cout<<"packing triangles, itr: "<<t<<", lb: "<<lb<<", found # of triangles: "<<empty_tri_index[0]<<", budget: "<<triangles_v1.size()<<std::endl;
    }
    for (int t = 0; t < max_tries_quads; t++)
    {
        pack_quadrangles_parallel<<<blockCount, threadCount>>>(num_rep_edges, 
            thrust::raw_pointer_cast(row_ids_rep.data()), 
            thrust::raw_pointer_cast(col_ids_rep.data()), 
            thrust::raw_pointer_cast(A_pos_row_offsets.data()),
            A_pos.get_col_ids_ptr(),
            thrust::raw_pointer_cast(A_row_offsets.data()),
            A.get_col_ids_ptr(),
            A.get_writeable_data_ptr(),
            nr_positive_edges,
            A.rows(),
            thrust::raw_pointer_cast(triangles_v1.data()),
            thrust::raw_pointer_cast(triangles_v2.data()),
            thrust::raw_pointer_cast(triangles_v3.data()),
            thrust::raw_pointer_cast(empty_tri_index.data()),
            triangles_v1.size());
        
        lb = get_lb(A.get_data());
        std::cout<<"packing quadrangles, itr: "<<t<<", lb: "<<lb<<", found # of triangles: "<<empty_tri_index[0]<<", budget: "<<triangles_v1.size()<<std::endl;
    }
    int nr_triangles = empty_tri_index[0];
    triangles_v1.resize(nr_triangles);
    triangles_v2.resize(nr_triangles);
    triangles_v3.resize(nr_triangles);
    return {lb, triangles_v1, triangles_v2, triangles_v3};
}

std::tuple<double, dCOO, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> parallel_small_cycle_packing_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const int max_tries_triangles, const int max_tries_quads)
{
    const int cuda_device = get_cuda_device();
    cudaSetDevice(cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);
    std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << ", device number " << cuda_device << "\n";
    
    dCOO A(i.begin(), i.end(),
        j.begin(), j.end(), 
        costs.begin(), costs.end(), true);
    
    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    double lb;
    std::tie(lb, triangles_v1, triangles_v2, triangles_v3) = parallel_small_cycle_packing_cuda(A, max_tries_triangles, max_tries_quads);
    return {lb, A, triangles_v1, triangles_v2, triangles_v3};
}

double parallel_small_cycle_packing_cuda_lower_bound(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const int max_tries_triangles, const int max_tries_quads)
{
    dCOO A; 
    double lb;
    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    std::tie(lb, A, triangles_v1, triangles_v2, triangles_v3) = parallel_small_cycle_packing_cuda(i, j, costs, max_tries_triangles, max_tries_quads);
    return lb;
}
