#include <cuda_runtime.h>
#include "dCSR.h"
#include "union_find.hxx"
#include "time_measure_util.h"
#include <algorithm>
#include <cstdlib>
#include "external/ECL-CC/ECLgraph.h"
#include <thrust/transform_scan.h>
#include <thrust/transform.h>
#include "icp.h"

int get_cuda_device()
{   
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_undirected(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j, const thrust::device_vector<float>& costs)
{
    assert(i.size() == j.size() && i.size() == costs.size());
    const size_t nr_edges = i.size();
    thrust::device_vector<int> col_ids_u(2*nr_edges);
    thrust::device_vector<int> row_ids_u(2*nr_edges);
    thrust::device_vector<float> costs_u(2*nr_edges);

    thrust::copy(i.begin(), i.end(), col_ids_u.begin());
    thrust::copy(j.begin(), j.end(), row_ids_u.begin());
    thrust::copy(i.begin(), i.end(), row_ids_u.begin() + i.size());
    thrust::copy(j.begin(), j.end(), col_ids_u.begin() + j.size());
    thrust::copy(costs.begin(), costs.end(), costs_u.begin());
    thrust::copy(costs.begin(), costs.end(), costs_u.begin() + costs.size());

    return {col_ids_u, row_ids_u, costs_u};
}

thrust::device_vector<int> compress_label_sequence(const thrust::device_vector<int>& data)
{
    assert(*thrust::max_element(data.begin(), data.end()) < data.size());

    // first get mask of used labels
    thrust::device_vector<int> label_mask(data.size(), 0);
    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + data.size(), data.begin(), label_mask.begin());

    // get map of original labels to consecutive ones
    thrust::device_vector<int> label_to_consecutive(data.size());
    thrust::exclusive_scan(label_mask.begin(), label_mask.end(), label_to_consecutive.begin());

    // apply compressed label map
    thrust::device_vector<int> result(data.size(), 0);
    thrust::gather(data.begin(), data.end(), label_to_consecutive.begin(), result.begin());

    return result;
}

std::tuple<dCSR,thrust::device_vector<int>> edge_contraction_matrix_cuda(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids, const int n)
{
    MEASURE_FUNCTION_EXECUTION_TIME;
    assert(col_ids.size() == row_ids.size());
    assert(n > *thrust::max_element(col_ids.begin(), col_ids.end()));
    assert(n > *thrust::max_element(row_ids.begin(), row_ids.end()));

    coo_sorting(handle, col_ids, row_ids);
    thrust::device_vector<int> row_offsets = dCSR::compute_row_offsets(handle, n, col_ids, row_ids);

    thrust::device_vector<int> cc_labels(n);
    computeCC_gpu(n, col_ids.size(), 
            thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(col_ids.data()), 
            thrust::raw_pointer_cast(cc_labels.data()), get_cuda_device());

    thrust::device_vector<int> node_mapping = compress_label_sequence(cc_labels);
    const int nr_ccs = *thrust::max_element(node_mapping.begin(), node_mapping.end()) + 1;

    assert(nr_ccs < n);

    // construct contraction matrix
    thrust::device_vector<int> c_row_ids(n);
    thrust::sequence(c_row_ids.begin(), c_row_ids.end());
    thrust::device_vector<int> c_col_ids = node_mapping;

    assert(nr_ccs > *thrust::max_element(c_col_ids.begin(), c_col_ids.end()));
    assert(n > *thrust::max_element(c_row_ids.begin(), c_row_ids.end()));

    thrust::device_vector<int> ones(c_row_ids.size(), 1);
    dCSR C(handle, n, nr_ccs, c_col_ids.begin(), c_col_ids.end(), c_row_ids.begin(), c_row_ids.end(), ones.begin(), ones.end());

    return {C, node_mapping};
}

struct positive_edge_indicator_func
{
    __host__ __device__
        bool operator()(const thrust::tuple<int,int,float> t)
        {
            if(thrust::get<0>(t) > thrust::get<1>(t) && thrust::get<2>(t) > 0.0)
                return false;
            else
                return true;
        }
};

struct edge_comparator_func {
    __host__ __device__
        inline bool operator()(const thrust::tuple<int, int, float>& a, const thrust::tuple<int, int, float>& b)
        {
            return thrust::get<2>(a) > thrust::get<2>(b);
        } 
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> edges_to_contract_cuda(cusparseHandle_t handle, dCSR& A, const size_t max_contractions)
{
    MEASURE_FUNCTION_EXECUTION_TIME;
    assert(max_contractions > 0);
    thrust::device_vector<int> row_ids;
    thrust::device_vector<int> col_ids;
    thrust::device_vector<float> data;

    std::tie(row_ids, col_ids, data) = A.export_coo(handle);

    auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    auto new_last = thrust::remove_if(first, last, positive_edge_indicator_func());
    const size_t nr_positive_edges = std::distance(first, new_last);
    col_ids.resize(nr_positive_edges);
    row_ids.resize(nr_positive_edges);
    data.resize(nr_positive_edges);

    if(max_contractions < nr_positive_edges)
    {
    MEASURE_FUNCTION_EXECUTION_TIME;
        auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));
        thrust::sort(first, last, edge_comparator_func()); // TODO: faster through sort by keys?

        col_ids.resize(max_contractions);
        row_ids.resize(max_contractions);
    }

    // add reverse edges
    const int old_size = col_ids.size();
    const int new_size = 2*col_ids.size();
    col_ids.resize(new_size);
    row_ids.resize(new_size);

    thrust::copy(col_ids.begin(), col_ids.begin() + old_size, row_ids.begin() + old_size);
    thrust::copy(row_ids.begin(), row_ids.begin() + old_size, col_ids.begin() + old_size);

    return {col_ids, row_ids};
}

std::vector<int> parallel_gaec_cuda(dCSR& A)
{
    MEASURE_FUNCTION_EXECUTION_TIME;
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");

    const double initial_lb = A.sum()/2.0;
    std::cout << "initial energy = " << initial_lb << "\n";

    thrust::device_vector<int> node_mapping(A.rows());
    thrust::sequence(node_mapping.begin(), node_mapping.end());
    constexpr static double contract_ratio = 0.1;
    assert(A.rows() == A.cols());

    for(size_t iter=0;; ++iter)
    {
        const size_t nr_edges_to_contract = std::max(size_t(1), size_t(A.rows() * contract_ratio));
        
        thrust::device_vector<int> contract_cols;
        thrust::device_vector<int> contract_rows;
        std::tie(contract_cols, contract_rows)  = edges_to_contract_cuda(handle, A, nr_edges_to_contract);

        if(contract_cols.size() == 0)
        {
            std::cout << "# iterations = " << iter << "\n";
            break;
        }
        dCSR C;
        thrust::device_vector<int> cur_node_mapping;
        std::tie(C, cur_node_mapping) = edge_contraction_matrix_cuda(handle, contract_cols, contract_rows, A.rows());

        thrust::gather(node_mapping.begin(), node_mapping.end(), cur_node_mapping.begin(), node_mapping.begin());

        {
            MEASURE_FUNCTION_EXECUTION_TIME;

            assert(A.cols() == A.rows());
            dCSR intermed = multiply(handle, A, C);
            dCSR C_trans = C.transpose(handle);
            dCSR new_A = multiply(handle, C_trans, intermed);
            A = new_A;
            assert(A.rows() == A.cols());
        }

        A.set_diagonal_to_zero(handle);
        //A.compress(handle); 
        std::cout << "energy after iteration " << iter << ": " << A.sum()/2.0 << ", #components = " << A.cols() << "\n";
    }

    const double lb = A.sum()/2.0;
    std::cout << "final energy = " << lb << "\n";

    cusparseDestroy(handle);
    std::vector<int> h_node_mapping(node_mapping.size());
    thrust::copy(node_mapping.begin(), node_mapping.end(), h_node_mapping.begin());
    return h_node_mapping;
}

void print_obj_original(const std::vector<int>& h_node_mapping, const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs)
{
    double obj = 0;
    const int nr_edges = costs.size();
    for (int e = 0; e < nr_edges; e++)
    {
        const int e1 = i[e];
        const int e2 = j[e];
        const float c = costs[e];
        if (h_node_mapping[e1] != h_node_mapping[e2])
            obj += c;
    }
    std::cout<<"Cost w.r.t original objective: "<<obj<<std::endl;
}

std::vector<int> parallel_gaec_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs)
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

    thrust::device_vector<int> i_d_reparam;
    thrust::device_vector<int> j_d_reparam;
    thrust::device_vector<int> costs_d_reparam;
    std::tie(i_d_reparam, j_d_reparam, costs_d_reparam) = parallel_cycle_packing_cuda(i_d, j_d, costs_d, 7);
    //TODO: 
    // 1. How to use the costs?
    // 2. Should zero edges be removed?

    thrust::device_vector<int> col_ids_u;
    thrust::device_vector<int> row_ids_u;
    thrust::device_vector<float> costs_u;
    std::tie(col_ids_u, row_ids_u, costs_u) = to_undirected(i_d, j_d, costs_d);

    dCSR A(handle, 
            col_ids_u.begin(), col_ids_u.end(),
            row_ids_u.begin(), row_ids_u.end(),
            costs_u.begin(), costs_u.end());

    cusparseDestroy(handle);

    const std::vector<int> h_node_mapping = parallel_gaec_cuda(A);
    print_obj_original(h_node_mapping, i, j, costs); 
    
    return h_node_mapping;
}
