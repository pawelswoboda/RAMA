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
    // if(const char* cuda_env = std::getenv("CUDA_VISIBLE_DEVICES"))
    // {
    //     std::cout << "Cuda device number to use = " << std::stoi(cuda_env) << "\n";
    //     return std::stoi(cuda_env); 
    // }
    // else
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> adjacency_edges(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs)
{
    // TODO: make faster
    assert(i.size() == j.size() && i.size() == costs.size());
    const size_t nr_edges = i.size();
    thrust::device_vector<int> d_col_ids(2*nr_edges);
    thrust::device_vector<int> d_row_ids(2*nr_edges);
    thrust::device_vector<float> d_costs(2*nr_edges);

    thrust::copy(i.begin(), i.end(), d_col_ids.begin());
    thrust::copy(j.begin(), j.end(), d_row_ids.begin());
    thrust::copy(i.begin(), i.end(), d_row_ids.begin() + i.size());
    thrust::copy(j.begin(), j.end(), d_col_ids.begin() + j.size());
    thrust::copy(costs.begin(), costs.end(), d_costs.begin());
    thrust::copy(costs.begin(), costs.end(), d_costs.begin() + costs.size());

    return {d_col_ids, d_row_ids, d_costs};
}

template<typename ITERATOR>
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>> separate_edges(ITERATOR entry_begin, ITERATOR entry_end)
{
    // TODO: make faster
    const size_t nr_edges = std::distance(entry_begin, entry_end);
    thrust::host_vector<int> col_ids(nr_edges);
    thrust::host_vector<int> row_ids(nr_edges);
    thrust::host_vector<float> cost(nr_edges);
    for(auto it=entry_begin; it!=entry_end; ++it)
    {
        const int i = std::get<0>(*it);
        const int j = std::get<1>(*it);
        const float c = std::get<2>(*it);
        col_ids[std::distance(entry_begin, it)] = i;
        row_ids[std::distance(entry_begin, it)] = j;
        cost[std::distance(entry_begin, it)] = c;
    }
    return {col_ids, row_ids, cost};
}

std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>> to_undirected(thrust::host_vector<int> col_ids, thrust::host_vector<int> row_ids, thrust::host_vector<float> cost) 
{
    // TODO: make faster
    const size_t nr_edges = col_ids.size();
    thrust::host_vector<int> col_ids_u(2 * nr_edges);
    thrust::host_vector<int> row_ids_u(2 * nr_edges);
    thrust::host_vector<float> cost_u(2 * nr_edges);
    for(auto i = 0; i != nr_edges; ++i)
    {
        col_ids_u[2 * i] = col_ids[i];
        row_ids_u[2 * i] = row_ids[i];
        cost_u[2 * i] = cost[i];

        col_ids_u[2 * i + 1] = row_ids[i];
        row_ids_u[2 * i + 1] = col_ids[i];
        cost_u[2 * i + 1] = cost[i];
    }
    return {col_ids_u, row_ids_u, cost_u};
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
    {
    MEASURE_FUNCTION_EXECUTION_TIME;
    std::tie(row_ids, col_ids, data) = A.export_coo(handle);
    std::cout << "coo sorting time:\n";
    }

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
    std::cout << "sort by edge weight time:\n";
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
            std::cout << "A dim = " << A.cols() << "x" << A.rows() << "\n";
            std::cout << "A nnz = " << A.nnz() << ", sparsity = " << 100.0 * double(A.nnz()) / double(A.cols() * A.rows()) << "%\n";
            std::cout << "A*C multiply time:\n";
            dCSR intermed = multiply(handle, A, C);
            std::cout << "A C dim = " << intermed.rows() << "x" << intermed.cols() << "\n"; 
            std::cout << "C' transpose time:\n";
            dCSR C_trans = C.transpose(handle);
            std::cout << "C' * (AC) multiply time:\n";
            dCSR new_A = multiply(handle, C_trans, intermed);
            std::cout << "C' A C dim = " << new_A.rows() << "x" << new_A.cols() << "\n"; 
            A = new_A;
            assert(A.rows() == A.cols());
            std::cout << "execution time for matrix multiplication:\n";
        }

        A.set_diagonal_to_zero(handle);
        //A.compress(handle); 
        std::cout << "energy after iteration " << iter << ": " << A.sum()/2.0 << "\n";
    }

    const double lb = A.sum()/2.0;
    std::cout << "final energy = " << lb << "\n";

    cusparseDestroy(handle);
    std::vector<int> h_node_mapping(node_mapping.size());
    thrust::copy(node_mapping.begin(), node_mapping.end(), h_node_mapping.begin());
    return h_node_mapping;
}

void print_obj_original(const std::vector<int>& h_node_mapping, const std::vector<std::tuple<int,int,float>>& edges)
{
    float obj = 0;
    const int nr_edges = edges.size();
    for (int i = 0; i < nr_edges; i++)
    {
        const auto [e1, e2, c] = edges[i];
        if (h_node_mapping[e1] != h_node_mapping[e2])
            obj += c;
    }
    std::cout<<"Cost w.r.t original objective: "<<obj<<std::endl;
}

std::vector<int> parallel_gaec_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs)

{
    const auto adj_edges = adjacency_edges(i,j,costs);

    const int cuda_device = get_cuda_device();
    cudaSetDevice(cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);
    std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << ", device number " << cuda_device << "\n";
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");

    auto [col_ids_d, row_ids_d, costs_d] = separate_edges(edges.begin(), edges.end());

    thrust::device_vector<int> col_ids_d_device = col_ids_d;
    thrust::device_vector<int> row_ids_d_device = row_ids_d;
    thrust::device_vector<float> costs_ids_d_device = costs_d;

    const auto [row_ids_reparam, col_ids_reparam, costs_ids_reparam] = parallel_cycle_packing_cuda(row_ids_d_device, col_ids_d_device, costs_ids_d_device, 7);

    const auto [col_ids, row_ids, costs] = to_undirected(row_ids_reparam, col_ids_reparam, costs_ids_reparam);

    dCSR A(handle, 
            col_ids.begin(), col_ids.end(),
            row_ids.begin(), row_ids.end(),
            costs.begin(), costs.end());

    // dCSR A(handle, 
    //     std::get<0>(adj_edges).begin(), std::get<0>(adj_edges).end(),
    //     std::get<1>(adj_edges).begin(), std::get<1>(adj_edges).end(),
    //     std::get<2>(adj_edges).begin(), std::get<2>(adj_edges).end());
    cusparseDestroy(handle);

    const std::vector<int> h_node_mapping = parallel_gaec_cuda(A);
    print_obj_original(h_node_mapping, edges);

    return h_node_mapping;
}
