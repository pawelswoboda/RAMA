#include <cuda_runtime.h>
#include "dCSR.h"
#include "union_find.hxx"
#include "time_measure_util.h"
#include <algorithm>
#include <cstdlib>
#include "external/ECL-CC/ECLgraph.h"
#include <thrust/transform_scan.h>
#include <thrust/transform.h>
#include "maximum_matching/maximum_matching.h"

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
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

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

struct cost_scaling_func {
    const float scaling_factor;
    __host__ __device__
        inline int operator()(const float x)
        {
            return int(scaling_factor * x);
        } 
};

struct remove_negative_edges_func {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,float> e)
        {
            if(thrust::get<0>(e) == thrust::get<1>(e))
                return true;
            else if(thrust::get<2>(e) <= 0.0)
                return true;
            else
                return false;
        }
};

struct remove_reverse_edges_func {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,float> e)
        {
            return thrust::get<0>(e) > thrust::get<1>(e);
        }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> filter_edges_by_matching(cusparseHandle_t handle, thrust::device_vector<int> i, thrust::device_vector<int> j, thrust::device_vector<float> w)
{
    assert(i.size() == j.size());
    assert(i.size() == w.size());
    assert(*thrust::max_element(i.begin(), i.end()) == *thrust::max_element(i.begin(), i.end())); 

    // remove negative edges
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin(), w.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end(), w.end()));
        auto new_last = thrust::remove_if(first, last, remove_negative_edges_func());
        i.resize(std::distance(first, new_last));
        j.resize(std::distance(first, new_last));
        w.resize(std::distance(first, new_last)); 
    }

    // remove reverse edges
    {
    //    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin(), w.begin()));
    //    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end(), w.end()));

    //    auto new_last = thrust::remove_if(first, last, remove_reverse_edges_func());
    //    i.resize(std::distance(first, new_last));
    //    j.resize(std::distance(first, new_last));
    //    w.resize(std::distance(first, new_last)); 
    }

    coo_sorting(handle,j,i,w);

    /*
    std::cout << "# edges for maximum matching: " << i.size() << "\n";
    for(size_t c=0; c<i.size(); ++c)
        std::cout << i[c] << " -> " << j[c] << "; " << w[c] << "\n";
        */

    // TODO: only needed for non-reverse copies of edges
    const int num_nodes = std::max(*thrust::max_element(i.begin(), i.end()) + 1, *thrust::max_element(j.begin(), j.end()) + 1);
    const int num_edges = i.size();

    thrust::device_vector<int> w_scaled(w.size());

    // TODO: put larger factor below
    const float scaling_factor = 1024.0 / *thrust::max_element(w.begin(), w.end());

    thrust::transform(w.begin(), w.end(), w_scaled.begin(), cost_scaling_func({scaling_factor}));

    thrust::device_vector<int> i_matched, j_matched;
    std::tie(i_matched, j_matched) = maximum_matching(
            num_nodes, num_edges,
            thrust::raw_pointer_cast(i.data()),
            thrust::raw_pointer_cast(j.data()),
            thrust::raw_pointer_cast(w_scaled.data()));

    /*
    std::cout << "edges to match:\n";
    for(size_t c=0; c<i_matched.size(); ++c)
        std::cout << i_matched[c] << " -> " << j_matched[c] << "\n";
    std::cout << "\n";
    */

    return {i_matched, j_matched}; 
}

std::tuple<dCSR,thrust::device_vector<int>> edge_contraction_matrix_cuda(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids, const int n)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

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
    //thrust::device_vector<int> c_col_ids = node_mapping;

    //assert(nr_ccs > *thrust::max_element(c_col_ids.begin(), c_col_ids.end()));
    assert(n > *thrust::max_element(c_row_ids.begin(), c_row_ids.end()));

    thrust::device_vector<int> ones(c_row_ids.size(), 1);
    dCSR C(handle, n, nr_ccs, node_mapping.begin(), node_mapping.end(), c_row_ids.begin(), c_row_ids.end(), ones.begin(), ones.end());

    return {C, node_mapping};
}

thrust::device_vector<float> per_cc_cost(cusparseHandle_t handle, const dCSR& A, const dCSR& C, const thrust::device_vector<int>& node_mapping, const int nr_ccs)
{
    thrust::device_vector<float> d = A.diagonal(handle);
    return d;

    // for all the components with positive contraction costs, remove half of the contraction edges with lower costs
}

struct is_negative
{
    __host__ __device__
        bool operator()(const float x)
        {
            return x < 0.0;
        }
};
bool has_bad_contractions(cusparseHandle_t handle, const dCSR& A)
{
    const thrust::device_vector<float> d = A.diagonal(handle);
    return thrust::count_if(d.begin(), d.end(), is_negative()) > 0;
}

struct remove_bad_contraction_edges_func
{
    const int* node_mapping;
    const float* cc_cost;
    __host__ __device__
        bool operator()(thrust::tuple<int,int> t)
        {
            const int i = thrust::get<0>(t);
            const int j = thrust::get<1>(t);
            assert(node_mapping[i] == node_mapping[j]);
            const int cc = node_mapping[i];
            return cc_cost[cc] <= 0.0;
        }
    typedef thrust::tuple<int,int> argument_type;
};
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> good_contract_edges(cusparseHandle_t handle, const dCSR& contracted_A, const thrust::device_vector<int>& node_mapping, const thrust::device_vector<int>& contract_cols, const thrust::device_vector<int>& contract_rows)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    // for each component, how profitable was it to contract it?
    const thrust::device_vector<float> d = contracted_A.diagonal(handle);

    remove_bad_contraction_edges_func func({
            thrust::raw_pointer_cast(node_mapping.data()),
            thrust::raw_pointer_cast(d.data())
            }); 

    thrust::device_vector<int> good_contract_cols = contract_cols;
    thrust::device_vector<int> good_contract_rows = contract_rows;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(good_contract_cols.begin(), good_contract_rows.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(good_contract_cols.end(), good_contract_rows.end()));
    auto new_last = thrust::remove_if(first, last, func);
    const int nr_good_edges = thrust::distance(first, new_last);
    good_contract_cols.resize(nr_good_edges);
    good_contract_rows.resize(nr_good_edges);

    return {good_contract_cols, good_contract_rows};
}


struct negative_edge_indicator_func
{
    const float w = 0.0;
    __host__ __device__
        bool operator()(const thrust::tuple<int,int,float> t)
        {
            if(thrust::get<0>(t) <= thrust::get<1>(t)) // we only want one representative
                return true;
            if(thrust::get<2>(t) <= w)
                return true;
            return false;
        }
};

struct edge_comparator_func {
    __host__ __device__
        inline bool operator()(const thrust::tuple<int, int, float>& a, const thrust::tuple<int, int, float>& b)
        {
            return thrust::get<2>(a) > thrust::get<2>(b);
        } 
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> edges_to_contract_cuda(cusparseHandle_t handle, dCSR& A, const float retain_ratio)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    assert(retain_ratio < 1.0 && 0.0 < retain_ratio);
    thrust::device_vector<int> row_ids;
    thrust::device_vector<int> col_ids;
    thrust::device_vector<float> data;

    std::tie(row_ids, col_ids, data) = A.export_coo(handle);

    std::tie(col_ids, row_ids) = filter_edges_by_matching(handle, col_ids, row_ids, data);

    /*
    auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    auto new_last = thrust::remove_if(first, last, negative_edge_indicator_func({0.0}));
    const size_t nr_positive_edges = std::distance(first, new_last);
    col_ids.resize(nr_positive_edges);
    row_ids.resize(nr_positive_edges);
    data.resize(nr_positive_edges);


    const double smallest_edge_weight = *thrust::min_element(data.begin(), data.end());
    const double largest_edge_weight = *thrust::max_element(data.begin(), data.end());
    std::cout << "contraction edges min/max: " << smallest_edge_weight << ", " << largest_edge_weight << "\n";
    const float mid_edge_weight = retain_ratio * largest_edge_weight;

    new_last = thrust::remove_if(first, last, negative_edge_indicator_func({mid_edge_weight}));
    const size_t nr_remaining_edges = std::distance(first, new_last);
    col_ids.resize(nr_remaining_edges);
    row_ids.resize(nr_remaining_edges);
    */

    /*
    if(max_contractions < nr_positive_edges)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));
        thrust::sort(first, last, edge_comparator_func()); // TODO: faster through sort by keys?

        col_ids.resize(max_contractions);
        row_ids.resize(max_contractions);
        data.resize(max_contractions);
    }
    */

    //const double smallest_weight = *thrust::max_element(data.begin(), data.end());
    //std::cout << "smallest edge weight: " << smallest_edge_weight

    // add reverse edges
    const int old_size = col_ids.size();
    const int new_size = 2*col_ids.size();
    col_ids.resize(new_size);
    row_ids.resize(new_size);

    thrust::copy(col_ids.begin(), col_ids.begin() + old_size, row_ids.begin() + old_size);
    thrust::copy(row_ids.begin(), row_ids.begin() + old_size, col_ids.begin() + old_size);

    return {col_ids, row_ids};
}

dCSR contract(cusparseHandle_t handle, dCSR& A, dCSR& C)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    assert(A.cols() == A.rows());
    dCSR intermed = multiply(handle, A, C);
    dCSR C_trans = C.transpose(handle);
    dCSR new_A = multiply(handle, C_trans, intermed);
    assert(new_A.rows() == new_A.cols());
    return new_A;
}

std::vector<int> parallel_gaec_cuda(dCSR& A)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");

    const double initial_lb = A.sum()/2.0;
    std::cout << "initial energy = " << initial_lb << "\n";

    thrust::device_vector<int> node_mapping(A.rows());
    thrust::sequence(node_mapping.begin(), node_mapping.end());
    double contract_ratio = 0.5;
    assert(A.rows() == A.cols());

    for(size_t iter=0;; ++iter)
    {
        //const size_t nr_edges_to_contract = std::max(size_t(1), size_t(A.rows() * contract_ratio));

        thrust::device_vector<int> contract_cols, contract_rows;
        std::tie(contract_cols, contract_rows) = edges_to_contract_cuda(handle, A, contract_ratio);
        //std::cout << "iter " << iter << ", edge contraction ratio = " << contract_ratio << ", # edges to contract request " << nr_edges_to_contract << ", # nr edges to contract provided = " << contract_cols.size() << "\n";

        if(contract_cols.size() == 0)
        {
            std::cout << "# iterations = " << iter << "\n";
            break;
        }

        dCSR C;
        thrust::device_vector<int> cur_node_mapping;
        std::tie(C, cur_node_mapping) = edge_contraction_matrix_cuda(handle, contract_cols, contract_rows, A.rows());

        dCSR new_A = contract(handle, A, C);
        std::cout << "contract C size " << C.cols() << "x" << C.rows() << "\n";
        std::cout << "original A size " << A.cols() << "x" << A.rows() << "\n";
        std::cout << "contracted A size " << new_A.cols() << "x" << new_A.rows() << "\n";
        assert(new_A.cols() < A.cols());

        const thrust::device_vector<float> diagonal = new_A.diagonal(handle);
        const float energy_reduction = thrust::reduce(diagonal.begin(), diagonal.end());
        std::cout << "energy reduction " << energy_reduction << "\n";
        //if(energy_reduction < 0.0)
        if(has_bad_contractions(handle, new_A))
        {
            contract_ratio *= 2.0; 
            //contract_ratio = std::max(contract_ratio, 0.005);
            // get contraction edges of the components which
            thrust::device_vector<int> good_contract_cols, good_contract_rows;
            std::tie(good_contract_cols, good_contract_rows) = good_contract_edges(handle, new_A, cur_node_mapping, contract_cols, contract_rows);
            const double perc_used_edges = double(good_contract_cols.size()) / double(contract_cols.size());
            std::cout << "% used contraction edges = " << perc_used_edges*100 << "\n";
            std::tie(C, cur_node_mapping) = edge_contraction_matrix_cuda(handle, good_contract_cols, good_contract_rows, A.rows());
            new_A = contract(handle, A, C);
            assert(!has_bad_contractions(handle, new_A));
        }
        else
        {
            contract_ratio *= 0.5;//1.3;
            contract_ratio = std::min(contract_ratio, 0.35);
        }

        thrust::swap(A,new_A);
        A.set_diagonal_to_zero(handle);
        std::cout << "energy after iteration " << iter << ": " << A.sum()/2.0 << ", #components = " << A.cols() << "\n";
        thrust::gather(node_mapping.begin(), node_mapping.end(), cur_node_mapping.begin(), node_mapping.begin());

        //A.compress(handle); 
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

struct combine_costs
{
    const float a;
    combine_costs(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& orig, const float& reparam) const { 
            return a * orig + (1.0f - a) * reparam;
        }
};

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
    thrust::device_vector<float> costs_d_reparam;

    // std::tie(i_d_reparam, j_d_reparam, costs_d_reparam) = parallel_cycle_packing_cuda(i_d, j_d, costs_d, 5, 1000);
    std::tie(i_d_reparam, j_d_reparam, costs_d_reparam) = parallel_small_cycle_packing_cuda(handle, i_d, j_d, costs_d, 1);

    // To combine costs:
    // thrust::transform(costs_d.begin(), costs_d.end(), costs_d_reparam.begin(), costs_d_reparam.begin(), combine_costs(0.5));

    thrust::device_vector<int> col_ids_u;
    thrust::device_vector<int> row_ids_u;
    thrust::device_vector<float> costs_u;
    std::tie(col_ids_u, row_ids_u, costs_u) = to_undirected(i_d_reparam, j_d_reparam, costs_d_reparam);
    // std::tie(col_ids_u, row_ids_u, costs_u) = to_undirected(i_d, j_d, costs_d);

    dCSR A(handle, 
            col_ids_u.begin(), col_ids_u.end(),
            row_ids_u.begin(), row_ids_u.end(),
            costs_u.begin(), costs_u.end());

    cusparseDestroy(handle);

    const std::vector<int> h_node_mapping = parallel_gaec_cuda(A);
    print_obj_original(h_node_mapping, i, j, costs); 
    
    return h_node_mapping;
}
