#include <cuda_runtime.h>
#include "union_find.hxx"
#include "time_measure_util.h"
#include <algorithm>
#include <cstdlib>
#include "ECLgraph.h"
#include <thrust/transform_scan.h>
#include <thrust/transform.h>
#include "maximum_matching_vertex_based.h"
#include "multicut_solver_options.h"
#include "dual_solver.h"
#include "parallel_gaec_utils.h"

thrust::device_vector<int> compress_label_sequence(const thrust::device_vector<int>& data, const int max_label)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    assert(*thrust::max_element(data.begin(), data.end()) <= max_label);

    // first get mask of used labels
    thrust::device_vector<int> label_mask(max_label + 1, 0);
    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + data.size(), data.begin(), label_mask.begin());

    // get map of original labels to consecutive ones
    thrust::device_vector<int> label_to_consecutive(max_label + 1);
    thrust::exclusive_scan(label_mask.begin(), label_mask.end(), label_to_consecutive.begin());

    // apply compressed label map
    thrust::device_vector<int> result(data.size(), 0);
    thrust::gather(data.begin(), data.end(), label_to_consecutive.begin(), result.begin());

    return result;
}


thrust::device_vector<float> per_cc_cost(const dCOO& A, const dCOO& C, const thrust::device_vector<int>& node_mapping, const int nr_ccs)
{
    thrust::device_vector<float> d = A.diagonal();
    return d;
}

struct is_negative
{
    __host__ __device__
        bool operator()(const float x)
        {
            return x < 0.0;
        }
};
bool has_bad_contractions(const dCOO& A)
{
    const thrust::device_vector<float> d = A.diagonal();
    return thrust::count_if(d.begin(), d.end(), is_negative()) > 0;
}

struct remove_bad_contraction_edges_func
{
    const int nr_ccs;
    const float* cc_cost;
    __host__ __device__
        int operator()(thrust::tuple<int,int> t)
        {
            const int cc = thrust::get<0>(t);
            const int node_id = thrust::get<1>(t);
            if (cc_cost[cc] > 0.0)
                return cc;
            return node_id + nr_ccs;
        }
};

thrust::device_vector<int> discard_bad_contractions(const dCOO& contracted_A, const thrust::device_vector<int>& node_mapping)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    int nr_ccs = *thrust::max_element(node_mapping.begin(), node_mapping.end()) + 1;
    // for each component, how profitable was it to contract it?
    const thrust::device_vector<float> d = contracted_A.diagonal();

    remove_bad_contraction_edges_func func({nr_ccs, thrust::raw_pointer_cast(d.data())}); 

    thrust::device_vector<int> good_node_mapping = node_mapping;
    thrust::device_vector<int> input_node_ids(node_mapping.size());
    thrust::sequence(input_node_ids.begin(), input_node_ids.end(), 0);

    auto first = thrust::make_zip_iterator(thrust::make_tuple(good_node_mapping.begin(), input_node_ids.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(good_node_mapping.end(), input_node_ids.end()));
    thrust::transform(first, last, good_node_mapping.begin(), func);
    
    good_node_mapping = compress_label_sequence(good_node_mapping, nr_ccs + node_mapping.size());
    assert(*thrust::max_element(good_node_mapping.begin(), good_node_mapping.end()) > nr_ccs);
    return good_node_mapping;
}


struct negative_edge_indicator_func
{
    const float w = 0.0;
    __host__ __device__
        bool operator()(const thrust::tuple<int,int,float> t)
        {
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

std::tuple<thrust::device_vector<int>, int> contraction_mapping_by_sorting(dCOO& A, const float retain_ratio)
{
    assert(A.is_directed());
    thrust::device_vector<int> row_ids = A.get_row_ids();
    thrust::device_vector<int> col_ids = A.get_col_ids();
    thrust::device_vector<float> data = A.get_data();

    auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    const double smallest_edge_weight = *thrust::min_element(data.begin(), data.end());
    const double largest_edge_weight = *thrust::max_element(data.begin(), data.end());
    const float mid_edge_weight = retain_ratio * largest_edge_weight;

    auto new_last = thrust::remove_if(first, last, negative_edge_indicator_func({mid_edge_weight}));
    const size_t nr_remaining_edges = std::distance(first, new_last);
    col_ids.resize(nr_remaining_edges);
    row_ids.resize(nr_remaining_edges);
    if (nr_remaining_edges == 0)
        return {thrust::device_vector<int>(0), 0};

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

    // add reverse edges
    std::tie(row_ids, col_ids) = to_undirected(row_ids.begin(), row_ids.end(), col_ids.begin(), col_ids.end());

    assert(col_ids.size() == row_ids.size());
    coo_sorting(row_ids, col_ids);
    thrust::device_vector<int> row_offsets = compute_offsets(row_ids, A.max_dim() - 1);

    thrust::device_vector<int> cc_labels(A.max_dim());
    computeCC_gpu(A.max_dim(), col_ids.size(), 
            thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(col_ids.data()), 
            thrust::raw_pointer_cast(cc_labels.data()), get_cuda_device());

    thrust::device_vector<int> node_mapping = compress_label_sequence(cc_labels, cc_labels.size() - 1);
    const int nr_ccs = *thrust::max_element(node_mapping.begin(), node_mapping.end()) + 1;

    assert(nr_ccs < A.max_dim());

    return {node_mapping, row_ids.size()};

}

std::tuple<thrust::device_vector<int>, int> contraction_mapping_by_maximum_matching(dCOO& A)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    MEASURE_FUNCTION_EXECUTION_TIME;
    thrust::device_vector<int> node_mapping;
    int nr_matched_edges;
    std::tie(node_mapping, nr_matched_edges) = filter_edges_by_matching_vertex_based(A.export_undirected());
    return {compress_label_sequence(node_mapping, node_mapping.size() - 1), nr_matched_edges};
}

std::tuple<std::vector<int>, double> parallel_gaec_cuda(dCOO& A, const multicut_solver_options& opts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    assert(A.is_directed());

    const double final_lb = dual_solver(A, opts.max_cycle_length_lb, opts.num_dual_itr_lb, opts.tri_memory_factor);

    const double initial_lb = A.sum();
    std::cout << "initial energy = " << initial_lb << "\n";

    thrust::device_vector<int> node_mapping(A.max_dim());
    thrust::sequence(node_mapping.begin(), node_mapping.end());
    double contract_ratio = 0.5;

    bool try_edges_to_contract_by_maximum_matching = true;
    if (opts.matching_thresh_crossover_ratio > 1.0)
        try_edges_to_contract_by_maximum_matching = false;
    
    for(size_t iter=0;; ++iter)
    {
        if (iter > 0)
        {
            dual_solver(A, opts.max_cycle_length_gaec, opts.num_dual_itr_gaec, 1.0);
        }
        thrust::device_vector<int> cur_node_mapping;
        int nr_edges_to_contract;
        if(try_edges_to_contract_by_maximum_matching)
        {
            std::tie(cur_node_mapping, nr_edges_to_contract) = contraction_mapping_by_maximum_matching(A);
            if(nr_edges_to_contract < A.rows() * opts.matching_thresh_crossover_ratio)
            {
                std::cout << "# edges to contract = " << nr_edges_to_contract << ", # vertices = " << A.rows() << "\n";
                std::cout << "switching to sorting based contraction edge selection\n";
                try_edges_to_contract_by_maximum_matching = false;    
            }
        }
        if(!try_edges_to_contract_by_maximum_matching)
            std::tie(cur_node_mapping, nr_edges_to_contract) = contraction_mapping_by_sorting(A, contract_ratio);

        if(nr_edges_to_contract == 0)
        {
            std::cout << "# iterations = " << iter << "\n";
            break;
        }

        dCOO new_A = A.contract_cuda(cur_node_mapping);
        std::cout << "original A size " << A.cols() << "x" << A.rows() << "\n";
        std::cout << "contracted A size " << new_A.cols() << "x" << new_A.rows() << "\n";
        assert(new_A.cols() < A.cols());

        const thrust::device_vector<float> diagonal = new_A.diagonal();
        const float energy_reduction = thrust::reduce(diagonal.begin(), diagonal.end());
        std::cout << "energy reduction " << energy_reduction << "\n";
        if(has_bad_contractions(new_A))
        {
            if(!try_edges_to_contract_by_maximum_matching)
                contract_ratio *= 2.0; 
            int nr_ccs = *thrust::max_element(cur_node_mapping.begin(), cur_node_mapping.end()) + 1;
            cur_node_mapping = discard_bad_contractions(new_A, cur_node_mapping);
            int good_nr_ccs = *thrust::max_element(cur_node_mapping.begin(), cur_node_mapping.end()) + 1;
            assert(good_nr_ccs > nr_ccs);
            std::cout << "Reverted from " << nr_ccs << " connected components to " << good_nr_ccs << "\n";
            if (good_nr_ccs == cur_node_mapping.size()) 
                break;
            
            new_A = A.contract_cuda(cur_node_mapping);
            assert(!has_bad_contractions(new_A));
        }
        else
        {
            if(!try_edges_to_contract_by_maximum_matching)
            {
                contract_ratio *= 0.5;//1.3;
                contract_ratio = std::min(contract_ratio, 0.35);
            }
        }

        thrust::swap(A,new_A);
        A.remove_diagonal();
        std::cout << "energy after iteration " << iter << ": " << A.sum() << ", #components = " << A.cols() << "\n";
        thrust::gather(node_mapping.begin(), node_mapping.end(), cur_node_mapping.begin(), node_mapping.begin());
    }

    const double lb = A.sum();
    std::cout << "final energy = " << lb << "\n";

    std::vector<int> h_node_mapping(node_mapping.size());
    thrust::copy(node_mapping.begin(), node_mapping.end(), h_node_mapping.begin());
    return {h_node_mapping, final_lb};
}

std::tuple<std::vector<int>, double> parallel_gaec_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const multicut_solver_options& opts)
{
    const int cuda_device = get_cuda_device();
    cudaSetDevice(cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);
    std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << ", device number " << cuda_device << "\n";

    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
    
    std::vector<int> h_node_mapping;
    double lb;
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::tie(h_node_mapping, lb) = parallel_gaec_cuda(A, opts);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    if (opts.out_info_file != "")
    {
        std::ofstream outfile;
        outfile.open(opts.out_info_file, std::ios_base::app);
        outfile << time <<",";
        outfile.close();
    }
    return {h_node_mapping, lb};
}
