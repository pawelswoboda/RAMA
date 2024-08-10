#include <cuda_runtime.h>
#include "union_find.hxx"
#include "time_measure_util.h"
#include <algorithm>
#include <cstdlib>
#include <persistency_preprocessor.h>

#include "ECLgraph.h"
#include <thrust/transform_scan.h>
#include <thrust/transform.h>
#include "maximum_matching_vertex_based.h"
#include "maximum_matching_thrust.h"
#include "multicut_solver_options.h"
#include "dual_solver.h"
#include "edge_contractions_woc.h"
#include "rama_utils.h"

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

int get_obj(const dCOO& A, thrust::device_vector<int>& mapping) {
    int M = A.get_col_ids().size();
    thrust::device_vector<float> costs = thrust::device_vector<float>(M,0.0);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(A.get_row_ids().begin(), A.get_col_ids().begin(), A.get_data().begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(A.get_row_ids().end(), A.get_col_ids().end(), A.get_data().end()));
    const auto ptr = thrust::raw_pointer_cast(mapping.data());
    auto kernel = [=] __host__ __device__ (thrust::tuple<int, int, float> t){
        return
            (t.get<2>()) * (ptr[t.get<0>()] != ptr[t.get<1>()]) ;
    };
    thrust::transform(thrust::device,begin, end, costs.begin(), kernel);
    int i = thrust::reduce(costs.begin(), costs.end(), 0);
    return i;
}

struct map_nodes_to_new_clusters_func
{
    const int* node_mapping_cont_graph;
    int* node_mapping_orig_graph;
    const unsigned long num_nodes_cont;
    __host__ __device__ void operator()(const int n)
    {
        const int n_map = node_mapping_orig_graph[n];
        if (n_map < num_nodes_cont)
            node_mapping_orig_graph[n] = node_mapping_cont_graph[n_map];
    }
};

__global__ void contracting_edges_from_mapping_kernel(
        const int* row_ids,
        const int* col_ids,
        const int* node_mapping,
        float* contract_edges,
        const unsigned long num_nodes_count)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_nodes_count){
        auto u = row_ids[n];
        auto v = col_ids[n];
        if (node_mapping[u] == node_mapping[v])
            contract_edges[n] = 1.0;
    }
}


void map_node_labels(const thrust::device_vector<int>& cur_node_mapping, thrust::device_vector<int>& orig_node_mapping)
{
    map_nodes_to_new_clusters_func node_mapper({thrust::raw_pointer_cast(cur_node_mapping.data()), 
                                                thrust::raw_pointer_cast(orig_node_mapping.data()),
                                                cur_node_mapping.size()});

    thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + orig_node_mapping.size(), node_mapper);
}

std::tuple<thrust::device_vector<int>, int> contraction_mapping_by_maximum_matching(dCOO& A, const float mean_multiplier_mm, const bool verbose = true)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    thrust::device_vector<int> node_mapping;
    int nr_matched_edges;
    std::tie(node_mapping, nr_matched_edges) = filter_edges_by_matching_vertex_based(A.export_undirected(), mean_multiplier_mm, verbose);
    // std::tie(node_mapping, nr_matched_edges) = filter_edges_by_matching_thrust(A, mean_multiplier_mm, verbose);
    return {compress_label_sequence(node_mapping, node_mapping.size() - 1), nr_matched_edges};
}

std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>> > rama_cuda(dCOO& A, const multicut_solver_options& opts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    thrust::device_vector<int> node_mapping(A.max_dim());
    thrust::sequence(node_mapping.begin(), node_mapping.end());

    if (opts.run_preprocessor) {
        auto[A_new, current_node_mapping] = preprocessor_cuda(A, opts,-1);
        map_node_labels(current_node_mapping,node_mapping);
        thrust::swap(A, A_new);
        if (opts.verbose)
            std::cout << "Energy after Preprocessor= " << A.sum() << "\n";
    }

    assert(A.is_directed());
    const double final_lb = dual_solver(A, opts.max_cycle_length_lb, opts.num_dual_itr_lb, opts.tri_memory_factor, opts.num_outer_itr_dual, 1e-4, opts.verbose);


    if (opts.verbose)
        std::cout << "initial energy = " << A.sum() << "\n";


    std::vector<std::vector<int>> timeline;

    if (opts.only_compute_lb)
        return {std::vector<int>(), final_lb, timeline};
        
    bool try_edges_to_contract_by_maximum_matching = true;
    if (opts.matching_thresh_crossover_ratio > 1.0)
        try_edges_to_contract_by_maximum_matching = false;
    dCOO C;
    thrust::device_vector<float> contracting_edges;


    for(size_t iter=0; A.nnz() > 0; ++iter)
    {
        if (iter > 0)
        {
            dual_solver(A, opts.max_cycle_length_primal, opts.num_dual_itr_primal, 1.0, 1, 1e-4, opts.verbose);
        }
        thrust::device_vector<int> cur_node_mapping;
        int nr_edges_to_contract;
        if(try_edges_to_contract_by_maximum_matching)
        {
            std::tie(cur_node_mapping, nr_edges_to_contract) = contraction_mapping_by_maximum_matching(A, opts.mean_multiplier_mm, opts.verbose);
            if(nr_edges_to_contract < A.rows() * opts.matching_thresh_crossover_ratio)
            {
                if (opts.verbose)
                {
                    std::cout << "# edges to contract = " << nr_edges_to_contract << ", # vertices = " << A.rows() << "\n";
                    std::cout << "switching to MST based contraction edge selection\n";
                }
                try_edges_to_contract_by_maximum_matching = false;    
            }
        }
        else
        {
            edge_contractions_woc c_mapper(A, opts.verbose);
            std::tie(cur_node_mapping, nr_edges_to_contract) = c_mapper.find_contraction_mapping();
        }


        if(nr_edges_to_contract == 0)
        {
            if (opts.verbose)
                std::cout << "# iterations = " << iter << "\n";
            break;
        }
        dCOO new_A = A.contract_cuda(cur_node_mapping);
        if (opts.verbose)
        {
            std::cout << "original A size " << A.cols() << "x" << A.rows() << "\n";
            std::cout << "contracted A size " << new_A.cols() << "x" << new_A.rows() << "\n";
        }
        assert(new_A.cols() < A.cols());

        if (opts.verbose)
        {
            const thrust::device_vector<float> diagonal = new_A.diagonal();
            const float energy_reduction = thrust::reduce(diagonal.begin(), diagonal.end());
            std::cout << "energy reduction " << energy_reduction << "\n";
        }
        if(has_bad_contractions(new_A))
            throw std::runtime_error("Found bad contractions");

        thrust::swap(A, new_A);
        A.remove_diagonal();
        if (opts.verbose)
            std::cout << "energy after iteration " << iter << ": " << A.sum() << ", #components = " << A.cols() << "\n";
        map_node_labels(cur_node_mapping, node_mapping);



         if (opts.run_preprocessor && opts.preprocessor_each_step) {
             auto [A_after_pp, pp_node_mapping] = preprocessor_cuda(A, opts,1);
             map_node_labels(pp_node_mapping,node_mapping);
             thrust::swap(A, A_after_pp);
         }


        if (opts.dump_timeline)
        {
            std::vector<int> current_timeline(node_mapping.size());
            thrust::copy(node_mapping.begin(), node_mapping.end(), current_timeline.begin());
            timeline.push_back(current_timeline);
        }
        if (opts.max_time_sec >= 0)
        {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
            if (time > opts.max_time_sec)
                break;
        }
    }

    if (opts.verbose)
        std::cout << "final energy = " << A.sum() << "\n";



    return {node_mapping, final_lb, timeline};
}

std::tuple<std::vector<int>, double, int, std::vector<std::vector<int>> > rama_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const multicut_solver_options& opts)
{
    initialize_gpu(opts.verbose);
    thrust::device_vector<int> i_gpu(i.begin(), i.end());
    thrust::device_vector<int> j_gpu(j.begin(), j.end());
    thrust::device_vector<float> costs_gpu(costs.begin(), costs.end());

    thrust::device_vector<int> sanitized_node_ids;
    if (opts.sanitize_graph)
        sanitized_node_ids = compute_sanitized_graph(i_gpu, j_gpu, costs_gpu);
    dCOO A(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu), true);


    double lb;
    std::vector<std::vector<int>> timeline;
    thrust::device_vector<int> node_mapping;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::tie(node_mapping, lb, timeline) = rama_cuda(A, opts);

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    int time_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (opts.sanitize_graph)
        node_mapping = desanitize_node_labels(node_mapping, sanitized_node_ids);
    std::vector<int> h_node_mapping(node_mapping.size());
    thrust::copy(node_mapping.begin(), node_mapping.end(), h_node_mapping.begin());
    return {h_node_mapping, lb, time_duration, timeline};
}

std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>> rama_cuda(thrust::device_vector<int>&& i, thrust::device_vector<int>&& j, thrust::device_vector<float>&& costs, const multicut_solver_options& opts, const int device)
{
    cudaSetDevice(device);
    thrust::device_vector<int> sanitized_node_ids;
    if (opts.sanitize_graph)
        sanitized_node_ids = compute_sanitized_graph(i, j, costs);

    dCOO A(std::move(j), std::move(i), std::move(costs), true);
    thrust::device_vector<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    
    std::tie(node_mapping, lb, timeline) = rama_cuda(A, opts);
    if (opts.sanitize_graph)
        node_mapping = desanitize_node_labels(node_mapping, sanitized_node_ids);

    return {node_mapping, lb, timeline};
}