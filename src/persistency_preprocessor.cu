#include <iostream>
#include <thrust/device_vector.h>
#include "rama_utils.h"
#include <cmath>
#include <dCOO.h>
#include <ECLgraph.h>
#include <graph.cuh>
#include <iomanip>
#include "multicut_solver_options.h"
#include "conflicted_cycles_cuda.h"

#define numThreads 256
#define maxNumberOfIterations 100

template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
    typedef typename Vector::value_type T;
    std::cout << "  " << std::setw(20) << name << "  ";
    thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

void print(std::string string) {
    std::cout << string << std::endl;
}


/**
 *  Calculate the sum of the absolute costs of all adjacent edges for each node
 *  @param A The graph
 *  @return A device vector containing the corresponding sums
*/
thrust::device_vector<float> calculate_sums(const dCOO& A) {
    thrust::device_vector<float> sums(A.max_dim(),0.0);
    const int E = A.get_row_ids().size();
    const int n = A.max_dim();
    thrust::device_vector<float> costs_gpu(2*E+ A.max_dim());
    thrust::copy(A.get_data().begin(), A.get_data().end(), costs_gpu.begin());
    using namespace thrust::placeholders;
    thrust::transform(A.get_data().begin(),
        A.get_data().end(),
        costs_gpu.begin(),
        [=] __host__ __device__ (const float x) {return abs(x); }
        );
    thrust::counting_iterator<int> all_nodes;
    thrust::device_vector<int> nodes(2*E + A.max_dim());
    thrust::copy(all_nodes, all_nodes + A.max_dim(), nodes.begin() + 2*E);
    thrust::copy(A.get_row_ids().begin(),A.get_row_ids().end(), nodes.begin());
    thrust::copy(A.get_col_ids().begin(),A.get_col_ids().end(), nodes.begin()+E);
    thrust::copy(costs_gpu.begin(),costs_gpu.begin()+E, costs_gpu.begin()+E);
    thrust::sort_by_key(nodes.begin(), nodes.end(), costs_gpu.begin());
    const auto end = thrust::reduce_by_key(thrust::device, nodes.begin(),nodes.end(), costs_gpu.begin(),nodes.begin(),costs_gpu.begin());
    auto size = end.first - nodes.begin();
    auto distance = std::distance(costs_gpu.begin(), costs_gpu.begin() + n);
    costs_gpu.resize(size);
    nodes.resize(size);
    return costs_gpu;
}


thrust::device_vector<float> calculate_contracting_edges_triangle_criterion(const dCOO& A, const thrust::device_vector<float>& node_costs) {
    dCOO A_pos;
    thrust::device_vector<int> row_ids_rep, col_ids_rep;
    auto min_rep_thresh = A.min() * 1e-4;
    return thrust::device_vector<float>();
    std::tie(A_pos, row_ids_rep, col_ids_rep) = create_matrices(A, min_rep_thresh);
}


/**
 * Calculates the edges that meet the persistency criterion for edges and can therefore be contracted
 * @param A The graph
 * @param node_costs The sums of the absolute costs of all adjacent edges for each node
 * @return A vector of the edges that can be contracted
 */
thrust::device_vector<float> calculate_contracting_edges_edge_criterion(const dCOO& A, const thrust::device_vector<float>& node_costs) {
    const int E = A.get_data().size();
    const auto begin = thrust::zip_iterator(thrust::make_tuple(A.get_row_ids().begin(), A.get_col_ids().begin(), A.get_data().begin()));
    const auto end = thrust::zip_iterator(thrust::make_tuple(A.get_row_ids().end(), A.get_col_ids().end(), A.get_data().end()));
    thrust::device_vector<float> collapse_edge(E, false);
    const auto ptr = thrust::raw_pointer_cast(node_costs.data());
    auto edge_condition = [=] __host__ __device__ (thrust::tuple<int, int, float> t){
        return
            (t.get<2>() >= 0) * ((ptr[t.get<0>()] - 2*t.get<2>() <= 0)  || (ptr[t.get<1>()] - 2*t.get<2>() <= 0) );
    };
    thrust::transform(thrust::device, begin, end,collapse_edge.begin(), edge_condition);
    return collapse_edge;
}

/**
 * Calculates the resulting subgraph
 * @param A inititial graph
 * @param edges vector of edges that are primed to be contraceted
 * @return subgraph of A wtih the given edges
 */
dCOO calculate_connected_subgraph(const dCOO& A, thrust::device_vector<float>& edges) {
    thrust::device_vector<int> i_gpu(A.get_row_ids());
    thrust::device_vector<int> j_gpu(A.get_col_ids());
    assert(i_gpu.size()==edges.size());
    const auto begin = thrust::zip_iterator(thrust::make_tuple(i_gpu.begin(), j_gpu.begin(), edges.begin()));
    const auto end = thrust::zip_iterator(thrust::make_tuple(i_gpu.end(), j_gpu.end(), edges.end()));
    const auto new_end = thrust::remove_if(begin, end , [=] __host__ __device__ (thrust::tuple<int,int,float> t){return t.get<2>() == 0;});
    const auto distance = thrust::distance(begin, new_end);
    i_gpu.resize(distance);
    j_gpu.resize(distance);
    edges.resize(distance);
    i_gpu.push_back(A.max_dim()-1);
    j_gpu.push_back(A.max_dim());
    edges.push_back(1);
    return dCOO(std::move(i_gpu), std::move(j_gpu), std::move(edges), true);
}

/**
 * Calculates the resulting graph, when contracting all connected components
 * @param A The initial graph
 * @param B The subgraph of contracting edges
 * @return The resulting graph
 */
std::tuple<dCOO,thrust::device_vector<int>> calculate_contracted_graph(dCOO& A, const dCOO& B) {
    if (B.cols() == 0) return {A, thrust::device_vector<int>()};
    auto col_ids = B.get_col_ids();
    auto row_offsets = B.compute_row_offsets();
    const int nnz = col_ids.size();
    const int num_rows = row_offsets.size()-1;
    thrust::device_vector<int> cc_labels(num_rows);
    computeCC_gpu(num_rows, nnz, thrust::raw_pointer_cast(row_offsets.data()),
        thrust::raw_pointer_cast(col_ids.data()),
        thrust::raw_pointer_cast(cc_labels.data())
        , get_cuda_device());
    auto C = A.contract_cuda(cc_labels);
    C.remove_diagonal();
    return {C,cc_labels};
}

/**
 * Calculates the reduced graph resulting from contracting the connected components given by the edges where the persistency criterion is met
 * @param i The row indices of the edges
 * @param j The column indices of the edges
 * @param costs The costs of the edges
 * @param opts The multicut solver options
 * @return The reduced Graph
 */
// dCOO preprocessor_cuda(const std::vector<int> &i, const std::vector<int> &j, const std::vector<float> &costs,
//                        const multicut_solver_options &opts){
//     initialize_gpu(opts.verbose);
//     thrust::device_vector<int> i_gpu(i.begin(), i.end());
//     thrust::device_vector<int> j_gpu(j.begin(), j.end());
//     thrust::device_vector<float> costs_gpu(costs.begin(), costs.end());
//     thrust::device_vector<int> sanitized_node_ids;
//     if constexpr (true)
//         sanitized_node_ids = compute_sanitized_graph(i_gpu, j_gpu, costs_gpu);
//     dCOO A(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu), true);
//     // Calculate the sum of the absolute costs of all adjacent edges for each node as an intermidiate step
//     const auto node_costs = calculate_sums(A);
//     auto contracting_edges = calculate_contracting_edges(A, node_costs);
//     if (const int n = thrust::count_if(contracting_edges.begin(), contracting_edges.end(), [=] __host__ __device__ (const float x) {
//         return x > 0;
//     }); n == 0) {
//         return A;
//     }
//     dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
//     dCOO C = calculate_contracted_graph(A, B);
//     return C;
// }



/**
 * Calculates the reduced graph resulting from contracting the connected components given by the edges where the persistency criterion is met
 * @param A The initial graph
 * @param opts The multicut solver options
 * @return The reduced Graph
 */
std::tuple<dCOO, thrust::device_vector<int>> preprocessor_cuda(dCOO& A, const multicut_solver_options& opts) {
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    const auto node_costs = calculate_sums(A);
    auto contracting_edges = calculate_contracting_edges_edge_criterion(A, node_costs);
    const int n = thrust::count_if(contracting_edges.begin(), contracting_edges.end(), [=] __host__ __device__ (const float x) {return x > 0; });
    if (opts.verbose)
        std::cout << "Number of Edges to rontract: " << n << std::endl;
    const dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
    dCOO C;
    thrust::device_vector<int> new_node_mapping;
    std::tie(C, new_node_mapping) = calculate_contracted_graph(A, B);
    if (opts.verbose)
        std::cout << "Reducing Graph Size From: " << A.get_col_ids().size() << " -> " << C.get_col_ids().size() <<std::endl;
    return {C, new_node_mapping};
}

inline void map_node_labels(const thrust::device_vector<int>& cur_node_mapping, thrust::device_vector<int>& orig_node_mapping) {
    auto node_mapper = [=] __host__ __device__ (const int n){
        return cur_node_mapping[n];
    };
    thrust::for_each(orig_node_mapping.begin(), orig_node_mapping.end(), node_mapper);
}

/**
 *
 * @param A The initial graph
 * @param opts The multicut solver options
 * @param n Number of iterations the preprocessor should run for (if -1 it runs until no further changes are possible)
 * @return The reduces graph
 */
std::tuple<dCOO, thrust::device_vector<int>> preprocessor_cuda(dCOO& A, const multicut_solver_options& opts, const int n) {
    thrust::device_vector<int> node_mapping(A.max_dim());
    thrust::sequence(node_mapping.begin(), node_mapping.end());
    if (n == -1) {
        for(int i = 0; i < maxNumberOfIterations; i++) {
            auto [B, current_node_mapping] = preprocessor_cuda(A, opts);
            map_node_labels(current_node_mapping,node_mapping);
            if (B.get_col_ids().size() == A.get_col_ids().size()) {
                if (opts.verbose)
                    std::cout << "Ran Preprocessor for :" << i << " Iterations"<< std::endl;
                return {A, node_mapping};
            }
            A = B;
        }
    }
    else {
        thrust::device_vector<int> current_node_mapping;
        for(int i = 0; i < n; i++) {
            std::tie(A, current_node_mapping) = preprocessor_cuda(A, opts);
            map_node_labels(current_node_mapping,node_mapping);
        }
    }
    return  {A, node_mapping};
}


