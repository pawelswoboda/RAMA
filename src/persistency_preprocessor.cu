#include <iostream>
#include <thrust/device_vector.h>
#include "rama_utils.h"
#include <cmath>
#include <dCOO.h>
#include <ECLgraph.h>
#include <graph.cuh>
#include <iomanip>
#include "multicut_solver_options.h"

#define numThreads 256

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
    thrust::device_vector<float> sums(A.cols(),0.0);
    const int E = A.get_row_ids().size();
    const int n = A.max_dim();
    thrust::device_vector<float> costs_gpu(2*E);
    thrust::copy(A.get_data().begin(), A.get_data().end(), costs_gpu.begin());
    using namespace thrust::placeholders;
    thrust::transform(A.get_data().begin(),
        A.get_data().end(),
        costs_gpu.begin(),
        [=] __host__ __device__ (const float x) {return abs(x); }
        );
    thrust::device_vector<int> nodes(2*E);
    thrust::copy(A.get_row_ids().begin(),A.get_row_ids().end(), nodes.begin());
    thrust::copy(A.get_col_ids().begin(),A.get_col_ids().end(), nodes.begin()+E);
    thrust::copy(costs_gpu.begin(),costs_gpu.begin()+E, costs_gpu.begin()+E);
    thrust::sort_by_key(nodes.begin(), nodes.end(), costs_gpu.begin());
    const auto end = thrust::reduce_by_key(thrust::device, nodes.begin(),nodes.end(), costs_gpu.begin(),nodes.begin(),costs_gpu.begin());
    costs_gpu.resize(std::distance(costs_gpu.begin(), costs_gpu.begin() + n));
    return costs_gpu;
}

/**
 * Calculates the edges that meet the persistency criteria and can therefore be contracted
 * @param A The graph
 * @param node_costs The sums of the absolute costs of all adjacent edges for each node
 * @return A vector of the edges that can be contracted
 */
thrust::device_vector<float> calculate_contracting_edges(const dCOO& A, const thrust::device_vector<float>& node_costs) {
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
    return dCOO(std::move(i_gpu), std::move(j_gpu), std::move(edges), true);
}

/**
 * Calculates the resulting graph, when contracting all connected components
 * @param A The initial graph
 * @param B The subgraph of contracting edges
 * @return The resulting graph
 */
dCOO calculate_contracted_graph(dCOO& A, const dCOO& B) {
    if (B.cols() == 0) return A;
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
    return C;
}

/**
 * Calculates the reduced graph resulting from contracting the connected components given by the edges where the persistency criterion is met
 * @param i The row indices of the edges
 * @param j The column indices of the edges
 * @param costs The costs of the edges
 * @param opts The multicut solver options
 * @return The reduced Graph
 */
dCOO preprocessor_cuda(const std::vector<int> &i, const std::vector<int> &j, const std::vector<float> &costs,
                       const multicut_solver_options &opts){
    initialize_gpu(opts.verbose);
    thrust::device_vector<int> i_gpu(i.begin(), i.end());
    thrust::device_vector<int> j_gpu(j.begin(), j.end());
    thrust::device_vector<float> costs_gpu(costs.begin(), costs.end());
    thrust::device_vector<int> sanitized_node_ids;
    if constexpr (true)
        sanitized_node_ids = compute_sanitized_graph(i_gpu, j_gpu, costs_gpu);
    dCOO A(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu), true);
    // Calculate the sum of the absolute costs of all adjacent edges for each node as an intermidiate step
    const auto node_costs = calculate_sums(A);
    auto contracting_edges = calculate_contracting_edges(A, node_costs);
    if (const int n = thrust::count_if(contracting_edges.begin(), contracting_edges.end(), [=] __host__ __device__ (const float x) {
        return x > 0;
    }); n == 0) {
        return A;
    }
    dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
    dCOO C = calculate_contracted_graph(A, B);
    return C;
}
/**
 * Calculates the reduced graph resulting from contracting the connected components given by the edges where the persistency criterion is met
 * @param A The initial graph
 * @param opts The multicut solver options
 * @return The reduced Graph
 */
dCOO preprocessor_cuda(dCOO& A, const multicut_solver_options& opts) {
    initialize_gpu(opts.verbose);
    const auto node_costs = calculate_sums(A);
    auto contracting_edges = calculate_contracting_edges(A, node_costs);
    dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
    dCOO C = calculate_contracted_graph(A, B);
    return C;
}


