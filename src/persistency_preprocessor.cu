#include <iostream>
#include <thrust/device_vector.h>
#include "rama_utils.h"
#include "rama_cuda.h"
#include <cmath>
#include <dCOO.h>
#include <ECLgraph.h>
#include <iomanip>
#include "multicut_solver_options.h"
#include "conflicted_cycles_cuda.h"

#define numThreads 256
#define maxNumberOfIterations 100

/**
 * Checks the triangle persistency criterion
 * @param number_of_triangles total number of triangles in the graph
 * @param row_offset graph row offset
 * @param col_ids graph column indices
 * @param costs graph edge costs
 * @param total_weights sum of absolute weights of outgoing edges for each node
 * @param pos_weights sum of positive weights of outgoing edges for each node (due to symmetry this is actually twice the sum of positive weights)
 * @param v1 triangle vertex 1
 * @param v2 triangle vertex 2
 * @param v3 triangle vertex 3
 * @param edges edges for which the triangle persistency criterion applies
 */
__global__ void check_persistency_criterion_triangle(const int number_of_triangles, const int *const row_offset,
                                                     const int *const col_ids, const float *const costs,
                                                     const float *const total_weights, const float *const pos_weights,
                                                     const int *const v1, const int *const v2, const int *const v3,
                                                     float *edges) {
        auto get_edge_cost = [=] (const int u, const int v) -> float {
            assert(u < v);
            for (int i = row_offset[u]; i < row_offset[u+1]; i++) {
                if (col_ids[i] == v) return costs[i];
            }
            return 0;
        };

        auto get_edge_index = [=] (const int u, const int v) -> int {
            for (int i = row_offset[u]; i < row_offset[u+1]; i++) {
                if (col_ids[i] == v) return i;
            }
            return 0;
        };

        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int i = start_index; i < number_of_triangles; i += num_threads) {

            const int u = v1[i];
            const int v = v2[i];
            const int w = v3[i];
            const float cost_uv = get_edge_cost(u,v);
            const float cost_uw = get_edge_cost(u,w);
            const float cost_vw = get_edge_cost(v,w);
            const float pos_neighborhood = (pos_weights[u] + pos_weights[v] + pos_weights[w])/2. - 2*(cost_uv)*(cost_uv > 0) - 2*(cost_uw)*(cost_uw > 0) - 2 * (cost_vw) * (cost_vw > 0);
            if (const bool pred_mc = (cost_uv + cost_uw + cost_vw) >= pos_neighborhood; not pred_mc) {
                continue;
            }
            const float cost_u = total_weights[v1[i]] - abs(cost_uv) - abs(cost_uw);
            const float cost_v = total_weights[v2[i]] - abs(cost_uv) - abs(cost_vw);
            const float cost_w = total_weights[v3[i]] - abs(cost_uw) - abs(cost_vw);
            const bool pred1 = (cost_uv + cost_uw) >= cost_u or (cost_uv + cost_uw) >= (cost_v + cost_w);
            const bool pred2 = (cost_uw + cost_vw) >= cost_w or (cost_uw + cost_vw) >= (cost_u + cost_v);
            const bool pred3 = (cost_uv + cost_vw) >= cost_v or (cost_uv + cost_vw) >= (cost_u + cost_w);
            if (pred1 or pred2) {
                edges[get_edge_index(u,w)] = 1;
            }
            if (pred1 or pred3) {
                edges[get_edge_index(u,v)] = 1;
            }
            if (pred2 or pred3) {
                edges[get_edge_index(v,w)] = 1;
            }
        }
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
    thrust::device_vector<float> costs_gpu(2*E + A.max_dim());
    thrust::copy(A.get_data().begin(), A.get_data().end(), costs_gpu.begin());
    thrust::transform(A.get_data().begin(),
        A.get_data().end(),
        costs_gpu.begin(),
        [=] __host__ __device__ (const float x) { return abs(x); }
        );
    thrust::counting_iterator<int> all_nodes;
    thrust::device_vector<int> nodes(2*E + A.max_dim());
    thrust::copy(all_nodes, all_nodes + A.max_dim(), nodes.begin() + 2*E);
    thrust::copy(A.get_row_ids().begin(),A.get_row_ids().end(), nodes.begin());
    thrust::copy(A.get_col_ids().begin(),A.get_col_ids().end(), nodes.begin()+E);
    thrust::copy(costs_gpu.begin(),costs_gpu.begin()+E, costs_gpu.begin()+E);
    thrust::sort_by_key(nodes.begin(), nodes.end(), costs_gpu.begin());
    const auto end = thrust::reduce_by_key(
        thrust::device, nodes.begin(),
        nodes.end(),
        costs_gpu.begin(),
        nodes.begin(),
        costs_gpu.begin()
        );
    auto size = end.first - nodes.begin();
    auto distance = std::distance(costs_gpu.begin(), costs_gpu.begin() + n);
    costs_gpu.resize(size);
    nodes.resize(size);
    return costs_gpu;
}

/**
 *
 * @param A The Graph
 * @param node_costs The absolute weights of the outgoing edges of all nodes
 * @param max_cycle_length The maximum length for a cycle to still be check (without effect)
 * @param tri_memory_factor The number of possible n-gons that can be found per edges (each n-gon will be found n times)
 * @param verbose If the output should be verbose
 * @return
 */
thrust::device_vector<float> calculate_contracting_edges_triangle_criterion(const dCOO& A, const thrust::device_vector<float>& node_costs, const int max_cycle_length, const float tri_memory_factor, const bool verbose) {
    dCOO A_pos;
    thrust::device_vector<int> row_ids_rep, col_ids_rep;
    auto min_rep_thresh = A.min() * 1e-4;
    std::tie(A_pos, row_ids_rep, col_ids_rep) = create_matrices(A, min_rep_thresh);
    auto B = A.export_undirected();
    thrust::device_vector<int> B_row_offset = B.compute_row_offsets();
    thrust::device_vector<int> A_pos_row_offsets = A_pos.compute_row_offsets();

    assert(A_pos_row_offsets.size() == A_pos.max_dim() + 1);
    int num_rep_edges = A.get_row_ids().size();
    int threadCount = 256;
    int blockCount = ceil(num_rep_edges / static_cast<float>(threadCount));
    const int max_num_tri = tri_memory_factor * num_rep_edges; // For memory pre-allocation.
    thrust::device_vector<int> triangles_v1(max_num_tri);
    thrust::device_vector<int> triangles_v2(max_num_tri);
    thrust::device_vector<int> triangles_v3(max_num_tri);
    thrust::device_vector<int> empty_tri_index(1, 0);
    find_triangles_parallel<<<blockCount, threadCount>>>(num_rep_edges,
        thrust::raw_pointer_cast(A.get_row_ids().data()),
        thrust::raw_pointer_cast(A.get_col_ids().data()),
        thrust::raw_pointer_cast(B_row_offset.data()),
        thrust::raw_pointer_cast(B.get_col_ids().data()),
        thrust::raw_pointer_cast(B.get_data().data()),
        thrust::raw_pointer_cast(triangles_v1.data()),
        thrust::raw_pointer_cast(triangles_v2.data()),
        thrust::raw_pointer_cast(triangles_v3.data()),
        thrust::raw_pointer_cast(empty_tri_index.data()),
        triangles_v1.size());
    auto begin = thrust::zip_iterator(thrust::make_tuple(triangles_v1.begin(), triangles_v2.begin(), triangles_v3.begin()));
    auto end = thrust::zip_iterator(thrust::make_tuple(triangles_v1.end(), triangles_v2.end(), triangles_v3.end()));
    auto new_end = thrust::unique(begin, end );
    auto size = new_end - begin;
    triangles_v1.resize(size-1);
    triangles_v2.resize(size-1);
    triangles_v3.resize(size-1);
    empty_tri_index[0] = size-1;
    if (verbose)
        std::cout << "Found: " << triangles_v1.size() << " triangles with " << A.get_col_ids().size() << " Total edges" << std::endl;
    thrust::device_vector<float> edges(A.get_col_ids().size());
    blockCount = ceil(edges.size() / (float) threadCount);
    check_persistency_criterion_triangle<<<blockCount, threadCount>>>(empty_tri_index[0],
                                                                      thrust::raw_pointer_cast(
                                                                              A.compute_row_offsets().data()),
                                                                      thrust::raw_pointer_cast(A.get_col_ids().data()),
                                                                      thrust::raw_pointer_cast(A.get_data().data()),
                                                                      thrust::raw_pointer_cast(node_costs.data()),
                                                                      thrust::raw_pointer_cast(
                                                                              calculate_sums(A_pos).data()),
                                                                      thrust::raw_pointer_cast(triangles_v1.data()),
                                                                      thrust::raw_pointer_cast(triangles_v2.data()),
                                                                      thrust::raw_pointer_cast(triangles_v3.data()),
                                                                      thrust::raw_pointer_cast(edges.data())
    );
    return edges;
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
            (t.get<2>() > 0) && ((ptr[t.get<0>()] - 2*t.get<2>() <= 0)  || (ptr[t.get<1>()] - 2*t.get<2>() <= 0) );
    };
    thrust::transform(thrust::device, begin, end,collapse_edge.begin(), edge_condition);
    return collapse_edge;
}


/**
 * Calculates the resulting subgraph
 * @param A initial graph
 * @param edges vector of edges that are primed to be contracted
 * @return subgraph of A with the given edges
 */
dCOO calculate_connected_subgraph(const dCOO& A, thrust::device_vector<float>& edges) {
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
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
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

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
 * Calculate the edges that are found to be persistent
 * @param A The initial graph
 * @param node_costs The vector containing the sums of the absolute costs for each node, should be calculated beforehand
 * @param opts The multicut solver options
 * @return A vector containing the edges that are found to be persistent
 */
thrust::device_vector<float> calculate_contracting_edges(dCOO& A, const thrust::device_vector<float>& node_costs, const multicut_solver_options opts) {
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    auto contracting_edges_edge_criterion = calculate_contracting_edges_edge_criterion(A, node_costs);
    auto edges = contracting_edges_edge_criterion;
    // Use this instead of the above line to combine edge and triangle criterion (generally no significant gain in persistency for high computational cost)
    // auto contracting_edges_triangle_criterion = calculate_contracting_edges_triangle_criterion(A, node_costs, opts.max_cycle_length_lb, opts.tri_memory_factor, opts.verbose)
    // thrust::device_vector<float> edges(contracting_edges_edge_criterion.size());
    // thrust::transform(contracting_edges_edge_criterion.begin(), contracting_edges_edge_criterion.end(), contracting_edges_triangle_criterion.begin(), edges.begin(),
    //     [=] __device__ __host__ (const float x, const float y) {
    //         return x or y;
    //     }
    // );
    if (opts.verbose) {
        std::cout << "Number of edges found by edge criterion: " << thrust::reduce(edges.begin(), edges.end()) << std::endl;
    }
    return edges;
}


/**
 * Calculates the reduced graph resulting from contracting the connected components given by the edges where the persistency criterion is met
 * @param A The initial graph
 * @param opts The multicut solver options
 * @param n Number of iterations the preprocessor should run for (if -1 it runs until no further changes are possible)
 * @return The reduced graph, contracted along persistent edges
 */
std::tuple<dCOO, thrust::device_vector<int>> preprocessor_cuda(dCOO& A, const multicut_solver_options& opts, const int n) {
    thrust::device_vector<int> node_mapping(A.max_dim());
    thrust::sequence(node_mapping.begin(), node_mapping.end());
    if (n == -1) {
        for(int i = 0; i < maxNumberOfIterations; i++) {
            float m_i = (float)A.get_col_ids().size();
            auto const node_costs = calculate_sums(A);
            auto contracting_edges = calculate_contracting_edges(A, node_costs, opts);
            const float M = thrust::reduce(contracting_edges.begin(), contracting_edges.end());
            if (opts.verbose)
                std::cout << "Percentage of Edges that are persistent: " << M/m_i*100.0 << "%" << std::endl;
            if (M < m_i * opts.preprocessor_threshold) return {A, node_mapping};
            if (M  == 0) return {A, node_mapping};

            const dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
            auto [C, new_node_mapping] = calculate_contracted_graph(A, B);
            map_node_labels(new_node_mapping,node_mapping);
            thrust::swap(A, C);
        }
        throw std::exception();
    }
    else {
        thrust::device_vector<int> current_node_mapping;
        for(int i = 0; i < n; i++) {
            float m_i = (float)A.get_col_ids().size();
            auto const node_costs = calculate_sums(A);
            auto contracting_edges = calculate_contracting_edges(A, node_costs, opts);
            const float M = thrust::reduce(contracting_edges.begin(), contracting_edges.end());
            if (opts.verbose)
                std::cout << "Percentage of Edges that are persistent: " << M/m_i*100.0 << "%" << std::endl;
            if (M < m_i * opts.preprocessor_threshold) return {A, node_mapping};
            const dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
            auto [C, new_node_mapping] = calculate_contracted_graph(A, B);
            map_node_labels(new_node_mapping,node_mapping);
            thrust::swap(A, C);
        }
        return {A, node_mapping};
    }
}


