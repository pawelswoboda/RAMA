#include "persistency_preprocessor.h"

#include <rama_utils.h>
#include <test.h>

#include "multicut_solver_options.h"
#include "dCOO.h"

inline dCOO createMockGraph(std::vector<int> i, std::vector<int> j, std::vector<float> costs) {
    thrust::device_vector<int> i_gpu(i.begin(), i.end());
    thrust::device_vector<int> j_gpu(j.begin(), j.end());
    thrust::device_vector<float> costs_gpu(costs.begin(), costs.end());
    dCOO A(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu), true);
    return A;
}

void test_preprocessor_with_edge_criterion() {
    {
        // Create a mock triangular graph
        const std::vector<int> i = {0,0,1};
        const std::vector<int> j = {1,2,2};
        const std::vector<float> costs = {-2.0,-5.0,4.0};
        thrust::device_vector<int> i_gpu(i.begin(), i.end());
        thrust::device_vector<int> j_gpu(j.begin(), j.end());
        thrust::device_vector<float> costs_gpu(costs.begin(), costs.end());
        dCOO A(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu), true);

        // The calculation of the sum of the absolute costs is correct for a mock graph
        const auto node_edge_weights = calculate_sums(A);
        test(node_edge_weights[0] == 7.0);
        test(node_edge_weights[1] == 6.0);
        test(node_edge_weights[2] == 9.0);
        // The resulting vector describing the contractable edges is correct
        auto contracting_edges  = calculate_contracting_edges_edge_criterion(A, node_edge_weights);
        test(contracting_edges[0] == 0);
        test(contracting_edges[1] == 0);
        test(contracting_edges[2] == 1);
        // The connected Subgraph is correct
        dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
        test(B.get_row_ids().size() == A.max_dim()+1);
        test(B.get_col_ids().size() == A.max_dim()+1);
        // The contracted graph is correct
        auto [C,_] = calculate_contracted_graph(A, B);
        test(C.get_row_ids().size() == 1);
        test(C.get_col_ids().size() == 1);
        test(C.get_data()[0] == -7);
    }
    {
        // A graph with only negative edges stays unchanged
        multicut_solver_options opts;
        const std::vector<int> i = {0,0,1};
        const std::vector<int> j = {1,2,2};
        const std::vector<float> costs = {-1.0,-1.0,-1.0};
        dCOO A = createMockGraph(i,j,costs);
        auto[ B,_] = preprocessor_cuda(A, opts,1);
        test(B.get_row_ids().size() == 3);
        test(B.get_col_ids().size() == 3);
        test(B.get_data()[0] == -1);
    }
    {
        multicut_solver_options opts;
        const std::vector<int> i = {0,0,1,2};
        const std::vector<int> j = {1,2,3,3};
        const std::vector<float> costs = {1.0,5.0,5.0,1.0};
        dCOO A = createMockGraph(i,j,costs);
        auto[ B,_]  = preprocessor_cuda(A, opts,1);
        test(B.get_row_ids().size() == 1);
        test(B.get_col_ids().size() == 1);
        test(B.get_data()[0] == 2);
    }
    {
        // A square graph with ascending values gets contracted into a single point
        multicut_solver_options opts;
        const std::vector<int> i = {0,0,1,2};
        const std::vector<int> j = {1,2,3,3};
        const std::vector<float> costs = {1.0,4.0,2.0,3.0};
        dCOO A = createMockGraph(i,j,costs);
        auto[ B,_]  = preprocessor_cuda(A, opts,1);
        test(B.get_row_ids().size() == 0);
        test(B.get_col_ids().size() == 0);
    }
    {
        multicut_solver_options opts;

        const std::vector<int> i = {0,0,1};
        const std::vector<int> j = {1,2,2};
        const std::vector<float> costs = {-2.0,-5.0,4.0};
        dCOO A = createMockGraph(i,j,costs);
        preprocessor_cuda(A, opts,1);
    }
    {
        multicut_solver_options opts;
        auto A = createMockGraph({0,0,1,1,2,2,3,4,4,5,5,6,7},
            {1,4,2,4,3,5,6,5,7,6,8,9,8},
            {1,3,4,-2,1,-1,1,-3,-2,-2,2,1,3});
        auto [B,_]= preprocessor_cuda(A, opts, 1);
        test(B.get_col_ids().size() == 5);
        test(thrust::reduce(B.get_data().begin(), B.get_data().end()) == -7);
    }
}

void test_preprocessor_with_triangle_criterion() {
    multicut_solver_options opts;
    const auto A = createMockGraph({0,0,1,1,2,2,3,4,4,5,5,5,6,7},
                                   {1,4,2,4,3,5,6,5,7,6,7,8,9,8},
                                   {1,3,4,-2,1,-1,1,-3,-2,-2,4,2,1,3});
    const auto weights = calculate_sums(A);
    const auto edges_tc = calculate_contracting_edges_triangle_criterion(A, weights, opts.max_cycle_length_lb, opts.tri_memory_factor, opts.verbose);
    const auto edges_ec = calculate_contracting_edges_edge_criterion(A, weights);
    test(edges_tc[10] == 1);
    test(edges_tc[11] == 1);
    test(edges_tc[13] == 1);
    test(edges_tc[3] == 0);
    test(edges_tc[7] == 0);
    test(edges_tc[9] == 0);
}

int main(int argc, char** argv)
{
    test_preprocessor_with_edge_criterion();
    test_preprocessor_with_triangle_criterion();
}