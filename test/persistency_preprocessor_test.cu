#include "persistency_preprocessor.h"

#include <test.h>

#include "multicut_solver_options.h"
#include "dCOO.h"

int main(int argc, char** argv)
{
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
        auto contracting_edges  = calculate_contracting_edges(A, node_edge_weights);
        test(contracting_edges[0] == 0);
        test(contracting_edges[1] == 0);
        test(contracting_edges[2] == 1);
        // The connected Subgraph is correct
        dCOO B = calculate_connected_subgraph(A, contracting_edges).export_undirected();
        test(B.get_row_ids().size() == 2);
        test(B.get_col_ids().size() == 2);
        // The contracted graph is correct
        dCOO C = calculate_contracted_graph(A, B);
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
        dCOO A = preprocessor_cuda(i, j, costs, opts);
        test(A.get_row_ids().size() == 3);
        test(A.get_col_ids().size() == 3);
        test(A.get_data()[0] == -1);
    }
    {
        multicut_solver_options opts;
        const std::vector<int> i = {0,0,1,2};
        const std::vector<int> j = {1,2,3,3};
        const std::vector<float> costs = {1.0,5.0,5.0,1.0};
        dCOO A = preprocessor_cuda(i, j, costs, opts);
        test(A.get_row_ids().size() == 1);
        test(A.get_col_ids().size() == 1);
        test(A.get_data()[0] == 2);
    }
    {
        // A square graph with ascending values gets contracted into a single point
        multicut_solver_options opts;
        const std::vector<int> i = {0,0,1,2};
        const std::vector<int> j = {1,2,3,3};
        const std::vector<float> costs = {1.0,4.0,2.0,3.0};
        dCOO A = preprocessor_cuda(i, j, costs, opts);
        test(A.get_row_ids().size() == 0);
        test(A.get_col_ids().size() == 0);
    }
    {
        multicut_solver_options opts;

        const std::vector<int> i = {0,0,1};
        const std::vector<int> j = {1,2,2};
        const std::vector<float> costs = {-2.0,-5.0,4.0};
        preprocessor_cuda(i, j, costs, opts);
    }

}