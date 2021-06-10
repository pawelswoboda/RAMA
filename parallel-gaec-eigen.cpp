#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <algorithm>
#include "union_find.hxx"
#include "time_measure_util.h"

#include "parallel-gaec-eigen.h"

std::tuple<Eigen::SparseMatrix<float>,std::vector<int>> edge_contraction_matrix(const std::vector<std::array<int,2>>& edges, const int n)
{
    // calculate nodes that will remain after contraction.
    // Filter out edges that are not needed for contraction.
    // Calculate transitive endpoints for remaining contraction edges
    
    union_find uf(n);

    // filter out unnecessary edges edges
    for(const auto [i,j] : edges)
        uf.merge(i,j);

    // compute node mapping from old set of nodes to new ones
    std::vector<char> node_id_present(n,false);
    for(int i=0; i<n; ++i)
        node_id_present[uf.find(i)] = 1;
    std::vector<int> uf_find_mapping(n, std::numeric_limits<int>::max());
    int c=0;
    for(int i=0; i<n; ++i)
        if(node_id_present[i])
            uf_find_mapping[i] = c++;

    assert(c == std::count(node_id_present.begin(), node_id_present.end(), 1));
    std::vector<int> node_mapping;
    node_mapping.reserve(n);
    for(int i=0; i<n; ++i)
    {
        assert(uf_find_mapping[uf.find(i)] != std::numeric_limits<int>::max());
        node_mapping.push_back( uf_find_mapping[uf.find(i)] );
    }

    // construct edge contraction matrix
    using T = Eigen::Triplet<float>;
    std::vector<T> coeffs;
    for(int i=0; i<n; ++i)
    {
        assert(node_mapping[i] < c && node_mapping[i] >= 0);
        coeffs.push_back({i, node_mapping[i], 1.0});
    }

    Eigen::SparseMatrix<float> C(n, c);
    C.setFromTriplets(coeffs.begin(), coeffs.end());
    return {C, node_mapping}; 
}

// first, filter out all negative edges. Second, get smallest valued ones.
std::vector<std::array<int,2>> edges_to_contract(Eigen::SparseMatrix<float>& A, const size_t max_contractions)
{
    assert(max_contractions > 0);
    std::vector<weighted_edge> positive_edges;
    for(int i=0; i<A.outerSize(); ++i)
    {
        for(Eigen::SparseMatrix<float>::InnerIterator it(A,i); it; ++it)
        {
            if(it.row() > it.col() && it.value() > 0.0)
            {
                positive_edges.push_back({it.row(), it.col(), it.value()});
            }
        }
    }
    if(max_contractions < positive_edges.size())
    {
        std::nth_element(positive_edges.begin(), positive_edges.begin() + max_contractions, positive_edges.end(), [](const auto& a, const auto& b) { return a.val > b.val; });
        positive_edges.resize(max_contractions);
    }

    std::vector<std::array<int,2>> edge_indices;
    edge_indices.reserve(positive_edges.size());
    for(const auto [i,j,x] : positive_edges)
        edge_indices.push_back({i,j});
    return edge_indices; 
}

double set_diagonal_to_zero(Eigen::SparseMatrix<float>& A)
{
    double diag_sum = 0.0;
    for(int i=0; i<A.outerSize(); ++i)
        for(Eigen::SparseMatrix<float>::InnerIterator it(A,i); it; ++it)
            if(it.row() == it.col())
            {
                diag_sum += it.value();
                it.valueRef() = 0.0;
            }
    return diag_sum;
}

std::vector<int> parallel_gaec(Eigen::SparseMatrix<float> A)
{
    double lb = A.sum()/2.0;
    std::cout << "initial energy = " << lb << "\n";

    std::vector<int> node_mapping(A.rows());
    std::iota(node_mapping.begin(), node_mapping.end(), 0);
    constexpr static double contract_ratio = 0.05;
    assert(A.rows() == A.cols());

    for(size_t iter=0;; ++iter)
    {
        //std::cout << "Adjacency matrix:\n";
        //std::cout << Eigen::MatrixXf(A) << "\n";
        const size_t nr_edges_to_contract = std::max(size_t(1), size_t(A.rows() * contract_ratio));
        
        const auto e = edges_to_contract(A, nr_edges_to_contract);
        //std::cout << "iteration " << iter << ", edges to contract = " << e.size() << ", nr nodes remaining = " << A.rows() << "\n";
        if(e.size() == 0)
        {
            std::cout << "# iterations = " << iter << "\n";
            break;
        }
        const auto [C, cur_node_mapping] = edge_contraction_matrix(e, A.rows());
        for(size_t i=0; i<node_mapping.size(); ++i)
            node_mapping[i] = cur_node_mapping[node_mapping[i]];
        A = C.transpose() * A * C;
        lb -= set_diagonal_to_zero(A)/2.0;
    }

    //std::cout << "solution:\n";
    //for(size_t i=0; i<node_mapping.size(); ++i)
    //    std::cout << i << " -> " << node_mapping[i] << "\n";
    lb = A.sum()/2.0;
    std::cout << "final energy = " << lb << "\n";
    return node_mapping;
}

Eigen::SparseMatrix<float> construct_adjacency_matrix(const std::vector<weighted_edge>& edges)
{
    using T = Eigen::Triplet<float>;
    std::vector<T> coeffs;
    coeffs.reserve(edges.size()*2);
    int n = 0;
    for(const auto [i,j,val] : edges)
    {
        assert(i != j);
        coeffs.push_back(T(i,j,val));
        coeffs.push_back(T(j,i,val));
        n = std::max({n,i+1,j+1});
    }
    Eigen::SparseMatrix<float> A(n, n);
    A.setFromTriplets(coeffs.begin(), coeffs.end());
    return A; 
}

std::vector<int> parallel_gaec(const std::vector<weighted_edge>& edges)
{
    MEASURE_FUNCTION_EXECUTION_TIME;
    Eigen::SparseMatrix<float> A = construct_adjacency_matrix(edges);
    return parallel_gaec(A); 
}
