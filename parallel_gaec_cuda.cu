#include <cuda_runtime.h>
#include "dCSR.h"
#include "union_find.hxx"
#include "time_measure_util.h"
#include <algorithm>
#include <cstdlib>

int get_cuda_device()
{
    if(const char* cuda_env = std::getenv("CUDA_VISIBLE_DEVICES"))
    {
        std::cout << "Cuda device number to use = " << std::stoi(cuda_env) << "\n";
        return std::stoi(cuda_env); 
    }
    else
        return 0; 
}

template<typename ITERATOR>
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>> adjacency_edges(ITERATOR entry_begin, ITERATOR entry_end)
{
    const size_t nr_edges = std::distance(entry_begin, entry_end);
    thrust::host_vector<int> col_ids(2*nr_edges);
    thrust::host_vector<int> row_ids(2*nr_edges);
    thrust::host_vector<float> cost(2*nr_edges);
    for(auto it=entry_begin; it!=entry_end; ++it)
    {
        const int i = std::get<0>(*it);
        const int j = std::get<1>(*it);
        const float c = std::get<2>(*it);
        col_ids[2*std::distance(entry_begin, it)] = i;
        row_ids[2*std::distance(entry_begin, it)] = j;
        cost[2*std::distance(entry_begin, it)] = c;
        col_ids[2*std::distance(entry_begin, it)+1] = j;
        row_ids[2*std::distance(entry_begin, it)+1] = i;
        cost[2*std::distance(entry_begin, it)+1] = c;
    }
    return {col_ids, row_ids, cost};
}

std::tuple<dCSR,std::vector<int>> edge_contraction_matrix_cuda(cusparseHandle_t handle, const std::vector<std::array<int,2>>& edges, const int n)
{
    union_find uf(n);
    for(size_t c=0; c<edges.size(); ++c)
        uf.merge(edges[c][0],edges[c][1]);

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

    std::vector<int> col_ids;
    std::vector<int> row_ids;
    std::vector<float> data;
    for(int i=0; i<n; ++i)
    {
        assert(node_mapping[i] < c && node_mapping[i] >= 0);
        row_ids.push_back(i);
        col_ids.push_back(node_mapping[i]);
        data.push_back(1.0);
    }
    dCSR C(handle, col_ids.begin(), col_ids.end(), row_ids.begin(), row_ids.end(), data.begin(), data.end());
    //dCSR C(handle, row_ids.begin(), row_ids.end(), col_ids.begin(), col_ids.end(), data.begin(), data.end());
    std::cout << "edge contraction matrix dim: " << C.cols() << ", " << C.rows() << "\n";

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
            return thrust::get<2>(a) < thrust::get<2>(b);
        } 
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> edges_to_contract_cuda(cusparseHandle_t handle, dCSR& A, const size_t max_contractions)
{
    assert(max_contractions > 0);
    thrust::device_vector<int> col_ids;
    thrust::device_vector<int> row_ids;
    thrust::device_vector<float> data;
    std::tie(col_ids, row_ids, data) = A.export_coo(handle);
    std::cout << "adjacency matrix nr of edges = " << col_ids.size() << "\n";


    auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    auto new_last = thrust::remove_if(first, last, positive_edge_indicator_func());
    const size_t nr_positive_edges = std::distance(first, new_last);
    col_ids.resize(nr_positive_edges);
    row_ids.resize(nr_positive_edges);
    data.resize(nr_positive_edges);

    if(max_contractions < nr_positive_edges)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));
        thrust::sort(first, last, edge_comparator_func()); // TODO: faster through sort by keys?

        col_ids.resize(max_contractions);
        row_ids.resize(max_contractions);
        data.resize(max_contractions);
    }

    return {col_ids, row_ids};
}

std::vector<std::array<int,2>> edges_to_contract(cusparseHandle_t handle, dCSR& A, const size_t max_contractions)
{
    assert(max_contractions > 0);
    const auto A_coo = A.export_coo(handle);
    const auto& col_ids = std::get<0>(A_coo);
    const auto& row_ids = std::get<1>(A_coo);
    const auto& data = std::get<2>(A_coo);
    std::cout << "adjacency matrix nr of edges = " << col_ids.size() << "\n";
    std::vector<std::tuple<int,int,float>> positive_edges;
    for(size_t c=0; c<col_ids.size(); ++c)
    {
        const int i = col_ids[c];
        const int j = row_ids[c];
        const float x = data[c];
        //std::cout << i << "," << j << "," << x << "\n";
        if(i > j && x > 0.0)
            positive_edges.push_back({i, j, x});
    }
    if(max_contractions < positive_edges.size())
    {
        std::nth_element(positive_edges.begin(), positive_edges.begin() + max_contractions, positive_edges.end(), [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
        positive_edges.resize(max_contractions);
    }

    std::vector<std::array<int,2>> edge_indices;
    edge_indices.reserve(positive_edges.size());
    for(auto it=positive_edges.begin(); it!=positive_edges.end(); ++it)
    {
        const int i = std::get<0>(*it);
        const int j = std::get<1>(*it);
        edge_indices.push_back({i,j});
    }
    return edge_indices; 
}

std::vector<int> parallel_gaec_cuda(dCSR& A)
{
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");

    const double initial_lb = A.sum()/2.0;
    std::cout << "initial energy = " << initial_lb << "\n";

    std::vector<int> node_mapping(A.rows());
    std::iota(node_mapping.begin(), node_mapping.end(), 0);
    constexpr static double contract_ratio = 0.1;
    assert(A.rows() == A.cols());

    for(size_t iter=0;; ++iter)
    {
        //std::cout << "Adjacency matrix:\n";
        //std::cout << Eigen::MatrixXf(A) << "\n";
        const size_t nr_edges_to_contract = std::max(size_t(1), size_t(A.rows() * contract_ratio));
        
        const auto e = edges_to_contract(handle, A, nr_edges_to_contract);
        std::cout << "edges to contract size = " << e.size() << "\n";
        //std::cout << "iteration " << iter << ", edges to contract = " << e.size() << ", nr nodes remaining = " << A.rows() << "\n";
        if(e.size() == 0)
        {
            std::cout << "# iterations = " << iter << "\n";
            break;
        }
        dCSR C;
        std::vector<int> cur_node_mapping;
        std::tie(C, cur_node_mapping) = edge_contraction_matrix_cuda(handle, e, A.rows());
        for(size_t i=0; i<node_mapping.size(); ++i)
            node_mapping[i] = cur_node_mapping[node_mapping[i]];

        {
            MEASURE_FUNCTION_EXECUTION_TIME;

            assert(A.cols() == A.rows());
            std::cout << "A dim = " << A.cols() << "x" << A.rows() << "\n";
            std::cout << "A*C multiply time:\n";
            dCSR intermed = multiply(handle, A, C);
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
    }

    //std::cout << "solution:\n";
    //for(size_t i=0; i<node_mapping.size(); ++i)
    //    std::cout << i << " -> " << node_mapping[i] << "\n";
    const double lb = A.sum()/2.0;
    std::cout << "final energy = " << lb << "\n";

    cusparseDestroy(handle);
    return node_mapping;
}

std::vector<int> parallel_gaec_cuda(const std::vector<std::tuple<int,int,float>>& edges)
{
    MEASURE_FUNCTION_EXECUTION_TIME;

    const auto adj_edges = adjacency_edges(edges.begin(), edges.end());

    const int cuda_device = get_cuda_device();
    cudaSetDevice(cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);
    std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << ", device number " << cuda_device << "\n";
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");

    dCSR A(handle, 
            std::get<0>(adj_edges).begin(), std::get<0>(adj_edges).end(),
            std::get<1>(adj_edges).begin(), std::get<1>(adj_edges).end(),
            std::get<2>(adj_edges).begin(), std::get<2>(adj_edges).end());
    cusparseDestroy(handle);

    return parallel_gaec_cuda(A); 
}
