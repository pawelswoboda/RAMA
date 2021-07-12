#include "dCOO.h"
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <ECLgraph.h>
#include "time_measure_util.h"
#include "utils.h"

    template<typename T>
void apply_permutation(thrust::device_vector<T>& keys, thrust::device_vector<int>& permutation)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    // copy keys to temporary vector
    thrust::device_vector<T> temp(keys.begin(), keys.end());

    // permute the keys
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}

void update_permutation(thrust::device_vector<int>& keys, thrust::device_vector<int>& permutation)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    // temporary storage for keys
    thrust::device_vector<int> temp(keys.size());

    // permute the keys with the current reordering
    thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}


void coo_sorting(thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    MEASURE_FUNCTION_EXECUTION_TIME;
    assert(row_ids.size() == col_ids.size());
    const size_t N = row_ids.size();
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    update_permutation(col_ids,  permutation);
    update_permutation(row_ids, permutation);

    apply_permutation(col_ids,  permutation);
    apply_permutation(row_ids, permutation);
    assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));
}

// TODO: remove this version
void coo_sorting(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids)
{
    coo_sorting(col_ids, row_ids);
}

void coo_sorting(thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids, thrust::device_vector<float>& data)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    MEASURE_FUNCTION_EXECUTION_TIME;
    assert(row_ids.size() == col_ids.size());
    assert(row_ids.size() == data.size());
    const size_t N = row_ids.size();
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    update_permutation(col_ids,  permutation);
    update_permutation(row_ids, permutation);

    apply_permutation(col_ids,  permutation);
    apply_permutation(row_ids, permutation);
    apply_permutation(data, permutation);
    assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));
}

// TODO: remove this version
void coo_sorting(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids, thrust::device_vector<float>& data)
{
    coo_sorting(col_ids, row_ids, data);
}



__global__ void map_nodes(const int num_edges, const int* const __restrict__ node_mapping, int* __restrict__ rows, int* __restrict__ cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < num_edges; e += num_threads)
    {
        rows[e] = node_mapping[rows[e]];
        cols[e] = node_mapping[cols[e]];
    }
}

struct is_same_edge
{
    __host__ __device__
        bool operator()(const thrust::tuple<int,int> e1, const thrust::tuple<int,int> e2)
        {
            if((thrust::get<0>(e1) == thrust::get<0>(e2)) && (thrust::get<1>(e1) == thrust::get<1>(e2)))
                return true;
            else
                return false;
        }
};

void dCOO::init()
{
    assert(col_ids.size() == data.size());
    assert(row_ids.size() == data.size());

    coo_sorting(col_ids, row_ids, data);

    // now row indices are non-decreasing
    assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));

    if(cols_ == 0)
        cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    assert(cols_ > *thrust::max_element(col_ids.begin(), col_ids.end()));
    if(rows_ == 0)
        rows_ = row_ids.back() + 1;
    assert(rows_ > *thrust::max_element(row_ids.begin(), row_ids.end()));
}

dCOO dCOO::contract_cuda(cusparseHandle_t handle, const thrust::device_vector<int>& node_mapping)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    const int numThreads = 256;

    thrust::device_vector<int> new_row_ids = row_ids;
    thrust::device_vector<int> new_col_ids = col_ids;
    thrust::device_vector<float> new_data = data;

    int num_edges = new_row_ids.size();
    int numBlocks = ceil(num_edges / (float) numThreads);

    map_nodes<<<numBlocks, numThreads>>>(num_edges, 
            thrust::raw_pointer_cast(node_mapping.data()), 
            thrust::raw_pointer_cast(new_row_ids.data()), 
            thrust::raw_pointer_cast(new_col_ids.data()));

    coo_sorting(handle, new_col_ids, new_row_ids, new_data); // in-place sorting by rows.

    auto first = thrust::make_zip_iterator(thrust::make_tuple(new_col_ids.begin(), new_row_ids.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(new_col_ids.end(), new_row_ids.end()));

    thrust::device_vector<int> out_rows(num_edges);
    thrust::device_vector<int> out_cols(num_edges);
    auto first_output = thrust::make_zip_iterator(thrust::make_tuple(out_cols.begin(), out_rows.begin()));
    thrust::device_vector<float> out_data(num_edges);

    auto new_end = thrust::reduce_by_key(first, last, new_data.begin(), first_output, out_data.begin(), is_same_edge());
    int new_num_edges = std::distance(out_data.begin(), new_end.second);
    out_rows.resize(new_num_edges);
    out_cols.resize(new_num_edges);
    out_data.resize(new_num_edges);

    int out_num_rows = out_rows.back() + 1;
    int out_num_cols = *thrust::max_element(out_cols.begin(), out_cols.end()) + 1;

    return dCOO(handle, out_num_rows, out_num_cols, 
            out_cols.begin(), out_cols.end(),
            out_rows.begin(), out_rows.end(), 
            out_data.begin(), out_data.end(),
            true);
}

struct is_diagonal
{
    __host__ __device__
        bool operator()(thrust::tuple<int,int,float> t)
        {
            return thrust::get<0>(t) == thrust::get<1>(t);
        }
};

void dCOO::remove_diagonal(cusparseHandle_t handle)
{
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    auto new_last = thrust::remove_if(begin, end, is_diagonal());
    int new_num_edges = std::distance(begin, new_last);
    col_ids.resize(new_num_edges);
    row_ids.resize(new_num_edges);
    data.resize(new_num_edges);
}

struct diag_func
{
    float* d;
    __host__ __device__
        void operator()(thrust::tuple<int,int,float> t)
        {
            if(thrust::get<0>(t) == thrust::get<1>(t))
            {
                assert(d[thrust::get<0>(t)] == 0.0);
                d[thrust::get<0>(t)] = thrust::get<2>(t);
            }
        }
};

thrust::device_vector<float> dCOO::diagonal(cusparseHandle_t handle) const
{
    assert(rows() == cols());
    thrust::device_vector<float> d(rows(), 0.0);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    thrust::for_each(begin, end, diag_func({thrust::raw_pointer_cast(d.data())})); 

    return d;
}

thrust::device_vector<int> dCOO::compute_row_offsets(cusparseHandle_t handle, const int rows, const thrust::device_vector<int>& col_ids, const thrust::device_vector<int>& row_ids)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        thrust::device_vector<int> row_offsets(rows+1);
    cusparseXcoo2csr(handle, thrust::raw_pointer_cast(row_ids.data()), row_ids.size(), rows, thrust::raw_pointer_cast(row_offsets.data()), CUSPARSE_INDEX_BASE_ZERO);
    return row_offsets;
}

thrust::device_vector<int> dCOO::compute_row_offsets(cusparseHandle_t handle) const
{
    return compute_row_offsets(handle, rows_, col_ids, row_ids);
}

float dCOO::sum()
{
    return thrust::reduce(data.begin(), data.end(), (float) 0.0, thrust::plus<float>());
}

dCOO dCOO::export_undirected()
{
    thrust::device_vector<int> row_ids_u, col_ids_u;
    thrust::device_vector<float> data_u;

    std::tie(row_ids_u, col_ids_u, data_u) = to_undirected(row_ids, col_ids, data);
    return dCOO(std::move(col_ids_u), std::move(row_ids_u), std::move(data_u));
}

dCOO dCOO::export_directed()
{
    thrust::device_vector<int> row_ids_d, col_ids_d;
    thrust::device_vector<float> data_d;

    std::tie(row_ids_d, col_ids_d, data_d) = to_directed(row_ids, col_ids, data);
    return dCOO(std::move(col_ids_d), std::move(row_ids_d), std::move(data_d));
}