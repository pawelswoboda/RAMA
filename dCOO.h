#pragma once

#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <cusparse.h>
#include "time_measure_util.h"

class dCOO {
    public:
        dCOO() {}

        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            dCOO(cusparseHandle_t handle, 
                    const int _rows, const int _cols,
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
                    bool input_sorted = false);

        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            dCOO(cusparseHandle_t handle, 
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
                    bool input_sorted = false);

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t edges() const { return row_ids.size(); }
        float sum();

        void remove_diagonal(cusparseHandle_t handle);

        thrust::device_vector<int> compute_row_offsets(cusparseHandle_t handle) const;
        static thrust::device_vector<int> compute_row_offsets(cusparseHandle_t handle, const int rows, const thrust::device_vector<int>& col_ids, const thrust::device_vector<int>& row_ids);

        const int* get_row_ids_ptr() const { return thrust::raw_pointer_cast(row_ids.data()); }
        const int* get_col_ids_ptr() const { return thrust::raw_pointer_cast(col_ids.data()); }
        const float* get_data_ptr() const { return thrust::raw_pointer_cast(data.data()); }
        float* get_writeable_data_ptr() { return thrust::raw_pointer_cast(data.data()); }

        const thrust::device_vector<int> get_row_ids() const { return row_ids; }
        const thrust::device_vector<int> get_col_ids() const { return col_ids; }
        const thrust::device_vector<float> get_data() const { return data; }

        thrust::device_vector<float> diagonal(cusparseHandle_t) const;
        dCOO contract_cuda(cusparseHandle_t handle, const thrust::device_vector<int>& node_mapping);
        dCOO export_undirected(cusparseHandle_t handle);
        dCOO export_directed(cusparseHandle_t handle);

    private:
        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            void init(cusparseHandle_t handle,
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end, 
                    bool input_sorted);
        int rows_ = 0;
        int cols_ = 0;
        thrust::device_vector<float> data;
        thrust::device_vector<int> row_ids;
        thrust::device_vector<int> col_ids; 
};

inline void update_permutation(thrust::device_vector<int>& keys, thrust::device_vector<int>& permutation)
{
    // temporary storage for keys
    thrust::device_vector<int> temp(keys.size());

    // permute the keys with the current reordering
    thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}


    template<typename T>
void apply_permutation(thrust::device_vector<T>& keys, thrust::device_vector<int>& permutation)
{
    // copy keys to temporary vector
    thrust::device_vector<T> temp(keys.begin(), keys.end());

    // permute the keys
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}

inline void coo_sorting(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
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

inline void coo_sorting(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids, thrust::device_vector<float>& data)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
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

/*
inline void coo_sorting(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    assert(row_ids.size() == col_ids.size());
    const size_t N = row_ids.size();

    size_t buffer_size_in_bytes = 0;
    cusparseXcoosort_bufferSizeExt(handle, 1, 1, N,
            thrust::raw_pointer_cast(row_ids.data()), thrust::raw_pointer_cast(col_ids.data()), 
            &buffer_size_in_bytes);

    thrust::device_vector<char> buffer(buffer_size_in_bytes);
    thrust::device_vector<int> P(N); // permutation

    cusparseCreateIdentityPermutation(handle, N, thrust::raw_pointer_cast(P.data()));

    cusparseXcoosortByRow(handle, 1, 1, N, // 1, 1 are not used.
            thrust::raw_pointer_cast(row_ids.data()), 
            thrust::raw_pointer_cast(col_ids.data()), 
            thrust::raw_pointer_cast(P.data()), 
            thrust::raw_pointer_cast(buffer.data()));
}

inline void coo_sorting(cusparseHandle_t handle, thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids, thrust::device_vector<float>& data)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    assert(row_ids.size() == col_ids.size());
    assert(row_ids.size() == data.size());
    const size_t N = row_ids.size();

    size_t buffer_size_in_bytes = 0;
    cusparseXcoosort_bufferSizeExt(handle, 1, 1, data.size(),
            thrust::raw_pointer_cast(row_ids.data()), thrust::raw_pointer_cast(col_ids.data()), 
            &buffer_size_in_bytes);

    thrust::device_vector<char> buffer(buffer_size_in_bytes);
    thrust::device_vector<float> coo_data(N);
    thrust::device_vector<int> P(N); // permutation

    cusparseCreateIdentityPermutation(handle, N, thrust::raw_pointer_cast(P.data()));

    cusparseXcoosortByRow(handle, 1, 1, N, 
            thrust::raw_pointer_cast(row_ids.data()), 
            thrust::raw_pointer_cast(col_ids.data()), 
            thrust::raw_pointer_cast(P.data()), 
            thrust::raw_pointer_cast(buffer.data()));

    // TODO: deprecated, replace by cusparseGather
    cusparseSgthr(handle, N,
            thrust::raw_pointer_cast(data.data()),
            thrust::raw_pointer_cast(coo_data.data()),
            thrust::raw_pointer_cast(P.data()), 
            CUSPARSE_INDEX_BASE_ZERO); 
    data = coo_data;
}
*/

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dCOO::dCOO(cusparseHandle_t handle, 
        const int _rows, const int _cols,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
        bool input_sorted)
    : rows_(_rows),
    cols_(_cols)
{
    init(handle, col_id_begin, col_id_end, row_id_begin, row_id_end, data_begin, data_end, input_sorted);
} 

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dCOO::dCOO(cusparseHandle_t handle,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
        bool input_sorted)
{
    init(handle, col_id_begin, col_id_end, row_id_begin, row_id_end, data_begin, data_end, input_sorted);
}

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
void dCOO::init(cusparseHandle_t handle,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end, 
        bool input_sorted)
{
    assert(std::distance(data_begin, data_end) == std::distance(col_id_begin, col_id_end));
    assert(std::distance(data_begin, data_end) == std::distance(row_id_begin, row_id_end));

    std::cout << "Allocation matrix with " << std::distance(data_begin, data_end) << " entries\n";
    row_ids = thrust::device_vector<int>(row_id_begin, row_id_end);
    col_ids = thrust::device_vector<int>(col_id_begin, col_id_end);
    data = thrust::device_vector<float>(data_begin, data_end);

    if (!input_sorted)
        coo_sorting(handle, col_ids, row_ids, data);

    // now row indices are non-decreasing
    assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));

    if(cols_ == 0)
        cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    assert(cols_ > *thrust::max_element(col_ids.begin(), col_ids.end()));
    if(rows_ == 0)
        rows_ = row_ids.back() + 1;
    assert(rows_ > *thrust::max_element(row_ids.begin(), row_ids.end()));
}
