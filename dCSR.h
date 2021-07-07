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
#include "utils.h"

class dCSR {
    public:
        dCSR() {}

        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            dCSR(cusparseHandle_t handle, 
                    const int _rows, const int _cols,
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end);

        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            dCSR(cusparseHandle_t handle, 
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end);

        dCSR transpose(cusparseHandle_t handle) const;
        void compress(cusparseHandle_t handle, const float tol = 1e-4);
        dCSR keep_top_k_positive_values(cusparseHandle_t handle, const int top_k);
        thrust::device_vector<int> compute_cc(const int device);

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t nnz() const { return data.size(); }

        friend dCSR multiply(cusparseHandle_t handle, dCSR& A, dCSR& B);
        friend dCSR multiply_slow(cusparseHandle_t handle, dCSR& A, dCSR& B);
        friend thrust::device_vector<float> multiply(cusparseHandle_t handle, const dCSR& A, const thrust::device_vector<float>& x);

        std::tuple<thrust::device_vector<int>, const thrust::device_vector<int>&, const thrust::device_vector<float>&> export_coo(cusparseHandle_t handle) const;

        thrust::device_vector<int> row_ids(cusparseHandle_t handle) const;
        void set_diagonal_to_zero(cusparseHandle_t handle);
        float sum();
        void print() const;
        void print_info_of(const int i) const;

        static thrust::device_vector<int> compute_row_offsets(cusparseHandle_t handle, const int rows, const thrust::device_vector<int>& col_ids, const thrust::device_vector<int>& row_ids);
        void normalize_rows();

        const int* get_row_offsets_ptr() const { return thrust::raw_pointer_cast(row_offsets.data()); }
        const int* get_col_ids_ptr() const { return thrust::raw_pointer_cast(col_ids.data()); }
        const float* get_data_ptr() const { return thrust::raw_pointer_cast(data.data()); }
        float* get_writeable_data_ptr() { return thrust::raw_pointer_cast(data.data()); }

        const thrust::device_vector<int> get_row_offsets() const { return row_offsets; }
        const thrust::device_vector<int> get_col_ids() const { return col_ids; }
        const thrust::device_vector<float> get_data() const { return data; }

        thrust::device_vector<float> diagonal(cusparseHandle_t) const;

    private:
        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            void init(cusparseHandle_t handle,
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end);
        int rows_ = 0;
        int cols_ = 0;
        thrust::device_vector<float> data;
        thrust::device_vector<int> row_offsets;
        thrust::device_vector<int> col_ids; 
};

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
    thrust::device_vector<int> coo_cols(N);
    thrust::device_vector<int> coo_rows(N);
    thrust::device_vector<int> P(N); // permutation

    cusparseCreateIdentityPermutation(handle, N, thrust::raw_pointer_cast(P.data()));

    cusparseXcoosortByRow(handle, 1, 1, N, 
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
    thrust::device_vector<int> coo_cols(N);
    thrust::device_vector<int> coo_rows(N);
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

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dCSR::dCSR(cusparseHandle_t handle, 
        const int _rows, const int _cols,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
    : rows_(_rows),
    cols_(_cols)
{
    init(handle, col_id_begin, col_id_end, row_id_begin, row_id_end, data_begin, data_end);
} 

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dCSR::dCSR(cusparseHandle_t handle,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
{
    init(handle, col_id_begin, col_id_end, row_id_begin, row_id_end, data_begin, data_end);
}

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
void dCSR::init(cusparseHandle_t handle,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
{
    assert(std::distance(data_begin, data_end) == std::distance(col_id_begin, col_id_end));
    assert(std::distance(data_begin, data_end) == std::distance(row_id_begin, row_id_end));

    std::cout << "Allocation matrix with " << std::distance(data_begin, data_end) << " entries\n";
    col_ids = thrust::device_vector<int>(col_id_begin, col_id_end);
    data = thrust::device_vector<float>(data_begin, data_end);
    thrust::device_vector<int> row_ids(row_id_begin, row_id_end);

    coo_sorting(handle, col_ids, row_ids, data);

    // now row indices are non-decreasing
    assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));

    if(cols_ == 0)
        cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    assert(cols_ > *thrust::max_element(col_ids.begin(), col_ids.end()));
    if(rows_ == 0)
        rows_ = row_ids.back() + 1;
    assert(rows_ > *thrust::max_element(row_ids.begin(), row_ids.end()));

    row_offsets = compute_row_offsets(handle, rows(), col_ids, row_ids);
}



dCSR multiply(cusparseHandle_t handle, const dCSR& A, const dCSR& B);
dCSR multiply_slow(cusparseHandle_t handle, const dCSR& A, const dCSR& B);
thrust::device_vector<float> multiply(cusparseHandle_t handle, const dCSR& A, const thrust::device_vector<float>& x);
