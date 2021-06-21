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

namespace {
void checkCuSparseError(cusparseStatus_t status, std::string errorMsg)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "CuSparse error: " << errorMsg << std::endl;
        throw std::exception();
    }
}
}

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

        dCSR transpose(cusparseHandle_t handle);
        dCSR eliminate_zeros(cusparseHandle_t handle, const float tol = 1e-4);
        dCSR keep_top_k_positive_values(cusparseHandle_t handle, const int top_k);
        thrust::device_vector<int> compute_cc(const int device);

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t nnz() const { return data.size(); }

        friend dCSR multiply(cusparseHandle_t handle, const dCSR& A, const dCSR& B);

        std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>> export_coo(cusparseHandle_t handle);

        thrust::device_vector<int> row_ids(cusparseHandle_t handle) const;
        void set_diagonal_to_zero(cusparseHandle_t handle);
        float sum();
        void print() const;

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

inline void coo_sorting(thrust::device_vector<int>& col_ids, thrust::device_vector<int>& row_ids, thrust::device_vector<float>& data)
{
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

    coo_sorting(col_ids, row_ids, data);

    // now row indices are non-decreasing
    assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));

    if(cols_ == 0)
        cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    assert(cols_ > *thrust::max_element(col_ids.begin(), col_ids.end()));
    if(rows_ == 0)
        rows_ = row_ids.back() + 1;
    assert(rows_ > *thrust::max_element(row_ids.begin(), row_ids.end()));

    row_offsets = thrust::device_vector<int>(row_ids.size()+1);
    thrust::sequence(row_offsets.begin(), row_offsets.end());

    row_offsets = thrust::device_vector<int>(rows_+1);
    cusparseXcoo2csr(handle, thrust::raw_pointer_cast(row_ids.data()), nnz(), rows(), thrust::raw_pointer_cast(row_offsets.data()), CUSPARSE_INDEX_BASE_ZERO);
}

dCSR multiply(cusparseHandle_t handle, const dCSR& A, const dCSR& B);
