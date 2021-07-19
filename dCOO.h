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
            dCOO(const int _rows, const int _cols,
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
                    bool is_sorted = false);

        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            dCOO(COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
                    bool is_sorted = false);

        dCOO(thrust::device_vector<int>&& _col_ids, thrust::device_vector<int>&& _row_ids, thrust::device_vector<float>&& _data, const bool is_sorted = false);
        dCOO(const int _rows, const int _cols, thrust::device_vector<int>&& _col_ids, thrust::device_vector<int>&& _row_ids, thrust::device_vector<float>&& _data, const bool is_sorted = false);

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t nnz() const { return data.size(); }
        float sum() const;
        float max() const;

        void remove_diagonal();

        thrust::device_vector<int> compute_row_offsets() const;

        const int* get_row_ids_ptr() const { return thrust::raw_pointer_cast(row_ids.data()); }
        const int* get_col_ids_ptr() const { return thrust::raw_pointer_cast(col_ids.data()); }
        const float* get_data_ptr() const { return thrust::raw_pointer_cast(data.data()); }
        float* get_writeable_data_ptr() { return thrust::raw_pointer_cast(data.data()); }

        const thrust::device_vector<int>& get_row_ids() const { return row_ids; }
        const thrust::device_vector<int>& get_col_ids() const { return col_ids; }
        const thrust::device_vector<float>& get_data() const { return data; }

        thrust::device_vector<float> diagonal() const;
        dCOO contract_cuda(const thrust::device_vector<int>& node_mapping);
        dCOO export_undirected() const;
        dCOO export_directed() const;
        dCOO export_filtered(const float lb, const float ub) const;

    private:
        void init(const bool is_sorted);
        int rows_ = 0;
        int cols_ = 0;
        thrust::device_vector<float> data;
        thrust::device_vector<int> row_ids;
        thrust::device_vector<int> col_ids; 
};

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

inline dCOO::dCOO(thrust::device_vector<int>&& _col_ids, thrust::device_vector<int>&& _row_ids, thrust::device_vector<float>&& _data, const bool is_sorted)
    : col_ids(std::move(_col_ids)),
    row_ids(std::move(_row_ids)),
    data(std::move(_data)) 
{
    init(is_sorted); 
}

inline dCOO::dCOO(const int _rows, const int _cols, thrust::device_vector<int>&& _col_ids, thrust::device_vector<int>&& _row_ids, thrust::device_vector<float>&& _data, const bool is_sorted)
    : rows_(_rows),
    cols_(_cols),
    col_ids(std::move(_col_ids)),
    row_ids(std::move(_row_ids)),
    data(std::move(_data)) 
{
    init(is_sorted); 
}

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dCOO::dCOO(const int _rows, const int _cols,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
        bool is_sorted)
    : rows_(_rows),
    cols_(_cols), 
    row_ids(row_id_begin, row_id_end),
    col_ids(col_id_begin, col_id_end),
    data(data_begin, data_end)
{
    init(is_sorted);
} 

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dCOO::dCOO(COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end,
        bool is_sorted)
    : row_ids(row_id_begin, row_id_end),
    col_ids(col_id_begin, col_id_end),
    data(data_begin, data_end)
{
    init(is_sorted);
}
