#include "dCSR.h"
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

void dCSR::print() const
{
    assert(rows() == row_offsets.size()-1);
    assert(col_ids.size() == data.size());
    std::cout << "dimension = " << rows() << "," << cols() << "\n";
    for(size_t i=0; i<rows(); ++i)
        for(size_t l=row_offsets[i]; l<row_offsets[i+1]; ++l)
            std::cout << i << "," << col_ids[l] << "," << data[l] << "\n"; 
}

dCSR dCSR::transpose(cusparseHandle_t handle)
{
    dCSR t;
    t.cols_ = rows();
    t.rows_ = cols();

    std::cout << "t.row_offsets.size() = " << t.row_offsets.size() << "\n";
    t.row_offsets = thrust::device_vector<int>(cols()+1);
    std::cout << "t.row_offsets.size() after = " << t.row_offsets.size() << "\n";
    t.col_ids = thrust::device_vector<int>(nnz());
    t.data = thrust::device_vector<float>(nnz());

    checkCuSparseError(cusparseScsr2csc(handle, rows(), cols(), nnz(), 
			thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(col_ids.data()),
			thrust::raw_pointer_cast(t.data.data()), thrust::raw_pointer_cast(t.col_ids.data()), thrust::raw_pointer_cast(t.row_offsets.data()),
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO),
            "transpose failed");

    return t;
}

dCSR multiply(cusparseHandle_t handle, const dCSR& A, const dCSR& B)
{
    assert(A.cols() == B.rows());
    int nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    float duration;
    dCSR C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cusparseMatDescr_t descrA;
    cusparseMatDescr_t descrB;
    cusparseMatDescr_t descrC;

    checkCuSparseError(cusparseCreateMatDescr(&descrA), "Matrix descriptor init failed");
    checkCuSparseError(cusparseCreateMatDescr(&descrB), "Matrix descriptor init failed");
    checkCuSparseError(cusparseCreateMatDescr(&descrC), "Matrix descriptor init failed");
    checkCuSparseError(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType failed");
    checkCuSparseError(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO), "cusparseSetMatIndexBase failed");
    checkCuSparseError(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType failed");
    checkCuSparseError(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO), "cusparseSetMatIndexBase failed");
    checkCuSparseError(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType failed");
    checkCuSparseError(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO), "cusparseSetMatIndexBase failed");

    // ############################
    cudaEventRecord(start);
    // ############################

    // Allocate memory for row indices
    C.row_offsets = thrust::device_vector<int>(A.rows()+1);

    // Precompute number of nnz in C
    checkCuSparseError(cusparseXcsrgemmNnz(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                A.rows(), B.cols(), A.cols(),
                descrA, A.nnz(), thrust::raw_pointer_cast(A.row_offsets.data()), thrust::raw_pointer_cast(A.col_ids.data()),
                descrB, B.nnz(), thrust::raw_pointer_cast(B.row_offsets.data()), thrust::raw_pointer_cast(B.col_ids.data()),
                descrC, thrust::raw_pointer_cast(C.row_offsets.data()), nnzTotalDevHostPtr), "cuSparse: Precompute failed"
            );

    C.rows_ = A.rows();
    C.cols_ = B.cols();
    C.col_ids = thrust::device_vector<int>(nnzC);
    C.data = thrust::device_vector<float>(nnzC);

    // Compute SpGEMM
    checkCuSparseError(cusparseScsrgemm(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                A.rows(), B.cols(), A.cols(),
                descrA, A.nnz(), thrust::raw_pointer_cast(A.data.data()), thrust::raw_pointer_cast(A.row_offsets.data()), thrust::raw_pointer_cast(A.col_ids.data()),
                descrB, B.nnz(), thrust::raw_pointer_cast(B.data.data()), thrust::raw_pointer_cast(B.row_offsets.data()), thrust::raw_pointer_cast(B.col_ids.data()),
                descrC, thrust::raw_pointer_cast(C.data.data()), thrust::raw_pointer_cast(C.row_offsets.data()), thrust::raw_pointer_cast(C.col_ids.data())),
            "cuSparse: SpGEMM failed");

    // ############################
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // ############################

    cudaEventElapsedTime(&duration, start, stop);

    checkCuSparseError(cusparseDestroyMatDescr(descrA), "Matrix descriptor destruction failed");
    checkCuSparseError(cusparseDestroyMatDescr(descrB), "Matrix descriptor destruction failed");
    checkCuSparseError(cusparseDestroyMatDescr(descrC), "Matrix descriptor destruction failed");

    return C;
}

std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>> dCSR::export_coo(cusparseHandle_t handle)
{
    thrust::host_vector<int> h_col_ids(col_ids);
    thrust::host_vector<float> h_data(data);
    thrust::device_vector<int> row_ids(nnz());

    cusparseXcsr2coo(handle, thrust::raw_pointer_cast(row_offsets.data()), nnz(), cols(), thrust::raw_pointer_cast(row_ids.data()), CUSPARSE_INDEX_BASE_ZERO);
            
    thrust::host_vector<int> h_row_ids(row_ids);

    return {h_col_ids, h_row_ids, h_data}; 
}

thrust::device_vector<int> dCSR::row_ids(cusparseHandle_t handle) const
{
    thrust::device_vector<int> _row_ids(nnz());

    cusparseXcsr2coo(handle, thrust::raw_pointer_cast(row_offsets.data()), nnz(), cols(), thrust::raw_pointer_cast(_row_ids.data()), CUSPARSE_INDEX_BASE_ZERO);
            
    return _row_ids;
}

struct diag_to_zero_func
{
    __host__ __device__
        void operator()(thrust::tuple<int&,int&,float&> t)
        {
            if(thrust::get<0>(t) == thrust::get<1>(t))
                thrust::get<2>(t) = 0.0;
        }
};
void dCSR::set_diagonal_to_zero(cusparseHandle_t handle)
{
    thrust::device_vector<int> _row_ids = row_ids(handle);
    
     auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), _row_ids.begin(), data.begin()));
     auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), _row_ids.end(), data.end()));


     thrust::for_each(thrust::device, begin, end, diag_to_zero_func());
}

float dCSR::sum()
{
    return thrust::reduce(data.begin(), data.end(), (float) 0.0, thrust::plus<float>());
}

