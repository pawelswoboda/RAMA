#include "dCSR.h"
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include "time_measure_util.h"

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
    MEASURE_FUNCTION_EXECUTION_TIME
    dCSR t;
    t.cols_ = rows();
    t.rows_ = cols();

    t.row_offsets = thrust::device_vector<int>(cols()+1);
    t.col_ids = thrust::device_vector<int>(nnz());
    t.data = thrust::device_vector<float>(nnz());

    checkCuSparseError(cusparseScsr2csc(handle, rows(), cols(), nnz(), 
			thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(col_ids.data()),
			thrust::raw_pointer_cast(t.data.data()), thrust::raw_pointer_cast(t.col_ids.data()), thrust::raw_pointer_cast(t.row_offsets.data()),
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO),
            "transpose failed");

    return t;
}

// Inspired from: https://docs.nvidia.com/cuda/cusparse/index.html#csr2csr_compress
dCSR dCSR::eliminate_zeros(cusparseHandle_t handle, const float tol)
{
    MEASURE_FUNCTION_EXECUTION_TIME
    thrust::device_vector<int> nnz_per_row_interm = thrust::device_vector<int>(rows(), 0);
    thrust::device_vector<int> total_nnz_interm = thrust::device_vector<int>(1);

    cusparseMatDescr_t descrA;
    checkCuSparseError(cusparseCreateMatDescr(&descrA), "Matrix descriptor init failed");
    checkCuSparseError(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType failed");

    checkCuSparseError(cusparseSnnz_compress(handle, rows(), descrA, thrust::raw_pointer_cast(data.data()),
                                         thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(nnz_per_row_interm.data()),
                                         thrust::raw_pointer_cast(total_nnz_interm.data()), tol), "cuSparse: stage 1 of eliminate_zeros failed");

    dCSR c;

    c.row_offsets = thrust::device_vector<int>(rows() + 1); 
    c.data = thrust::device_vector<int>(total_nnz_interm[0]); // TODO: direct access to device memory?
    c.col_ids = thrust::device_vector<int>(total_nnz_interm[0]);

    checkCuSparseError(cusparseScsr2csr_compress(handle, rows(), cols(), descrA, 
                                              thrust::raw_pointer_cast(data.data()),
                                              thrust::raw_pointer_cast(col_ids.data()), 
                                              thrust::raw_pointer_cast(row_offsets.data()),
                                              nnz(), thrust::raw_pointer_cast(nnz_per_row_interm.data()),
                                              thrust::raw_pointer_cast(c.data.data()), 
                                              thrust::raw_pointer_cast(c.col_ids.data()),
                                              thrust::raw_pointer_cast(c.row_offsets.data()), tol), "cuSparse: stage 2 of eliminate_zeros failed");
                                            
    return c;
}

template<typename T>
struct keep_geq : public thrust::unary_function<T,T>
{
   __host__ __device__ T operator()(const T &x) const
   {
     return x > T(0) ? x : 0;
   }
};

struct is_positive
{
    __host__ __device__ bool operator()(int &x)
    {
        return x > 0;
    }
};

dCSR keep_top_k_positive_values(cusparseHandle_t handle, const int top_k)
{
    MEASURE_FUNCTION_EXECUTION_TIME
    // Create copy of self:
    dCSR p;
    p.cols_ = rows();
    p.rows_ = cols();
    p.row_offsets = row_offsets;
    p.col_ids = col_ids;
    p.data = data;

    // Set all negatives values to zero.
    thrust::transform(p.data.begin(), p.data.end(), p.data.begin(), keep_geq(0.0));
    int num_positive = thrust::count_if(p.data.begin(), p.data.end(), is_positive());

    if (top_k < num_positive)
    {
        thurst::device_vector<float> temp = p.data();
        thrust::sort(temp.begin(), temp.end(), thrust::greater<float>()); // Ideal would be https://github.com/NVIDIA/thrust/issues/75

        float min_value_to_keep = temp[top_k];
        thrust::transform(p.data.begin(), p.data.end(), p.data.begin(), keep_geq(min_value_to_keep));
    }

    return p.eliminate_zeros(handle);
}

dCSR multiply(cusparseHandle_t handle, const dCSR& A, const dCSR& B)
{
    MEASURE_FUNCTION_EXECUTION_TIME
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

