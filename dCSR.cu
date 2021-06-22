#include "dCSR.h"
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <ECLgraph.h>
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

    // make buffer
    void* dbuffer = NULL;
    size_t bufferSize = 0;
    checkCuSparseError(cusparseCsr2cscEx2_bufferSize(handle, rows(), cols(), nnz(), 
			thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(col_ids.data()),
			thrust::raw_pointer_cast(t.data.data()), thrust::raw_pointer_cast(t.row_offsets.data()), thrust::raw_pointer_cast(t.col_ids.data()), 
            CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize), "transpose buffer failed");
    
    checkCudaError(cudaMalloc((void**) &dbuffer, bufferSize), "transpose buffer allocation failed");

    checkCuSparseError(cusparseCsr2cscEx2(handle, rows(), cols(), nnz(), 
			thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(col_ids.data()),
			thrust::raw_pointer_cast(t.data.data()), thrust::raw_pointer_cast(t.row_offsets.data()), thrust::raw_pointer_cast(t.col_ids.data()), 
            CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, dbuffer),
            "transpose failed");

    cudaFree(dbuffer);
    return t;
}

template <typename T>
struct non_zero_indicator_func
{
    const T _tol;
    non_zero_indicator_func(T tol): _tol(tol) {} 

    __host__ __device__
        bool operator()(const thrust::tuple<int,int,float> t)
        {
            if(fabs(thrust::get<2>(t)) >= _tol)
                return true;
            else
                return false;
        }
};

void dCSR::compress(cusparseHandle_t handle, const float tol)
{
    MEASURE_FUNCTION_EXECUTION_TIME
    thrust::device_vector<int> _row_ids = row_ids(handle);
    
    auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), _row_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), _row_ids.end(), data.end()));

    auto new_last = thrust::remove_if(first, last, non_zero_indicator_func<float>(tol));

    const size_t nr_non_zeros = std::distance(first, new_last);
    col_ids.resize(nr_non_zeros);
    _row_ids.resize(nr_non_zeros);
    data.resize(nr_non_zeros);

    coo_sorting(col_ids, _row_ids, data);

    // now row indices are non-decreasing
    assert(thrust::is_sorted(_row_ids.begin(), _row_ids.end()));

    cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    rows_ = _row_ids.back() + 1;

    row_offsets = thrust::device_vector<int>(rows_ + 1);
    cusparseXcoo2csr(handle, thrust::raw_pointer_cast(_row_ids.data()), nnz(), rows(), thrust::raw_pointer_cast(row_offsets.data()), CUSPARSE_INDEX_BASE_ZERO);
}

// // Inspired from: https://docs.nvidia.com/cuda/cusparse/index.html#csr2csr_compress
// dCSR dCSR::compress(cusparseHandle_t handle, const float tol) const
// {
//     MEASURE_FUNCTION_EXECUTION_TIME
//     thrust::device_vector<int> nnz_per_row_interm = thrust::device_vector<int>(rows(), 0);
//     thrust::device_vector<int> total_nnz_interm = thrust::device_vector<int>(1);

//     cusparseMatDescr_t descrA;
//     checkCuSparseError(cusparseCreateMatDescr(&descrA), "Matrix descriptor init failed");
//     checkCuSparseError(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO), "cusparseSetMatIndex failed");
//     checkCuSparseError(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType failed");

//     checkCuSparseError(cusparseSnnz_compress(handle, rows(), descrA, thrust::raw_pointer_cast(data.data()),
//                                          thrust::raw_pointer_cast(row_offsets.data()), thrust::raw_pointer_cast(nnz_per_row_interm.data()),
//                                          thrust::raw_pointer_cast(total_nnz_interm.data()), tol), "cuSparse: stage 1 of compress failed");

//     dCSR c;
//     c.rows_ = rows();

//     c.row_offsets = thrust::device_vector<int>(rows() + 1); 
//     thrust::host_vector<int> total_nnz_interm_h = total_nnz_interm;
//     c.data = thrust::device_vector<float>(total_nnz_interm_h[0]); 
//     c.col_ids = thrust::device_vector<int>(total_nnz_interm_h[0]);

//     //DEBUG:
//     int max_rows = *thrust::max_element(row_offsets.begin(), row_offsets.end());
//     int max_cols = *thrust::max_element(col_ids.begin(), col_ids.end());
//     std::cout<<"max_rows: "<<max_rows<<", max_cols: "<<max_cols<<std::endl;
//     thrust::host_vector<int> row_offsets_h = row_offsets;
//     for (int i = 0; i < rows(); i++)
//     {
//         assert(row_offsets_h[i] >= 0);
//         assert(row_offsets_h[i + 1] >= row_offsets_h[i]);
//     }
//     assert(row_offsets_h[rows()] = nnz());
//     assert(nnz() == col_ids.size());
    
//     checkCuSparseError(cusparseScsr2csr_compress(handle, rows(), cols(), descrA, 
//                                               thrust::raw_pointer_cast(data.data()),
//                                               thrust::raw_pointer_cast(col_ids.data()), 
//                                               thrust::raw_pointer_cast(row_offsets.data()),
//                                               nnz(), thrust::raw_pointer_cast(nnz_per_row_interm.data()),
//                                               thrust::raw_pointer_cast(c.data.data()), 
//                                               thrust::raw_pointer_cast(c.col_ids.data()),
//                                               thrust::raw_pointer_cast(c.row_offsets.data()), tol), "cuSparse: stage 2 of compress failed");

//     c.cols_ = cols();
//     //DEBUG:
//     int max_rows_after = *thrust::max_element(row_offsets.begin(), row_offsets.end());
//     int max_cols_after = *thrust::max_element(col_ids.begin(), col_ids.end());
//     std::cout<<"max_rows: "<<max_rows_after<<", max_cols: "<<max_cols_after<<std::endl;

//     if (max_cols_after != max_cols or max_rows == 1182577)
//     {
//         for(int i = 54; i < 55; i++)
//         {
//             int j_n = row_offsets[i + 1];
//             int nnz_exp = nnz_per_row_interm[i];
//             for(int j_i = row_offsets[i]; j_i < j_n; j_i++)
//             {
//                 int j = col_ids[j_i];
//                 if (j == max_cols_after)
//                 {
//                     float val = data[j_i];
//                     int a = 1;
//                 }
//             }
//         }
//     }
//     return c;
// }

template <typename T>
struct keep_geq
{
    const T _thresh;
    keep_geq(T thresh): _thresh(thresh) {} 
   __host__ __device__ float operator()(const T &x) const
   {
     return x > _thresh ? x : 0;
   }
};

template <typename T>
struct is_positive
{
    __host__ __device__ bool operator()(const T &x)
    {
        return x > 0;
    }
};

dCSR dCSR::keep_top_k_positive_values(cusparseHandle_t handle, const int top_k)
{
    MEASURE_FUNCTION_EXECUTION_TIME
    // Create copy of self:
    dCSR p;
    p.rows_ = rows();
    p.cols_ = cols();
    p.row_offsets = row_offsets;
    p.col_ids = col_ids;
    p.data = data;

    // Set all negatives values to zero.
    thrust::transform(p.data.begin(), p.data.end(), p.data.begin(), keep_geq<float>(0.0f));
    int num_positive = thrust::count_if(thrust::device, p.data.begin(), p.data.end(), is_positive<float>());

    if (top_k < num_positive)
    {
        thrust::device_vector<float> temp = p.data;
        thrust::sort(temp.begin(), temp.end(), thrust::greater<float>()); // Ideal would be https://github.com/NVIDIA/thrust/issues/75

        float min_value_to_keep = temp[top_k];
        thrust::transform(p.data.begin(), p.data.end(), p.data.begin(), keep_geq<float>(min_value_to_keep));
    }

    p.compress(handle);

    return p;
}

dCSR multiply(cusparseHandle_t handle, dCSR& A, dCSR& B)
{
    MEASURE_FUNCTION_EXECUTION_TIME
    assert(A.cols() == B.rows());
    float duration;
    dCSR C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CUSPARSE API 
    cusparseSpMatDescr_t matA, matB, matC;
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    void* dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    int* rp = thrust::raw_pointer_cast(A.row_offsets.data());

    checkCuSparseError(cusparseCreateCsr(&matA, A.rows(), A.cols(), A.nnz(),
                                      thrust::raw_pointer_cast(A.row_offsets.data()), 
                                      thrust::raw_pointer_cast(A.col_ids.data()), 
                                      thrust::raw_pointer_cast(A.data.data()),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), "Matrix descriptor init failed");

    checkCuSparseError(cusparseCreateCsr(&matB, B.rows(), B.cols(), B.nnz(),
                                      thrust::raw_pointer_cast(B.row_offsets.data()), 
                                      thrust::raw_pointer_cast(B.col_ids.data()), 
                                      thrust::raw_pointer_cast(B.data.data()),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), "Matrix descriptor init failed");

    checkCuSparseError(cusparseCreateCsr(&matC, A.rows(), B.cols(), 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), "Matrix descriptor init failed");

    // SpGEMM Computation
    // ############################
    cudaEventRecord(start);
    // ############################
    
    cusparseSpGEMMDescr_t spgemmDesc;
    checkCuSparseError(cusparseSpGEMM_createDescr(&spgemmDesc), "sparse MM desc. failed");

    // ask bufferSize1 bytes for external memory
    checkCuSparseError(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL), "spGEMM work estimation 1 failed");

    checkCudaError(cudaMalloc((void**) &dBuffer1, bufferSize1), "buffer 1 allocation failed");

    // inspect the matrices A and B to understand the memory requirement for the next step
    checkCuSparseError(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1), "spGEMM work estimation 2 failed.");

    // ask bufferSize2 bytes for external memory
    checkCuSparseError(cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL), "cusparseSpGEMM_compute 1 failed");
    checkCudaError(cudaMalloc((void**) &dBuffer2, bufferSize2), "buffer 2 allocation failed");

    // compute A * B
    checkCuSparseError(cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2), "cusparseSpGEMM_compute 2 failed");
    // get matrix C sizes
    int64_t rows_C, cols_C, nnzC;
    checkCuSparseError(cusparseSpMatGetSize(matC, &rows_C, &cols_C, &nnzC), "matC get size failed");
    assert(rows_C == A.rows());
    assert(cols_C == B.cols());

    // Allocate memory for C
    C.rows_ = A.rows();
    C.cols_ = B.cols();
    C.row_offsets = thrust::device_vector<int>(A.rows()+1);
    C.col_ids = thrust::device_vector<int>(nnzC);
    C.data = thrust::device_vector<float>(nnzC);

    // update matC with the new pointers
    checkCuSparseError(cusparseCsrSetPointers(matC, thrust::raw_pointer_cast(C.row_offsets.data()), 
                                                    thrust::raw_pointer_cast(C.col_ids.data()), 
                                                    thrust::raw_pointer_cast(C.data.data())), "Setting matC pointers failed");

    // copy the final products to the matrix C.
    checkCuSparseError(cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc), "Copying to matC failed");

    // ############################
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // ############################

    cudaEventElapsedTime(&duration, start, stop);

    checkCuSparseError(cusparseSpGEMM_destroyDescr(spgemmDesc), "SPGEMM descriptor destruction failed");
    checkCuSparseError(cusparseDestroySpMat(matA), "Matrix descriptor destruction failed");
    checkCuSparseError(cusparseDestroySpMat(matB), "Matrix descriptor destruction failed");
    checkCuSparseError(cusparseDestroySpMat(matC), "Matrix descriptor destruction failed");
    checkCudaError(cudaFree(dBuffer1), "dBuffer1 free failed");
    checkCudaError(cudaFree(dBuffer2), "dBuffer2 free failed");

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

thrust::device_vector<int> dCSR::compute_cc(const int device)
{
    thrust::device_vector<int> cc_ids(rows());
    computeCC_gpu(rows(), nnz(), 
                thrust::raw_pointer_cast(row_offsets.data()), 
                thrust::raw_pointer_cast(col_ids.data()), 
                thrust::raw_pointer_cast(cc_ids.data()), device);
    return cc_ids;
}

void dCSR::print_info_of(const int i) const
{   
    std::cout<<"Row offsets of "<<i<<", start: "<<row_offsets[i]<<", end excl.: "<<row_offsets[i+1]<<std::endl;
    std::cout<<"Neighbours:"<<std::endl;
    for(size_t l=row_offsets[i]; l<row_offsets[i+1]; ++l)
        std::cout << i << "," << col_ids[l] << "," << data[l] << "\n"; 
}