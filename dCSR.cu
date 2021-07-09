#include "dCSR.h"
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <ECLgraph.h>
#include "time_measure_util.h"
#include "utils.h"

void dCSR::print() const
{
    assert(rows() == row_offsets.size()-1);
    assert(col_ids.size() == data.size());
    std::cout << "dimension = " << rows() << "," << cols() << "\n";
    for(size_t i=0; i<rows(); ++i)
        for(size_t l=row_offsets[i]; l<row_offsets[i+1]; ++l)
            std::cout << i << ", " << col_ids[l] << ", " << data[l] << "\n"; 
}

dCSR dCSR::transpose(cusparseHandle_t handle) const
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
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
                return false;
            else
                return true;
        }
};

// computes an upperbound.
__global__ void contracted_node_degrees(const int num_contractions,
    const int* const __restrict__ in_node_degrees, 
    const int* const __restrict__ contract_rows, 
    const int* const __restrict__ contract_cols, 
    int* const __restrict__ out_node_degrees,
    bool* const __restrict__ v_to_contract,
    int* const __restrict__ v_new_labels)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < num_contractions; e += num_threads) // First element of both node_degrees is 0.
    {
        out_node_degrees[contract_rows[e]] += out_node_degrees[contract_cols[e]] - 1; // Assign neighbours of col to row and do not count col -> row edge.
        out_node_degrees[contract_cols[e]] = 0;
        v_to_contract[contract_rows[e]] = true;
        v_to_contract[contract_cols[e]] = true;
        v_new_labels[contract_cols[e]] = contract_rows[e];
    }
}

__global__ void copy_uncontracted_nodes(const int num_nodes,
                                        const int* const __restrict__ orig_row_offsets,
                                        const int* const __restrict__ orig_col_ids, 
                                        const int* const __restrict__ orig_data, 
                                        const bool* const __restrict__ v_to_contract,
                                        const int* const __restrict__ c_row_offsets,
                                        int* const __restrict__ c_col_ids, 
                                        int* const __restrict__ c_data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int n = tid; n < num_nodes; n += num_threads)
    {
        if (!v_to_contract[n])
        {
            int i_c = c_row_offsets[n];
            for(int i_o = orig_row_offsets[n]; i_o < orig_row_offsets[n + 1]; ++i_o)
            {
                int current_col = col_ids[i_o]; 
                // Only add edges starting from n. Also do not add edges between
                // n and a contracted edge as it will be done later.
                if (current_col < n && !v_to_contract[current_col])
                {
                    c_col_ids[i_c] = current_col;
                    c_data[i_c] = orig_data[i_o];
                    ++i_c;
                }
            }
        }
    }
}

__device__ float get_neighbour_index(const int n1, const int n2, 
                                    const int* const __restrict__ row_offsets,
                                    const int* const __restrict__ col_ids)
{
    for(int i = row_offsets[n1]; i != row_offsets[n1 + 1]; ++i) //TODO: Binary search?
        if (col_ids[i] == n2)
            return i;

    assert(false); // Other direction edge should always be present. 
}

__device__ float get_lowest_vertex_id_cost(const int n1, const int n2, 
                                        const int* const __restrict__ row_offsets,
                                        const int* const __restrict__ col_ids, 
                                        const float* const __restrict__ data,
                                        int& n1_index, int& n2_index, int& return_vertex_id)
{
    int n1_next_vertex = -1; 
    int n2_next_vertex = -1;
    if (n1_index < row_offsets[n1 + 1])
        n1_next_vertex = col_ids[n1_index];

    if (n2_index < row_offsets[n2 + 1])
        n2_next_vertex = col_ids[n2_index];

    if (n1_next_vertex < n2_next_vertex || (n1_next_vertex >= 0 && n2_next_vertex == -1))
    {
        return_vertex_id = n1_next_vertex;
        return data[n1_index++];
    }

    else if (n2_next_vertex < n1_next_vertex || (n2_next_vertex >= 0 && n1_next_vertex == -1))
    {
        return_vertex_id = n2_next_vertex;
        return data[n2_index++];
    }

    else if(n2_next_vertex == n1_next_vertex && n1_next_vertex >= 0)
    {
        return_vertex_id = n1_next_vertex;
        return data[n1_index++] + data[n2_index++];
    }

    return 0.0; // Both neighbour lists are exhausted.
}

// Changes graph structure for each contracted edge and also for each node incident to any node of a contracted edge.
__global__ void copy_contracted_nodes(const int num_contractions,
    const int* const __restrict__ orig_row_offsets,
    const int* const __restrict__ orig_col_ids, 
    const int* const __restrict__ orig_data, 
    const int* const __restrict__ contract_rows, 
    const int* const __restrict__ contract_cols, 
    const int* const __restrict__ v_new_labels,
    const int* const __restrict__ c_row_offsets,
    int* const __restrict__ c_col_ids,
    int* const __restrict__ c_data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < num_contractions; e += num_threads)
    {
        int n1 = contract_rows[e];
        int n2 = contract_cols[e]; 
        int n1_index = row_offsets[n1];
        int n2_index = row_offsets[n2];
        int new_index = c_row_offsets[n1]; // n2 is contracted towards n1
        while(1) // assign neighbours of n2 to n1 by making a directed edge starting from n1. 
        {
            int lowest_id_neighbour = -1;
            float merge_cost = get_lowest_vertex_id_cost(n1, n2, orig_row_offsets, orig_col_ids, orig_data, n1_index, n2_index, lowest_id_neighbour);
            if (lowest_id_neighbour == n2 || lowest_id_neighbour == n1)
                continue; // contracted edge.

            if (lowest_id_neighbour == -1)
                break;
            
            c_col_ids[new_index] = v_new_labels[lowest_id_neighbour];
            c_data[new_index] = merge_cost;
        }
    }
}

__global__ void make_symmetric(const int num_edges,
    const int* const __restrict__ e_index, 
    const int* const __restrict__ e_source_vertex, 
    const int* const __restrict__ row_offsets,
    const int* const __restrict__ col_ids,
    const int* const __restrict__ data,
    int* const __restrict__ data_symm)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < num_edges; e += num_threads)
    {
        int current_e_index = e_index[e];
        int n1 = e_source_vertex[e];
        int n2 = col_ids[current_e_index];

        int other_e_index = get_neighbour_index(n2, n1, row_offsets, col_ids, data);
        data_symm[other_e_index] += data[current_e_index];
    }
}

struct not_make_symmetric {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int> e)
        {
            return thrust::get<1>(e) == -1;
        }
};

//TODO: self-edges?
void dCSR::contract_matched_edges(const thrust::device_vector<int>& contract_rows, const thrust::device_vector<int>& contract_cols)
{
    int num_contractions = contract_rows.size();
    int threadCount = 256;
    int blockCount = ceil(num_contractions / (float) threadCount);

    // 1. calculate input node degrees:
    thrust::device_vector<int> in_node_degrees(rows_ + 1);
    thrust::adjacent_difference(row_offsets.begin(), row_offsets.end(), in_node_degrees.begin()); // First element of adjacent_difference is the original element i.e. 0.

    thrust::device_vector<int> new_row_offsets = in_node_degrees;

    // 2.1 Calculate output node degrees:
    thrust::device_vector<bool> v_to_contract(rows_, false);
    thrust::device_vector<int> v_new_labels(rows_);
    thrust::sequence(v_new_labels.begin(), v_new_labels.end(), 0);

    contracted_node_degrees<<<blockCount, threadCount>>>(num_contractions,
                                thrust::raw_pointer_cast(in_node_degrees.data()), 
                                thrust::raw_pointer_cast(contract_rows.data()), 
                                thrust::raw_pointer_cast(contract_cols.data()), 
                                thrust::raw_pointer_cast(new_row_offsets.data()),
                                thrust::raw_pointer_cast(v_to_contract.data()),
                                thrust::raw_pointer_cast(v_new_labels.data()));

    // 2.2 Convert output node degrees to offsets:
    thrust::inclusive_scan(new_row_offsets.begin(), new_row_offsets.end(), new_row_offsets.begin());

    // 3 Allocate new col ids and data:
    int new_nnz = new_row_offsets[rows_];
    thrust::device_vector<int> new_col_ids(new_nnz, -1);
    thrust::device_vector<float> new_data(new_nnz, 0.0);

    // 4. Create new graph containing original costs and 0 costs for newly introduced edges due to contraction.
    // i.e. for vertex i keep all its neighbours and their costs as-is and introduce new neighbours of i (with zero costs)
    // if a vertex j is contracted towards i. This operation should be done by taking into account symmetry of dCSR representation. 
    blockCount = ceil(rows_ / (float) threadCount);
    copy_uncontracted_nodes<<<blockCount, threadCount>>>(rows_,
        thrust::raw_pointer_cast(row_offsets.data()),
        thrust::raw_pointer_cast(col_ids.data()),
        thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(v_to_contract.data()),
        thrust::raw_pointer_cast(new_row_offsets.data()),
        thrust::raw_pointer_cast(new_col_ids.data()),
        thrust::raw_pointer_cast(new_data.data()));


    // 5. Now do the edge contraction.
    thrust::device_vector<int> e_consider_other_direction(new_nnz, -1);
    blockCount = ceil(num_contractions / (float) threadCount);
    copy_contracted_nodes<<<blockCount, threadCount>>>(num_contractions,
        thrust::raw_pointer_cast(row_offsets.data()),
        thrust::raw_pointer_cast(col_ids.data()),
        thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(contract_rows.data()), 
        thrust::raw_pointer_cast(contract_cols.data()), 
        thrust::raw_pointer_cast(v_new_labels.data()),
        thrust::raw_pointer_cast(new_row_offsets.data()),
        thrust::raw_pointer_cast(new_col_ids.data()),
        thrust::raw_pointer_cast(new_data.data()),
        thrust::raw_pointer_cast(e_consider_other_direction.data()));

    // 6. Make the edge costs symmetric:
    thrust::device_vector<int> e_indices(new_nnz);

    thrust::sequence(e_indices.begin(), e_indices.end(), 0);
    auto first = thrust::make_zip_iterator(thrust::make_tuple(e_indices.begin(), e_consider_other_direction.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(e_indices.end(), e_consider_other_direction.end()));
    auto valid_last = thrust::remove_if(first, last, not_make_symmetric());
    const int nr_edges_to_symm = std::distance(first, valid_last);
    e_indices.resize(nr_edges_to_symm);
    e_consider_other_direction.resize(nr_edges_to_symm);

    thrust::device_vector<float> new_data_symm = new_data;
    blockCount = ceil(nr_edges_to_symm / (float) threadCount);
    make_symmetric<<<blockCount, threadCount>>>(nr_edges_to_symm,
        thrust::raw_pointer_cast(e_indices.data()), 
        thrust::raw_pointer_cast(e_consider_other_direction.data()), 
        thrust::raw_pointer_cast(row_offsets.data()),
        thrust::raw_pointer_cast(col_ids.data()),
        thrust::raw_pointer_cast(data.data()), 
        thrust::raw_pointer_cast(data_symm.data()));
    
    // 7. TODO: Delete all vertices in contract_cols since they are merged to contract_rows and relabel vertices.
    
    return dCSR()
}


void dCSR::compress(cusparseHandle_t handle, const float tol)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    thrust::device_vector<int> _row_ids = row_ids(handle);
    
    auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), _row_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), _row_ids.end(), data.end()));

    auto new_last = thrust::remove_if(first, last, non_zero_indicator_func<float>(tol));

    const size_t nr_non_zeros = std::distance(first, new_last);
    col_ids.resize(nr_non_zeros);
    _row_ids.resize(nr_non_zeros);
    data.resize(nr_non_zeros);

    // remove_if is stable so sorting should not be required.
    // coo_sorting(handle, col_ids, _row_ids, data);

    // // now row indices are non-decreasing
    // assert(thrust::is_sorted(_row_ids.begin(), _row_ids.end()));

    cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    rows_ = _row_ids.back() + 1;

    row_offsets = thrust::device_vector<int>(rows_ + 1);
    cusparseXcoo2csr(handle, thrust::raw_pointer_cast(_row_ids.data()), nnz(), rows(), thrust::raw_pointer_cast(row_offsets.data()), CUSPARSE_INDEX_BASE_ZERO);
}

template <typename T>
struct keep_geq
{
    const T _thresh;
    keep_geq(T thresh): _thresh(thresh) {} 
   __host__ __device__ float operator()(const T &x) const
   {
     return x >= _thresh ? x : 0;
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
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
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

dCSR multiply_slow(cusparseHandle_t handle, dCSR& A, dCSR& B)
{
    float alpha = 1.0;
    MEASURE_FUNCTION_EXECUTION_TIME
    assert(A.cols() == B.rows());
    dCSR C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);

    csrgemm2Info_t info = NULL;
    cusparseCreateCsrgemm2Info(&info);

    size_t buffer_size;
    cusparseScsrgemm2_bufferSizeExt(handle, A.rows(), B.cols(), A.cols(), 
                                &alpha,
                                desc, A.nnz(), 
                                thrust::raw_pointer_cast(A.row_offsets.data()), 
                                thrust::raw_pointer_cast(A.col_ids.data()),
                                desc, B.nnz(),
                                thrust::raw_pointer_cast(B.row_offsets.data()), 
                                thrust::raw_pointer_cast(B.col_ids.data()),
                                NULL,
                                desc, B.nnz(), 
                                thrust::raw_pointer_cast(B.row_offsets.data()), 
                                thrust::raw_pointer_cast(B.col_ids.data()),
                                info, &buffer_size);
    void* buffer = NULL;
    cudaMalloc(&buffer, buffer_size);

    // Allocate memory for C
    C.rows_ = A.rows();
    C.cols_ = B.cols();
    C.row_offsets = thrust::device_vector<int>(A.rows()+1);
    int nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    cusparseXcsrgemm2Nnz(handle, A.rows(), B.cols(), A.cols(),
                        desc, A.nnz(),
                        thrust::raw_pointer_cast(A.row_offsets.data()), 
                        thrust::raw_pointer_cast(A.col_ids.data()),
                        desc, B.nnz(), 
                        thrust::raw_pointer_cast(B.row_offsets.data()), 
                        thrust::raw_pointer_cast(B.col_ids.data()),
                        desc, B.nnz(), 
                        thrust::raw_pointer_cast(B.row_offsets.data()), 
                        thrust::raw_pointer_cast(B.col_ids.data()),
                        desc, 
                        thrust::raw_pointer_cast(C.row_offsets.data()), 
                        nnzTotalDevHostPtr,
                        info, buffer);

    C.col_ids = thrust::device_vector<int>(nnzC);
    C.data = thrust::device_vector<float>(nnzC);

    cusparseScsrgemm2(handle, A.rows(), B.cols(), A.cols(), &alpha,
                            desc, A.nnz(), 
                            thrust::raw_pointer_cast(A.data.data()), 
                            thrust::raw_pointer_cast(A.row_offsets.data()), 
                            thrust::raw_pointer_cast(A.col_ids.data()),
                            desc, B.nnz(), 
                            thrust::raw_pointer_cast(B.data.data()), 
                            thrust::raw_pointer_cast(B.row_offsets.data()), 
                            thrust::raw_pointer_cast(B.col_ids.data()),
                            NULL,
                            desc, B.nnz(), 
                            thrust::raw_pointer_cast(B.data.data()), 
                            thrust::raw_pointer_cast(B.row_offsets.data()), 
                            thrust::raw_pointer_cast(B.col_ids.data()),
                            desc, 
                            thrust::raw_pointer_cast(C.data.data()), 
                            thrust::raw_pointer_cast(C.row_offsets.data()), 
                            thrust::raw_pointer_cast(C.col_ids.data()),
                            info, buffer);

    cusparseDestroyCsrgemm2Info(info);
    cusparseDestroyMatDescr(desc);
    cudaFree(buffer);

    return C;
}

dCSR multiply(cusparseHandle_t handle, dCSR& A, dCSR& B)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
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

thrust::device_vector<float> multiply(cusparseHandle_t handle, const dCSR& A, const thrust::device_vector<float>& x)
{
    throw std::runtime_error("not implemented yet"); 
    return thrust::device_vector<float>(0);
}

std::tuple<thrust::device_vector<int>, const thrust::device_vector<int>&, const thrust::device_vector<float>&> dCSR::export_coo(cusparseHandle_t handle) const
{
    thrust::device_vector<int> row_ids(nnz());

    cusparseXcsr2coo(handle, thrust::raw_pointer_cast(row_offsets.data()), nnz(), cols(), thrust::raw_pointer_cast(row_ids.data()), CUSPARSE_INDEX_BASE_ZERO); // TODO: should be rows?
            
    return {row_ids, col_ids, data}; 
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
thrust::device_vector<float> dCSR::diagonal(cusparseHandle_t handle) const
{
    assert(cols() == rows());
    thrust::device_vector<float> d(rows(), 0.0);

    thrust::device_vector<int> _row_ids = row_ids(handle);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), _row_ids.begin(), data.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), _row_ids.end(), data.end()));

    thrust::for_each(begin, end, diag_func({thrust::raw_pointer_cast(d.data())})); 

    return d;
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

thrust::device_vector<int> dCSR::compute_row_offsets(cusparseHandle_t handle, const int rows, const thrust::device_vector<int>& col_ids, const thrust::device_vector<int>& row_ids)
{
    assert(row_ids.size() == col_ids.size());
    assert(rows > *thrust::max_element(row_ids.begin(), row_ids.end()));
    assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));
    thrust::device_vector<int> row_offsets(rows+1);
    cusparseXcoo2csr(handle, thrust::raw_pointer_cast(row_ids.data()), row_ids.size(), rows, thrust::raw_pointer_cast(row_offsets.data()), CUSPARSE_INDEX_BASE_ZERO);
    return row_offsets;
}

void dCSR::print_info_of(const int i) const
{   
    std::cout<<"Row offsets of "<<i<<", start: "<<row_offsets[i]<<", end excl.: "<<row_offsets[i+1]<<std::endl;
    std::cout<<"Neighbours:"<<std::endl;
    for(size_t l=row_offsets[i]; l<row_offsets[i+1]; ++l)
        std::cout << i << "," << col_ids[l] << "," << data[l] << "\n"; 
}

__global__ void normalize_rows_cuda(const int num_rows, const int* const __restrict__ row_offsets, const int* const __restrict__ col_ids, float* __restrict__ data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int r = tid; r < num_rows; r += num_threads) 
    {
        float sum = 0.0;
        for(int l = row_offsets[r]; l < row_offsets[r + 1]; ++l)
            sum += data[l];

        for(int l = row_offsets[r]; l < row_offsets[r + 1]; ++l)
            data[l] /= sum;

        __syncthreads();
    }
}

void dCSR::normalize_rows()
{
    int threadCount = 256;
    int blockCount = ceil(rows_ / (float) threadCount);

    normalize_rows_cuda<<<blockCount, threadCount>>>(rows_, 
        thrust::raw_pointer_cast(row_offsets.data()), 
        thrust::raw_pointer_cast(col_ids.data()), 
        thrust::raw_pointer_cast(data.data()));
}