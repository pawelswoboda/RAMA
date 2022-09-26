#pragma once

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

inline int get_cuda_device()
{   
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

inline void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

inline void checkCudaError(cudaError_t status, std::string errorMsg)
{
    if (status != cudaSuccess) {
        std::cout << "CUDA error: " << errorMsg << ", status" <<cudaGetErrorString(status) << std::endl;
        throw std::exception();
    }
}

inline void initialize_gpu(bool verbose = false)
{
    const int cuda_device = get_cuda_device();
    cudaSetDevice(cuda_device);
    if (verbose)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cuda_device);
        std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << ", device number " << cuda_device << "\n";
    }
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// Assumes a symmetric CSR matrix.
// Initialize v1_mid_edge_index by row_offsets[v1] and v2_mid_edge_index by row_offsets[v2].
__device__ inline int compute_lowest_common_neighbour(const int v1, const int v2, 
                                            const int* const __restrict__ row_offsets, 
                                            const int* const __restrict__ col_ids, 
                                            const float* const __restrict__ data,
                                            int& v1_mid_edge_index, int& v2_mid_edge_index)
{
    while(v1_mid_edge_index < row_offsets[v1 + 1] && v2_mid_edge_index < row_offsets[v2 + 1])
    {
        const int v1_n = col_ids[v1_mid_edge_index];
        const int v2_n = col_ids[v2_mid_edge_index];
        if (v1_n == v2_n)
        {
            ++v1_mid_edge_index;
            ++v2_mid_edge_index;
            return v1_n;
        }
        if (v1_n < v2_n)
            ++v1_mid_edge_index;
        if (v1_n > v2_n)
            ++v2_mid_edge_index;
    }
    return -1;
}

template<typename ROW_ITERATOR, typename COL_ITERATOR>
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> to_undirected(
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end)
{
    assert(std::distance(row_id_begin, row_id_end) == std::distance(col_id_begin, col_id_end));

    const size_t nr_edges = std::distance(row_id_begin, row_id_end);
    thrust::device_vector<int> row_ids_u(2 * nr_edges);
    thrust::device_vector<int> col_ids_u(2 * nr_edges);

    thrust::copy(row_id_begin, row_id_end, row_ids_u.begin());
    thrust::copy(row_id_begin, row_id_end, col_ids_u.begin() + nr_edges);

    thrust::copy(col_id_begin, col_id_end, col_ids_u.begin());
    thrust::copy(col_id_begin, col_id_end, row_ids_u.begin() + nr_edges);

    return {row_ids_u, col_ids_u};
}


template<typename ROW_ITERATOR, typename COL_ITERATOR, typename DATA_ITERATOR>
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_undirected(
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
{
    assert(std::distance(data_begin, data_end) == std::distance(col_id_begin, col_id_end));
    assert(std::distance(data_begin, data_end) == std::distance(row_id_begin, row_id_end));

    const size_t nr_edges = std::distance(data_begin, data_end);
    thrust::device_vector<int> col_ids_u(2 * nr_edges);
    thrust::device_vector<int> row_ids_u(2 * nr_edges);
    thrust::device_vector<float> costs_u(2 * nr_edges);

    thrust::copy(row_id_begin, row_id_end, row_ids_u.begin());
    thrust::copy(row_id_begin, row_id_end, col_ids_u.begin() + nr_edges);

    thrust::copy(col_id_begin, col_id_end, col_ids_u.begin());
    thrust::copy(col_id_begin, col_id_end, row_ids_u.begin() + nr_edges);

    thrust::copy(data_begin, data_end, costs_u.begin());
    thrust::copy(data_begin, data_end, costs_u.begin() + nr_edges);

    return {row_ids_u, col_ids_u, costs_u};
}

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> to_undirected(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j)
{
    assert(i.size() == j.size());
    return to_undirected(i.begin(), i.end(), j.begin(), j.end());
}

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_undirected(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j, const thrust::device_vector<float>& costs)
{
    assert(i.size() == j.size() && i.size() == costs.size());
    return to_undirected(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
}

inline thrust::device_vector<int> offsets_to_degrees(const thrust::device_vector<int>& offsets)
{
    thrust::device_vector<int> degrees(offsets.size());
    thrust::adjacent_difference(offsets.begin(), offsets.end(), degrees.begin());
    return thrust::device_vector<int>(degrees.begin() + 1, degrees.end());
}

inline thrust::device_vector<int> degrees_to_offsets(const thrust::device_vector<int>& degrees)
{
    thrust::device_vector<int> offsets(degrees.size() + 1);
    thrust::exclusive_scan(degrees.begin(), degrees.end(), offsets.begin());
    offsets[offsets.size() - 1] = offsets[offsets.size() - 2] + degrees[degrees.size() - 1];
    return offsets;
}

inline thrust::device_vector<int> compress_label_sequence(const thrust::device_vector<int>& data, const int max_label)
{
    assert(*thrust::max_element(data.begin(), data.end()) <= max_label);

    // first get mask of used labels
    thrust::device_vector<int> label_mask(max_label + 1, 0);
    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + data.size(), data.begin(), label_mask.begin());

    // get map of original labels to consecutive ones
    thrust::device_vector<int> label_to_consecutive(max_label + 1);
    thrust::exclusive_scan(label_mask.begin(), label_mask.end(), label_to_consecutive.begin());

    // apply compressed label map
    thrust::device_vector<int> result(data.size(), 0);
    thrust::gather(data.begin(), data.end(), label_to_consecutive.begin(), result.begin());

    return result;
}

struct compute_lb
{
    __host__ __device__ double operator()(const float& val) const
    {
        return val < 0.0 ? val : 0.0;
    }
};

inline double get_lb(const thrust::device_vector<float>& costs)
{
    return thrust::transform_reduce(costs.begin(), costs.end(), compute_lb(), 0.0, thrust::plus<double>());
}

inline double get_obj(const std::vector<int>& h_node_mapping, const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs)
{
    double obj = 0;
    const int nr_edges = costs.size();
    for (int e = 0; e < nr_edges; e++)
    {
        const int e1 = i[e];
        const int e2 = j[e];
        const float c = costs[e];
        if (h_node_mapping[e1] != h_node_mapping[e2])
            obj += c;
    }
    return obj;
}

struct remove_reverse_edges_func {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,float> e)
        {
            return thrust::get<0>(e) >= thrust::get<1>(e);
        }
};

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_directed(const thrust::device_vector<int>& i_symm, const thrust::device_vector<int>& j_symm, const thrust::device_vector<float>& costs_symm)
{
    assert(i_symm.size() == j_symm.size() && i_symm.size() == costs_symm.size());
    thrust::device_vector<int> i = i_symm;
    thrust::device_vector<int> j = j_symm;
    thrust::device_vector<float> costs = costs_symm;
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin(), costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end(), costs.end()));
    auto new_last = thrust::remove_if(first, last, remove_reverse_edges_func());
    i.resize(std::distance(first, new_last));
    j.resize(std::distance(first, new_last));
    costs.resize(std::distance(first, new_last)); 

    return {i, j, costs};
}

struct sort_edge_nodes_func
{
    __host__ __device__
        void operator()(const thrust::tuple<int&,int&> t)
        {
            int& x = thrust::get<0>(t);
            int& y = thrust::get<1>(t);
            const int smallest = min(x, y);
            const int largest = max(x, y);
            x = smallest;
            y = largest;
        }
};

inline void sort_edge_nodes(thrust::device_ptr<int> i, thrust::device_ptr<int> j, const size_t size)
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i, j));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i + size, j + size));
    thrust::for_each(first, last, sort_edge_nodes_func());
}

inline void sort_edge_nodes(thrust::device_vector<int>& i, thrust::device_vector<int>& j)
{
    assert(i.size() == j.size());
    sort_edge_nodes(i.data(), j.data(), i.size());
}

struct map_values_func
{
    const int* mapping;
    size_t mapping_size;
    __host__ __device__ int operator()(const int& v)
    {
        assert(v >= 0);
        assert(v < mapping_size);
        return mapping[v];
    }
};


inline void coo_sorting(thrust::device_vector<int>& i, thrust::device_vector<int>& j, thrust::device_vector<int>& k)
{
    assert(i.size() == j.size());
    assert(i.size() == k.size());

    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin(), k.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end(), k.end()));

    thrust::sort(first, last);
}

inline void coo_sorting(thrust::device_ptr<int> i, thrust::device_ptr<int> j, const size_t size)
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i, j));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i + size, j + size));
    thrust::sort(first, last);
}

inline void coo_sorting(thrust::device_vector<int>& i, thrust::device_vector<int>& j)
{
    assert(i.size() == j.size());
    coo_sorting(i.data(), j.data(), i.size());
}

inline void coo_sorting(thrust::device_vector<int>& i, thrust::device_vector<int>& j, thrust::device_vector<float>& data)
{
    assert(i.size() == j.size());
    assert(i.size() == data.size());
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    thrust::sort_by_key(first, last, data.begin());
}

struct triangle_duplicate_nodes
{
    __host__ __device__
        bool operator()(const thrust::tuple<int,int,int>& t)
        {
            return thrust::get<0>(t) == thrust::get<1>(t) || 
                thrust::get<0>(t) == thrust::get<2>(t) ||
                thrust::get<1>(t) == thrust::get<2>(t);
        }
};

struct sort_triangle_nodes_func
{
    __host__ __device__
        void operator()(const thrust::tuple<int&,int&,int&> t)
        {
            int& x = thrust::get<0>(t);
            int& y = thrust::get<1>(t);
            int& z = thrust::get<2>(t);
            const int smallest = min(min(x, y), z);
            const int middle = max(min(x,y), min(max(x,y),z));
            const int largest = max(max(x, y), z);
            assert(smallest < middle && middle < largest);
            x = smallest;
            y = middle;
            z = largest;
        }
};

inline void normalize_triangles(thrust::device_vector<int>& t1, thrust::device_vector<int>& t2, thrust::device_vector<int>& t3)
{
    
    // bring triangles into normal form (first node < second node < third node)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        auto new_last = thrust::remove_if(first, last, triangle_duplicate_nodes());
        thrust::for_each(first, new_last, sort_triangle_nodes_func());
        t1.resize(std::distance(first, new_last)); 
        t2.resize(std::distance(first, new_last)); 
        t3.resize(std::distance(first, new_last)); 
    }

    // sort triangles and remove duplicates
    {
        coo_sorting(t1, t2, t3);
        assert(thrust::is_sorted(t1.begin(), t1.end()));
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        auto new_last = thrust::unique(first, last);
        t1.resize(std::distance(first, new_last)); 
        t2.resize(std::distance(first, new_last)); 
        t3.resize(std::distance(first, new_last)); 
    }
}

inline int rearrange_triangles(thrust::device_vector<int>& t1, thrust::device_vector<int>& t2, thrust::device_vector<int>& t3, int valid_num)
{
    
    auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.begin() + valid_num, t2.begin() + valid_num, t3.begin() + valid_num));
    auto new_last = thrust::remove_if(first, last, triangle_duplicate_nodes());
    thrust::for_each(first, new_last, sort_triangle_nodes_func());

    thrust::sort(first, new_last);
    auto new_last_unique = thrust::unique(first, new_last);
    return std::distance(first, new_last_unique);
}

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> get_unique_with_counts(const thrust::device_vector<int>& input)
{
    assert(thrust::is_sorted(input.begin(), input.end()));
    thrust::device_vector<int> unique_counts(input.size() + 1);
    thrust::device_vector<int> unique_values(input.size());

    auto new_end = thrust::unique_by_key_copy(input.begin(), input.end(), thrust::make_counting_iterator(0), unique_values.begin(), unique_counts.begin());
    int num_unique = std::distance(unique_values.begin(), new_end.first);
    unique_values.resize(num_unique);
    unique_counts.resize(num_unique + 1); // contains smallest index of each unique element.
    
    unique_counts[num_unique] = input.size();
    thrust::adjacent_difference(unique_counts.begin(), unique_counts.end(), unique_counts.begin());
    unique_counts = thrust::device_vector<int>(unique_counts.begin() + 1, unique_counts.end());

    return {unique_values, unique_counts};
}

inline thrust::device_vector<int> invert_unique(const thrust::device_vector<int>& values, const thrust::device_vector<int>& counts)
{
    thrust::device_vector<int> counts_sum(counts.size() + 1);
    counts_sum[0] = 0;
    thrust::inclusive_scan(counts.begin(), counts.end(), counts_sum.begin() + 1);
    
    int out_size = counts_sum.back();
    thrust::device_vector<int> output_indices(out_size, 0);

    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + values.size(), counts_sum.begin(), output_indices.begin());

    thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin());
    thrust::transform(output_indices.begin(), output_indices.end(), thrust::make_constant_iterator(1), output_indices.begin(), thrust::minus<int>());

    thrust::device_vector<int> out_values(out_size);
    thrust::gather(output_indices.begin(), output_indices.end(), values.begin(), out_values.begin());

    return out_values;
}

inline thrust::device_vector<int> compute_offsets(const thrust::device_vector<int>& i, const int max_value)
{
    assert(thrust::is_sorted(i.begin(), i.end()));
    thrust::device_vector<int> offsets(max_value + 2, 0);
    thrust::device_vector<int> unique_ids, counts;
    std::tie(unique_ids, counts) = get_unique_with_counts(i);
    thrust::transform(unique_ids.begin(), unique_ids.end(), thrust::make_constant_iterator<int>(1), unique_ids.begin(), thrust::plus<int>());
    thrust::scatter(counts.begin(), counts.end(), unique_ids.begin(), offsets.begin());
    thrust::inclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
    return offsets;
}

// Map old_values to 0:size(old_values) - 1 and replace all old values which are in src by this map.
// Values which are not in old_values are mapped to -1. 
inline void map_old_values_consec(thrust::device_vector<int>& src, 
                                const thrust::device_vector<int>& old_values, 
                                const int old_max_value)
{
    thrust::device_vector<int> mapping(old_max_value + 1, -1);
    thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + old_values.size(), 
                    old_values.begin(), mapping.begin());

    map_values_func mapper({thrust::raw_pointer_cast(mapping.data()), mapping.size()}); 
    thrust::transform(src.begin(), src.end(), src.begin(), mapper);
}

inline thrust::device_vector<int> compute_sanitized_graph(thrust::device_vector<int>& i, thrust::device_vector<int>& j, thrust::device_vector<float>& data)
{
    // First find and remove duplicate edges. The corresponding costs are also discarded!
    sort_edge_nodes(i, j); // makes all j > i 

    coo_sorting(i, j, data);
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    auto new_last = thrust::unique_by_key(first, last, data.begin());
    auto num_unique_edges = thrust::distance(first, new_last.first);
    i.resize(num_unique_edges);
    j.resize(num_unique_edges);
    data.resize(num_unique_edges);

    const int max_node_id = *thrust::max_element(j.begin(), j.end());
    thrust::device_vector<int> ids_mapping(max_node_id + 1, 0);
    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + i.size(), i.begin(), ids_mapping.begin());
    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + j.size(), j.begin(), ids_mapping.begin());

    thrust::exclusive_scan(ids_mapping.begin(), ids_mapping.end(), ids_mapping.begin());
    
    // Assign original node IDs to new IDs such that all nodes have incident edges in the new graph.
    map_values_func mapper({thrust::raw_pointer_cast(ids_mapping.data()), ids_mapping.size()}); 
    thrust::transform(i.begin(), i.end(), i.begin(), mapper);
    thrust::transform(j.begin(), j.end(), j.begin(), mapper);
     
    return ids_mapping;
}

struct desanitize_node_labels_func
{
    const int* ids_mapping;
    const int* node_labels_on_sanitized;
    const unsigned long ids_mapping_size;
    __host__ __device__ int operator()(const int input_graph_node_index)
    {
        const int current_node_mapping = ids_mapping[input_graph_node_index];
        if (input_graph_node_index == ids_mapping_size - 1)
            return node_labels_on_sanitized[current_node_mapping];
        if (current_node_mapping != ids_mapping[input_graph_node_index + 1]) 
            return node_labels_on_sanitized[current_node_mapping];
        return -1;
    }
};

inline void desanitize_node_labels(const thrust::device_vector<int>& node_labels, const thrust::device_vector<int>& ids_mapping, thrust::device_ptr<int> out_ptr)
{
    desanitize_node_labels_func desanitizer({thrust::raw_pointer_cast(ids_mapping.data()), 
                                            thrust::raw_pointer_cast(node_labels.data()), 
                                            ids_mapping.size()}); 

    thrust::transform(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + ids_mapping.size(), 
                    out_ptr, desanitizer);
}

inline thrust::device_vector<int> desanitize_node_labels(const thrust::device_vector<int>& node_labels, const thrust::device_vector<int>& ids_mapping)
{
    thrust::device_vector<int> all_nodes_labels(ids_mapping.size(), -1); // -1 label is for nodes which did not have any incident edges!
    desanitize_node_labels(node_labels, ids_mapping, all_nodes_labels.data());
    return all_nodes_labels;
}

template<typename T>
inline thrust::device_vector<T> concatenate(const thrust::device_vector<T>& a, const thrust::device_vector<T>& b)
{
    thrust::device_vector<T> ab(a.size() + b.size());
    thrust::copy(a.begin(), a.end(), ab.begin());
    thrust::copy(b.begin(), b.end(), ab.begin() + a.size());
    return ab;
}

template<typename T>
inline void print_vector(const thrust::device_vector<T>& v, const char* name, const int num = 0)
{
    std::cout<<name<<": ";
    if (num == 0)
        thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    else
    {
        int size = std::distance(v.begin(), v.end());
        thrust::copy(v.begin(), v.begin() + std::min(size, num), std::ostream_iterator<T>(std::cout, " "));
    }
    std::cout<<"\n";
}

struct pos_part
{
    __host__ __device__
        thrust::tuple<int, float> operator()(const thrust::tuple<int, float>& t)
        {
            if(thrust::get<1>(t) >= 0.0)
                return t;
            return thrust::make_tuple(0, 0.0f);
        }
};

struct tuple_sum
{
    template<typename T1, typename T2>
    __host__ __device__
        thrust::tuple<T1, T2> operator()(const thrust::tuple<T1, T2>& t1, const thrust::tuple<T1, T2>& t2)
        {
            return {thrust::get<0>(t1) + thrust::get<0>(t2), thrust::get<1>(t1) + thrust::get<1>(t2)};
        }
};

struct is_positive_edge
{
    const float tol;
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) > tol;
    }
};

/*
__host__ __device__
int min(const int a, const int b)
{
    if(a < b)
        return a;
    else
        return b; 
}
__host__ __device__
int max(const int a, const int b)
{
    if(a > b)
        return a;
    else
        return b; 
}
*/
