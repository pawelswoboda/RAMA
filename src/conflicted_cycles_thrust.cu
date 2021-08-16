#include "conflicted_cycles_thrust.h"
#include "parallel_gaec_utils.h"

#define tol 1e-6

__global__ void copy_neighbourhood(const int num_edges,
    const int* const __restrict__ v_frontier_offsets,
    const int* const __restrict__ expanded_src_compressed,
    const int* const __restrict__ expanded_src_offsets,
    const int* const __restrict__ expanded_dst,
    const int* const __restrict__ expanded_edge_index,
    const float* const __restrict__ expanded_edge_value,
    const int* const __restrict__ v_frontier_parent_edge,
    const int* const __restrict__ v_frontier_bottleneck_edge_index,
    const float* const __restrict__ v_frontier_bottleneck_edge_value,
    const int* const __restrict__ v_frontier_parent_node,
    const int* const __restrict__ v_frontier_all_src_offsets,
    int* __restrict__ v_frontier_all_neighbours,
    int* __restrict__ v_frontier_all_parent_edge,
    int* __restrict__ v_frontier_all_bottleneck_edge_index,
    float* __restrict__ v_frontier_all_bottleneck_edge_value)
{
    const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;

    for (int e = start_index; e < num_edges; e += num_threads) 
    {
        const int src_compressed = expanded_src_compressed[e];
        const int parent_start_offset = v_frontier_offsets[src_compressed];
        const int parent_end_offset = v_frontier_offsets[src_compressed + 1];
        int output_start_offset = v_frontier_all_src_offsets[src_compressed] + (e - expanded_src_offsets[src_compressed]) * (parent_end_offset - parent_start_offset);
        const int dst = expanded_dst[e];
        const float new_edge_value = expanded_edge_value[e];
        // Write same vertices with different parents:
        for (int v_frontier_parent_index = parent_start_offset; v_frontier_parent_index != parent_end_offset; ++output_start_offset, ++v_frontier_parent_index)
        {
            const int parent_node = v_frontier_parent_node[v_frontier_parent_index];
            if (dst == parent_node)
            {
                v_frontier_all_neighbours[output_start_offset] = -1; // mark for deletion
            }
            else
            {
                const int parent_edge = v_frontier_parent_edge[v_frontier_parent_index];
                const float prev_edge_value = v_frontier_bottleneck_edge_value[v_frontier_parent_index];
                if (new_edge_value < prev_edge_value)
                { // Update bottleneck edge
                    v_frontier_all_bottleneck_edge_index[output_start_offset] = expanded_edge_index[e];
                    v_frontier_all_bottleneck_edge_value[output_start_offset] = new_edge_value;
                }
                else
                { // Keep old information
                    v_frontier_all_bottleneck_edge_index[output_start_offset] = v_frontier_bottleneck_edge_index[v_frontier_parent_index];
                    v_frontier_all_bottleneck_edge_value[output_start_offset] = prev_edge_value;
                }
                v_frontier_all_neighbours[output_start_offset] = dst;
                v_frontier_all_parent_edge[output_start_offset] = parent_edge;
            }
        }
    }
}

struct is_positive_edge
{
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) > tol;
    }
};

struct is_neg_edge
{
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) < tol;
    }
};

struct discard_edge
{
    const bool* v_present;
    __host__ __device__ bool operator()(const thrust::tuple<int,int>& t)
    {
        const int i = thrust::get<0>(t);
        const int j = thrust::get<1>(t);
        return !(v_present[i] && v_present[j]);
    }
};

struct edge_in_frontier
{
    const int* v_frontier_indicator;
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float,int>& t)
    {
        const int i = thrust::get<0>(t);
        if(v_frontier_indicator[i] != -1)
            return true;
        return false;
    }
};

struct map_src_nodes
{
    const int* v_frontier_indicator;
    __host__ __device__ int operator()(const int& v)
    {
        return v_frontier_indicator[v];
    }
};

struct is_gr_zero
{
    __host__ __device__ int operator()(const bool& v)
    {
        return v > 0;
    }
};

// A should be directed thus containing same number of elements as in original problem.
conflicted_cycles::conflicted_cycles(const dCOO& A, const int _max_cycle_length)
{
    max_cycle_length = _max_cycle_length;
    init(A);
}

void conflicted_cycles::init(const dCOO& A)
{  
    // Partition edges into positive and negative.
    const thrust::device_vector<int> row_ids = A.get_row_ids();
    const thrust::device_vector<int> col_ids = A.get_col_ids();
    const thrust::device_vector<float> costs = A.get_data();

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin(), costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end(), costs.end()));

    thrust::device_vector<int> row_ids_pos(row_ids.size());
    thrust::device_vector<int> col_ids_pos(row_ids.size());
    thrust::device_vector<float> costs_pos(row_ids.size());
    auto first_pos = thrust::make_zip_iterator(thrust::make_tuple(row_ids_pos.begin(), col_ids_pos.begin(), costs_pos.begin()));
    auto last_pos = thrust::copy_if(first, last, first_pos, is_positive_edge());
    const size_t nr_positive_edges = std::distance(first_pos, last_pos);
    row_ids_pos.resize(nr_positive_edges);
    col_ids_pos.resize(nr_positive_edges);
    costs_pos.resize(nr_positive_edges);

    row_ids_rep = thrust::device_vector<int>(row_ids.size());
    col_ids_rep = thrust::device_vector<int>(row_ids.size());
    thrust::device_vector<float> costs_neg(row_ids.size());
    auto first_neg = thrust::make_zip_iterator(thrust::make_tuple(row_ids_rep.begin(), col_ids_rep.begin(), costs_neg.begin()));
    auto last_neg = thrust::copy_if(first, last, first_neg, is_neg_edge());
    const size_t nr_neg_edges = std::distance(first_neg, last_neg);
    row_ids_rep.resize(nr_neg_edges);
    col_ids_rep.resize(nr_neg_edges);

    // Create symmetric adjacency matrix of positive edges.
    if (nr_positive_edges > 0)
    {
        std::tie(row_ids_pos, col_ids_pos, costs_pos) = to_undirected(row_ids_pos, col_ids_pos, costs_pos);
        A_pos_symm = dCOO(A.max_dim(), A.max_dim(),
                    std::move(col_ids_pos), std::move(row_ids_pos), std::move(costs_pos), false);
        // TODO: Preprocess and remove edges from 'A_pos_symm' which are more than 'max_cycle_length' away from any repulsive edge?

        // remove those repulsive edges whose nodes do not exist in A_pos_symm
        thrust::device_vector<int> nodes_A_pos = A_pos_symm.get_row_ids();
        auto end = thrust::unique(nodes_A_pos.begin(), nodes_A_pos.end());
        nodes_A_pos.resize(std::distance(nodes_A_pos.begin(), end));
        thrust::device_vector<bool> v_in_pos(A.max_dim(), false);
        thrust::scatter(thrust::constant_iterator<bool>(true), thrust::constant_iterator<bool>(true) + nodes_A_pos.size(), nodes_A_pos.begin(), v_in_pos.begin());

        auto first_rep = thrust::make_zip_iterator(thrust::make_tuple(row_ids_rep.begin(), col_ids_rep.begin()));
        auto last_rep = thrust::make_zip_iterator(thrust::make_tuple(row_ids_rep.end(), col_ids_rep.end()));
        discard_edge func({thrust::raw_pointer_cast(v_in_pos.data())}); 

        auto last_rep_to_keep = thrust::remove_if(first_rep, last_rep, func);
        const int nr_rep_final = std::distance(first_rep, last_rep_to_keep);
        row_ids_rep.resize(nr_rep_final);
        col_ids_rep.resize(nr_rep_final);
    }
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> conflicted_cycles::filter_self_intersecting_paths(
    const thrust::device_vector<int>& new_src_offsets, const thrust::device_vector<int>& new_dst, const thrust::device_vector<int>& new_dst_parent,
    const thrust::device_vector<int>& new_dst_bottleneck_index, const thrust::device_vector<float>& new_dst_bottleneck_value,
    const thrust::device_vector<int>& seeds, const thrust::device_vector<int>& seeds_parents, 
    const device_vectors& expanded, const device_vectors& expanded_parent_edges)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

    thrust::device_vector<int> keep_mask(new_dst.size(), 1);
    thrust::device_vector<int> new_dst_sorted = new_dst;
    thrust::device_vector<int> new_dst_parent_sorted = new_dst_parent;
    thrust::device_vector<int> new_dst_sorting_order(new_dst.size());
    thrust::sequence(new_dst_sorting_order.begin(), new_dst_sorting_order.end());
    auto first_new = thrust::make_zip_iterator(thrust::make_tuple(new_dst_sorted.begin(), new_dst_parent_sorted.begin()));
    auto last_new = thrust::make_zip_iterator(thrust::make_tuple(new_dst_sorted.end(), new_dst_parent_sorted.end()));
    thrust::sort_by_key(first_new, last_new, new_dst_sorting_order.begin());

    thrust::device_vector<int> current_src_sorted, current_src_parents;
    for (int i = -2; i != expanded.size() + 1; ++i)
    {
        // i < 0 filters expansions to nodes of repulsive edges.
        if (i < 0)
        {
            thrust::device_vector<int> rep_nodes(2 * row_ids_rep.size());
            assert(thrust::is_sorted(row_ids_rep.begin(), row_ids_rep.end()));
            thrust::device_vector<int> col_ids_rep_sorted = col_ids_rep;
            thrust::sort(col_ids_rep_sorted.begin(), col_ids_rep_sorted.end());
            auto last_rep_node = thrust::set_union(row_ids_rep.begin(), row_ids_rep.end(), col_ids_rep_sorted.begin(), col_ids_rep_sorted.end(), rep_nodes.begin());
            auto last_unique = thrust::unique(rep_nodes.begin(), last_rep_node);
            rep_nodes.resize(std::distance(rep_nodes.begin(), last_unique));

            thrust::device_vector<int> new_dst_indices_to_remove(new_dst_sorting_order.size());
            auto last_to_remove = thrust::set_intersection_by_key(new_dst_sorted.begin(), new_dst_sorted.end(), rep_nodes.begin(), rep_nodes.end(), 
                                                                new_dst_sorting_order.begin(), thrust::make_discard_iterator(), new_dst_indices_to_remove.begin());
            const int num_to_remove = std::distance(new_dst_indices_to_remove.begin(), last_to_remove.second);
            new_dst_indices_to_remove.resize(num_to_remove);
            thrust::scatter(thrust::constant_iterator<int>(0), thrust::constant_iterator<int>(0) + num_to_remove, new_dst_indices_to_remove.begin(), keep_mask.begin());
        }
        else 
        {
            if (i == 0)
            {
                current_src_sorted = seeds;
                current_src_parents = seeds_parents;
            }
            else 
            {
                current_src_sorted = expanded[i - 1];
                current_src_parents = expanded_parent_edges[i - 1];
            }
            auto first_src = thrust::make_zip_iterator(thrust::make_tuple(current_src_sorted.begin(), current_src_parents.begin()));
            auto last_src = thrust::make_zip_iterator(thrust::make_tuple(current_src_sorted.end(), current_src_parents.end()));
            thrust::sort(first_src, last_src);

            thrust::device_vector<int> new_dst_indices_to_remove(new_dst_sorting_order.size());
            auto last_to_remove = thrust::set_intersection_by_key(first_new, last_new, first_src, last_src, new_dst_sorting_order.begin(), 
                                                                thrust::make_discard_iterator(), new_dst_indices_to_remove.begin());
            const int num_to_remove = std::distance(new_dst_indices_to_remove.begin(), last_to_remove.second);
            new_dst_indices_to_remove.resize(num_to_remove);
            thrust::scatter(thrust::constant_iterator<int>(0), thrust::constant_iterator<int>(0) + num_to_remove, new_dst_indices_to_remove.begin(), keep_mask.begin());
        }
    }
    // Compute segments of offsets e.g. [1, 3, 4, 7] -> [0, 1, 1, 2, 3, 3, 3]
    thrust::device_vector<int> new_src_degrees(new_src_offsets.size());
    thrust::adjacent_difference(new_src_offsets.begin(), new_src_offsets.end(), new_src_degrees.begin()); // degree for node n is at n + 1 location. 
    new_src_degrees = thrust::device_vector<int>(new_src_degrees.begin() + 1, new_src_degrees.end());

    thrust::device_vector<int> v_indices(new_src_degrees.size());
    thrust::sequence(v_indices.begin(), v_indices.end());

    const thrust::device_vector<int> segments = invert_unique(v_indices, new_src_degrees);

    thrust::reduce_by_key(segments.begin(), segments.end(), keep_mask.begin(), thrust::make_discard_iterator(), new_src_degrees.begin());
    thrust::device_vector<int> out_src_offsets(new_src_degrees.size() + 1);
    thrust::exclusive_scan(new_src_degrees.begin(), new_src_degrees.end(), out_src_offsets.begin());
    out_src_offsets[out_src_offsets.size() - 1] = out_src_offsets[out_src_offsets.size() - 2] + new_src_degrees[new_src_degrees.size() - 1];

    thrust::device_vector<int> out_dst(new_dst.size());
    auto out_dst_end = thrust::copy_if(new_dst.begin(), new_dst.end(), keep_mask.begin(), out_dst.begin(), is_gr_zero());
    out_dst.resize(std::distance(out_dst.begin(), out_dst_end));

    thrust::device_vector<int> out_dst_parent(new_dst_parent.size());
    auto out_dst_p_end = thrust::copy_if(new_dst_parent.begin(), new_dst_parent.end(), keep_mask.begin(), out_dst_parent.begin(), is_gr_zero());
    out_dst_parent.resize(std::distance(out_dst_parent.begin(), out_dst_p_end));

    return {out_src_offsets, out_dst, out_dst_parent};
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> 
    conflicted_cycles::expand_frontier(const thrust::device_vector<int>& v_parent_edges, const thrust::device_vector<int>& v_frontier,
        const thrust::device_vector<int>& v_bottleneck_edge_index, const thrust::device_vector<float>& v_bottleneck_edge_value, 
        const thrust::device_vector<int>& v_parent_nodes)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

    assert(v_parent_edges.size() == v_frontier.size());
    assert(v_bottleneck_edge_index.size() = v_frontier.size());
    assert(v_bottleneck_edge_value.size() = v_frontier.size());
    assert(v_parent_nodes.size() = v_frontier.size());

    assert(thrust::is_sorted(v_frontier.begin(), v_frontier.end())); // can contain duplicates.
    // 1. Find unique vertices and their counts:
    thrust::device_vector<int> v_frontier_unique, v_frontier_counts;

    std::tie(v_frontier_unique, v_frontier_counts) = get_unique_with_counts(v_frontier);

    // 2. Iterate over all edges and keep the ones whose first end-point is in the frontier. 
    assert(A_pos_symm.rows() == A_pos_symm.cols());
    assert(!A_pos_symm.is_directed());
    
    thrust::device_vector<int> v_frontier_indicator(A_pos_symm.rows(), -1);
    thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(v_frontier_unique.size()), 
                    v_frontier_unique.begin(), v_frontier_indicator.begin());
    const thrust::device_vector<int> row_ids = A_pos_symm.get_row_ids();
    const thrust::device_vector<int> col_ids = A_pos_symm.get_col_ids();
    const thrust::device_vector<float> pos_data = A_pos_symm.get_data();

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin(), pos_data.begin(), thrust::make_counting_iterator(0)));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end(), pos_data.end(), thrust::make_counting_iterator(0) + pos_data.size()));

    thrust::device_vector<int> expanded_src(A_pos_symm.nnz());
    thrust::device_vector<int> expanded_dst(A_pos_symm.nnz());
    thrust::device_vector<int> expanded_edge_index(A_pos_symm.nnz());
    thrust::device_vector<float> expanded_edge_value(A_pos_symm.nnz());

    auto first_output = thrust::make_zip_iterator(thrust::make_tuple(expanded_src.begin(), expanded_dst.begin(), expanded_edge_value.begin(), expanded_edge_index.begin()));

    edge_in_frontier expansion_func({thrust::raw_pointer_cast(v_frontier_indicator.data())}); 
    auto last_output = thrust::copy_if(first, last, first_output, expansion_func);
    const int num_expanded_edges = std::distance(first_output, last_output);
    if (num_expanded_edges == 0)
        return {thrust::device_vector<int>(), thrust::device_vector<int>(), thrust::device_vector<int>()};

    expanded_src.resize(num_expanded_edges);
    expanded_dst.resize(num_expanded_edges);
    expanded_edge_index.resize(num_expanded_edges);
    expanded_edge_value.resize(num_expanded_edges);

    // 3. Check how many neighbours of each unique frontier vertex are created:
    thrust::device_vector<int> expanded_src_unique, expanded_src_count;
    std::tie(expanded_src_unique, expanded_src_count) = get_unique_with_counts(expanded_src);

    assert(thrust::is_sorted(expanded_src_unique.begin(), expanded_src_unique.end()));
    assert(thrust::is_sorted(v_frontier_unique.begin(), v_frontier_unique.end()));
    assert(thrust::equal(expanded_src_unique.begin(), expanded_src_unique.end(), v_frontier_unique.begin()));

    // 4. Now revert step 1. and account for duplicates:
    thrust::device_vector<int> out_unique_src_offsets(v_frontier_counts.size() + 1); // is calculated w.r.t compressed unique vertices in frontier!
    thrust::transform(v_frontier_counts.begin(), v_frontier_counts.end(), expanded_src_count.begin(), out_unique_src_offsets.begin(), thrust::multiplies<int>());
    thrust::fill(out_unique_src_offsets.begin() + out_unique_src_offsets.size(), out_unique_src_offsets.end(), 0);
    thrust::exclusive_scan(out_unique_src_offsets.begin(), out_unique_src_offsets.end(), out_unique_src_offsets.begin());
    const int output_num = out_unique_src_offsets.back();
    thrust::device_vector<int> expanded_dst_all(output_num);
    thrust::device_vector<int> expanded_dst_parent_edges(output_num);
    thrust::device_vector<int> expanded_edge_index_all(output_num);
    thrust::device_vector<float> expanded_edge_value_all(output_num);

    // compress source vertex labels:
    map_src_nodes src_compress({thrust::raw_pointer_cast(v_frontier_indicator.data())}); 
    thrust::transform(expanded_src.begin(), expanded_src.end(), expanded_src.begin(), src_compress);
    thrust::device_vector<int> expanded_src_offsets(expanded_src_count.size() + 1);
    thrust::copy(expanded_src_count.begin(), expanded_src_count.end(), expanded_src_offsets.begin() + 1);
    expanded_src_offsets[0] = 0;
    thrust::inclusive_scan(expanded_src_offsets.begin(), expanded_src_offsets.end(), expanded_src_offsets.begin());

    thrust::device_vector<int> v_frontier_offsets(v_frontier_counts.size() + 1);
    thrust::exclusive_scan(v_frontier_counts.begin(), v_frontier_counts.end(), v_frontier_offsets.begin());
    thrust::fill(v_frontier_offsets.begin() + v_frontier_counts.size(), v_frontier_offsets.end(), v_frontier_counts[v_frontier_counts.size() - 1] + v_frontier_offsets[v_frontier_counts.size() - 1]);

    const int threadCount = 256;
    const int blockCount = ceil(num_expanded_edges / (float) threadCount);
    copy_neighbourhood<<<blockCount, threadCount>>>(num_expanded_edges, 
                        thrust::raw_pointer_cast(v_frontier_offsets.data()),
                        thrust::raw_pointer_cast(expanded_src.data()),
                        thrust::raw_pointer_cast(expanded_src_offsets.data()),
                        thrust::raw_pointer_cast(expanded_dst.data()),
                        thrust::raw_pointer_cast(expanded_edge_index.data()),
                        thrust::raw_pointer_cast(expanded_edge_value.data()),
                        thrust::raw_pointer_cast(v_parent_edges.data()),
                        thrust::raw_pointer_cast(v_bottleneck_edge_index.data()),
                        thrust::raw_pointer_cast(v_bottleneck_edge_value.data()),
                        thrust::raw_pointer_cast(v_parent_nodes.data()),
                        thrust::raw_pointer_cast(out_unique_src_offsets.data()),
                        thrust::raw_pointer_cast(expanded_dst_all.data()),
                        thrust::raw_pointer_cast(expanded_dst_parent_edges.data()),
                        thrust::raw_pointer_cast(expanded_edge_index_all.data()),
                        thrust::raw_pointer_cast(expanded_edge_value_all.data())
                    );

    
    return {out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges, expanded_edge_index_all, expanded_edge_value_all};
}

void conflicted_cycles::expand_merge_triangulate(device_vectors& dst_nodes, device_vectors& dst_nodes_parent_edges, device_vectors& src_offsets)
{
    return expand_merge_triangulate(dst_nodes.back(), dst_nodes_parent_edges.back(), dst_nodes, dst_nodes_parent_edges, src_offsets);
}

void conflicted_cycles::expand_merge_triangulate(const thrust::device_vector<int>& seeds, const thrust::device_vector<int>& edge_ids,
    device_vectors& dst_nodes, device_vectors& dst_nodes_parent_edges, device_vectors& src_offsets)
{
    {
        thrust::device_vector<int> seeds_sorted = seeds;
        thrust::device_vector<int> edge_ids_sorted = edge_ids;
        thrust::sort_by_key(seeds_sorted.begin(), seeds_sorted.end(), edge_ids_sorted.begin());
        thrust::device_vector<int> current_src_offsets, current_expanded_dst, current_expanded_dst_parent_edges;
        std::tie(current_src_offsets, current_expanded_dst, current_expanded_dst_parent_edges) = expand_frontier(edge_ids_sorted, seeds_sorted, dst_nodes, dst_nodes_parent_edges);
        src_offsets.push_back(std::move(current_src_offsets));
        dst_nodes.push_back(std::move(current_expanded_dst));
        dst_nodes_parent_edges.push_back(std::move(current_expanded_dst_parent_edges));
    }
    {
        thrust::device_vector<int> intersect_dst, intersect_parents;
        std::tie(intersect_dst, intersect_parents) = merge_paths();
        triangulate_intersection(intersect_dst, intersect_parents);
    }
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> conflicted_cycles::merge_paths() const
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    if (up_dst.size() == 0 || down_dst.size() == 0)
        return {thrust::device_vector<int>(), thrust::device_vector<int>()};

    assert(up_dst.size() == up_dst_parent_edges.size());
    assert(down_dst.size() == down_dst_parent_edges.size());
    // For both up and down paths, create a container with (vertex, parent edge id) and sort them.
    thrust::device_vector<int> up_final_dst = up_dst[up_dst.size() - 1];
    thrust::device_vector<int> up_final_parents = up_dst_parent_edges[up_dst_parent_edges.size() - 1];
    auto first_up = thrust::make_zip_iterator(thrust::make_tuple(up_final_dst.begin(), up_final_parents.begin()));
    auto last_up = thrust::make_zip_iterator(thrust::make_tuple(up_final_dst.end(), up_final_parents.end()));
    thrust::sort(first_up, last_up);

    thrust::device_vector<int> down_final_dst = down_dst[down_dst.size() - 1];
    thrust::device_vector<int> down_final_parents = down_dst_parent_edges[down_dst_parent_edges.size() - 1];
    auto first_down = thrust::make_zip_iterator(thrust::make_tuple(down_final_dst.begin(), down_final_parents.begin()));
    auto last_down = thrust::make_zip_iterator(thrust::make_tuple(down_final_dst.end(), down_final_parents.end()));
    thrust::sort(first_down, last_down);

    // now find intersecting vertices and their corresponding repulsive edges.
    thrust::device_vector<int> intersect_dst(max(up_final_dst.size(), down_final_dst.size()));
    thrust::device_vector<int> intersect_parents(max(up_final_dst.size(), down_final_dst.size()));
    auto first_intersect = thrust::make_zip_iterator(thrust::make_tuple(intersect_dst.begin(), intersect_parents.begin()));
    auto last_intersect = thrust::set_intersection(first_up, last_up, first_down, last_down, first_intersect);
    const int number_intersect = std::distance(first_intersect, last_intersect);

    intersect_dst.resize(number_intersect);
    intersect_parents.resize(number_intersect);

    return {intersect_dst, intersect_parents};
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> get_predecessors(
    const thrust::device_vector<int>& v_dst_select, const thrust::device_vector<int>& v_dst_select_parents,
    const thrust::device_vector<int>& v_source, const thrust::device_vector<int>& v_source_offsets, 
    const thrust::device_vector<int>& v_dst, const thrust::device_vector<int>& v_dst_parents)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    assert(v_dst_select.size() == v_dst_select_parents.size());
    assert(v_dst.size() == v_dst_parents.size());
    if (v_dst_select.size() == 0)
        return {thrust::device_vector<int>(), thrust::device_vector<int>()};

    // first get all predecessors (stored in v_source_expanded)
    thrust::device_vector<int> v_source_expanded;
    {
        thrust::device_vector<int> v_source_degrees(v_source_offsets.size());
        thrust::adjacent_difference(v_source_offsets.begin(), v_source_offsets.end(), v_source_degrees.begin()); // degree for node n is at n + 1 location. 
        v_source_degrees = thrust::device_vector<int>(v_source_degrees.begin() + 1, v_source_degrees.end());

        thrust::device_vector<int> v_source_sorted = v_source;
        thrust::sort(v_source_sorted.begin(), v_source_sorted.end());

        auto last_unique = thrust::unique(v_source_sorted.begin(), v_source_sorted.end());
        v_source_sorted.resize(std::distance(v_source_sorted.begin(), last_unique));
        v_source_expanded = invert_unique(v_source_sorted, v_source_degrees);
        assert(v_source_expanded.size() == v_dst.size());
    }
    // now sort the two sets (required for set_intersection_by_key)
    thrust::device_vector<int> v_dst_sorted = v_dst;
    thrust::device_vector<int> v_dst_parents_sorted = v_dst_parents;
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sorted.begin(), v_dst_parents_sorted.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sorted.end(), v_dst_parents_sorted.end()));
        thrust::sort_by_key(first, last, v_source_expanded.begin()); 
        assert(thrust::is_sorted(v_dst_sorted.begin(), v_dst_sorted.end()));
    }

    thrust::device_vector<int> v_dst_sel_sorted = v_dst_select;
    thrust::device_vector<int> v_dst_sel_parents_sorted = v_dst_select_parents;
    {
        auto first_sel = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sel_sorted.begin(), v_dst_sel_parents_sorted.begin()));
        auto last_sel = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sel_sorted.end(), v_dst_sel_parents_sorted.end()));
        thrust::sort(first_sel, last_sel); // required for set_intersection_by_key
        assert(thrust::is_sorted(v_dst_sel_sorted.begin(), v_dst_sel_sorted.end()));
    }
    
    // get only the predecessors of the given destination vertices (in v_dst_select)
    auto first = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sorted.begin(), v_dst_parents_sorted.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sorted.end(), v_dst_parents_sorted.end()));
    auto first_sel = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sel_sorted.begin(), v_dst_sel_parents_sorted.begin()));
    auto last_sel = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sel_sorted.end(), v_dst_sel_parents_sorted.end()));
    thrust::device_vector<int> v_source_sel(v_source_expanded.size());
    thrust::device_vector<int> v_source_sel_parents(v_source_expanded.size());
    auto first_intersect = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), v_source_sel_parents.begin()));
    auto last_intersect = thrust::set_intersection_by_key(first, last, first_sel, last_sel, v_source_expanded.begin(), first_intersect, v_source_sel.begin());
    const int num_source_selected = std::distance(v_source_sel.begin(), last_intersect.second);
    v_source_sel.resize(num_source_selected);
    v_source_sel_parents.resize(num_source_selected);
    assert(v_source_sel.size() == v_dst_select.size());

    return {v_source_sel, v_source_sel_parents};
}

void conflicted_cycles::add_to_triangulation(
    const thrust::device_vector<int>& v1, const thrust::device_vector<int>& v2, const thrust::device_vector<int>& v3)
{
    assert(v1.size() == v2.size());
    assert(v1.size() == v3.size());
    tri_v1 = concatenate(tri_v1, v1);
    tri_v2 = concatenate(tri_v2, v2);
    tri_v3 = concatenate(tri_v3, v3);
    // print_vector(tri_v1, "tri_v1");
    // print_vector(tri_v2, "tri_v2");
    // print_vector(tri_v3, "tri_v3");
}

void conflicted_cycles::triangulate_intersection(const thrust::device_vector<int>& intersected_vertices, const thrust::device_vector<int>& v_parent_edges)
{
    thrust::device_vector<int> v1_root(intersected_vertices.size());
    if (up_dst.size() == 0 || down_dst.size() == 0)
        return;

    if (up_dst.size() == 1 && down_dst.size() == 1)
    {
        thrust::device_vector<int> current_tri_v2(intersected_vertices.size());

        thrust::gather(v_parent_edges.begin(), v_parent_edges.end(), col_ids_rep.begin(), v1_root.begin());
        thrust::gather(v_parent_edges.begin(), v_parent_edges.end(), row_ids_rep.begin(), current_tri_v2.begin());
        add_to_triangulation(v1_root, current_tri_v2, intersected_vertices);
    }
    else
    {
        // triangulate above the intersection point
        thrust::device_vector<int> current_vertices = intersected_vertices;
        thrust::device_vector<int> current_parents = v_parent_edges;
        for (int index = up_dst.size() - 1; index >= 0; index--)
        {
            thrust::device_vector<int> pred_vertices, pred_parents, v_source;
            if (index > 0)
                std::tie(pred_vertices, pred_parents) = get_predecessors(current_vertices, current_parents, up_dst[index - 1], up_src_offsets[index], up_dst[index], up_dst_parent_edges[index]);
            else
            {
                pred_vertices = thrust::device_vector<int>(intersected_vertices.size());
                thrust::gather(v_parent_edges.begin(), v_parent_edges.end(), row_ids_rep.begin(), pred_vertices.begin());
            }

            add_to_triangulation(v1_root, current_vertices, pred_vertices);
            current_vertices = std::move(pred_vertices);
            current_parents = std::move(pred_parents);
        }
        // triangulate below the intersection point, one less triangle would be created because 'root' vertex is col_ids_rep
        current_vertices = intersected_vertices;
        current_parents = v_parent_edges;
        for (int index = down_dst.size() - 1; index > 0; index--)
        {
            thrust::device_vector<int> pred_vertices, pred_parents;
            std::tie(pred_vertices, pred_parents) = get_predecessors(current_vertices, current_parents, down_dst[index - 1], down_src_offsets[index], down_dst[index], down_dst_parent_edges[index]);
            add_to_triangulation(v1_root, current_vertices, pred_vertices);
            current_vertices = std::move(pred_vertices);
            current_parents = std::move(pred_parents);
        }
    }
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> conflicted_cycles::enumerate_conflicted_cycles()
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    if (max_cycle_length < 3)
        return {tri_v1, tri_v2, tri_v3};

    thrust::device_vector<int> edge_ids(row_ids_rep.size());
    thrust::sequence(edge_ids.begin(), edge_ids.end());
    std::cout<<"Found "<<edge_ids.size()<< " repulsive edges\n";

    // expand rows:
    expand_merge_triangulate(row_ids_rep, edge_ids, up_dst, up_dst_parent_edges, up_src_offsets);

    // expand cols:
    expand_merge_triangulate(col_ids_rep, edge_ids, down_dst, down_dst_parent_edges, down_src_offsets);

    std::cout<<"3-cycles: # triangles "<<tri_v1.size()<< "\n";

    const int num_itr = (int) ceil((max_cycle_length - 3) / 2.0);
    for (int c = 0; c < num_itr; c++)
    {
        // expand up and check for even valued cycles:
        expand_merge_triangulate(up_dst, up_dst_parent_edges, up_src_offsets);

        std::cout<<4 + 2 * c<<"-cycles: # triangles "<<tri_v1.size()<< "\n";

        if (c == num_itr - 1 && max_cycle_length % 2 == 0)
            break;

        // expand down and check for odd valued cycles:
        expand_merge_triangulate(down_dst, down_dst_parent_edges, down_src_offsets);

        std::cout<<5 + 2 * c<<"-cycles: # triangles "<<tri_v1.size()<< "\n";
    }
    return {tri_v1, tri_v2, tri_v3};
}