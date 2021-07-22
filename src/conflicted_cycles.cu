#include "conflicted_cycles.h"
#include "parallel_gaec_utils.h"

#define tol 1e-6

typedef std::vector<thrust::device_vector<int>> device_vectors;
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

struct vertex_parent_compare
{
    __host__ __device__ bool operator()(const thrust::tuple<int,int>& t1, const thrust::tuple<int,int>& t2)
    {
        const int v1 = thrust::get<0>(t1);
        const int v2 = thrust::get<0>(t2);
        if (v1 != v2)
            return v1 < v2;

        const int p1 = thrust::get<1>(t1);
        const int p2 = thrust::get<1>(t2);
        return p1 < p2;
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

std::tuple<dCOO, thrust::device_vector<int>, thrust::device_vector<int>> create_matrices(const dCOO& A)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    
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

    thrust::device_vector<int> row_ids_neg(row_ids.size());
    thrust::device_vector<int> col_ids_neg(row_ids.size());
    thrust::device_vector<float> costs_neg(row_ids.size());
    auto first_neg = thrust::make_zip_iterator(thrust::make_tuple(row_ids_neg.begin(), col_ids_neg.begin(), costs_neg.begin()));
    auto last_neg = thrust::copy_if(first, last, first_neg, is_neg_edge());
    const size_t nr_neg_edges = std::distance(first_neg, last_neg);
    row_ids_neg.resize(nr_neg_edges);
    col_ids_neg.resize(nr_neg_edges);

    // Create symmetric adjacency matrix of positive edges.
    dCOO A_pos_symm;
    if (nr_positive_edges > 0)
    {
        std::tie(row_ids_pos, col_ids_pos, costs_pos) = to_undirected(row_ids_pos, col_ids_pos, costs_pos);
        A_pos_symm = dCOO(std::max(A.rows(), A.cols()), std::max(A.rows(), A.cols()),
                    std::move(col_ids_pos), std::move(row_ids_pos), std::move(costs_pos));

        // remove those repulsive edges whose nodes do not exist in A_pos_symm
        thrust::device_vector<int> nodes_A_pos = A_pos_symm.get_row_ids();
        auto end = thrust::unique(nodes_A_pos.begin(), nodes_A_pos.end());
        nodes_A_pos.resize(std::distance(nodes_A_pos.begin(), end));
        thrust::device_vector<bool> v_in_pos(std::max(A.rows(), A.cols()), false);
        thrust::scatter(thrust::constant_iterator<bool>(true), thrust::constant_iterator<bool>(true) + nodes_A_pos.size(), nodes_A_pos.begin(), v_in_pos.begin());

        auto first_rep = thrust::make_zip_iterator(thrust::make_tuple(row_ids_neg.begin(), col_ids_neg.begin()));
        auto last_rep = thrust::make_zip_iterator(thrust::make_tuple(row_ids_neg.end(), col_ids_neg.end()));
        discard_edge func({thrust::raw_pointer_cast(v_in_pos.data())}); 

        auto last_rep_to_keep = thrust::remove_if(first_rep, last_rep, func);
        const int nr_rep_final = std::distance(first_rep, last_rep_to_keep);
        row_ids_neg.resize(nr_rep_final);
        col_ids_neg.resize(nr_rep_final);
    }
    return {A_pos_symm, row_ids_neg, col_ids_neg};
}

__global__ void copy_neighbourhood(const int num_edges,
                                const int* const __restrict__ v_frontier_offsets,
                                const int* const __restrict__ expanded_src_compressed,
                                const int* const __restrict__ expanded_src_offsets,
                                const int* const __restrict__ expanded_dst,
                                const int* const __restrict__ v_frontier_parent_edge,
                                const int* const __restrict__ v_frontier_all_src_offsets,
                                int* __restrict__ v_frontier_all_neighbours,
                                int* __restrict__ v_frontier_all_parent_edge)
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
        // Write same vertices with different parents:
        for (int v_frontier_parent_index = parent_start_offset; v_frontier_parent_index != parent_end_offset; ++output_start_offset, ++v_frontier_parent_index)
        {
            const int parent_edge = v_frontier_parent_edge[v_frontier_parent_index];
            v_frontier_all_neighbours[output_start_offset] = dst;
            v_frontier_all_parent_edge[output_start_offset] = parent_edge;
        }
    }
}

struct edge_in_frontier
{
    const int* v_frontier_indicator;
    __host__ __device__ bool operator()(const thrust::tuple<int,int>& t)
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


std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> filter_self_intersecting_paths(
    const thrust::device_vector<int>& new_src_offsets, const thrust::device_vector<int>& new_dst, const thrust::device_vector<int>& new_dst_parent,
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
    thrust::sort_by_key(first_new, last_new, new_dst_sorting_order.begin(), vertex_parent_compare());

    thrust::device_vector<int> current_src_sorted, current_src_parents;
    for (int i = 0; i < expanded.size() + 1; i++)
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
        thrust::sort(first_src, last_src, vertex_parent_compare());

        thrust::device_vector<int> new_dst_indices_to_remove(new_dst_sorting_order.size());
        auto last_to_remove = thrust::set_intersection_by_key(first_new, last_new, first_src, last_src, new_dst_sorting_order.begin(), 
                                                            thrust::make_discard_iterator(), new_dst_indices_to_remove.begin(), vertex_parent_compare());
        const int num_to_remove = std::distance(new_dst_indices_to_remove.begin(), last_to_remove.second);
        new_dst_indices_to_remove.resize(num_to_remove);
        thrust::scatter(thrust::constant_iterator<int>(0), thrust::constant_iterator<int>(0) + num_to_remove, new_dst_indices_to_remove.begin(), keep_mask.begin());
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
    expand_frontier(const thrust::device_vector<int>& v_parent_edges, const thrust::device_vector<int>& v_frontier, 
                    const dCOO& A_pos, const device_vectors& expanded_paths, const device_vectors& expanded_paths_parent_edges)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

    assert(v_parent_edges.size() == v_frontier.size());
    assert(thrust::is_sorted(v_frontier.begin(), v_frontier.end())); // can contain duplicates.
    // 1. Find unique vertices and their counts:
    thrust::device_vector<int> v_frontier_unique, v_frontier_counts;

    std::tie(v_frontier_unique, v_frontier_counts) = get_unique_with_counts(v_frontier);

    // 2. Iterate over all edges and keep the ones whose first end-point is in the frontier. 
    thrust::device_vector<int> v_frontier_indicator(std::max(A_pos.rows(), A_pos.cols()), -1);
    thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(v_frontier_unique.size()), 
                    v_frontier_unique.begin(), v_frontier_indicator.begin());
    const thrust::device_vector<int> row_ids = A_pos.get_row_ids();
    const thrust::device_vector<int> col_ids = A_pos.get_col_ids();
    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end()));
    thrust::device_vector<int> expanded_src(A_pos.nnz());
    thrust::device_vector<int> expanded_dst(A_pos.nnz());
    auto first_output = thrust::make_zip_iterator(thrust::make_tuple(expanded_src.begin(), expanded_dst.begin()));
    edge_in_frontier expansion_func({thrust::raw_pointer_cast(v_frontier_indicator.data())}); 
    auto last_output = thrust::copy_if(first, last, first_output, expansion_func);
    const int num_expanded_edges = std::distance(first_output, last_output);
    if (num_expanded_edges == 0)
        return {thrust::device_vector<int>(), thrust::device_vector<int>(), thrust::device_vector<int>()};

    expanded_src.resize(num_expanded_edges);
    expanded_dst.resize(num_expanded_edges);

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
                        thrust::raw_pointer_cast(v_parent_edges.data()),
                        thrust::raw_pointer_cast(out_unique_src_offsets.data()),
                        thrust::raw_pointer_cast(expanded_dst_all.data()),
                        thrust::raw_pointer_cast(expanded_dst_parent_edges.data()));

    std::tie(out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges) = filter_self_intersecting_paths(
        out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges, v_frontier, v_parent_edges, expanded_paths, expanded_paths_parent_edges);

    return {out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges};
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> merge_paths(
        const device_vectors& up_dst, const device_vectors& up_parents,
        const device_vectors& down_dst, const device_vectors& down_parents,
        const thrust::device_vector<int>& row_ids_rep, const thrust::device_vector<int>& col_ids_rep, 
        thrust::device_vector<int>& tri_v1, thrust::device_vector<int>& tri_v2, thrust::device_vector<int>& tri_v3)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

    assert(up_dst.size() == up_parents.size());
    assert(down_dst.size() == down_parents.size());
    // For both up and down paths, create a container with (vertex, parent edge id) and sort them.
    thrust::device_vector<int> up_final_dst = up_dst[up_dst.size() - 1];
    thrust::device_vector<int> up_final_parents = up_parents[up_parents.size() - 1];
    auto first_up = thrust::make_zip_iterator(thrust::make_tuple(up_final_dst.begin(), up_final_parents.begin()));
    auto last_up = thrust::make_zip_iterator(thrust::make_tuple(up_final_dst.end(), up_final_parents.end()));
    thrust::sort(first_up, last_up, vertex_parent_compare());

    thrust::device_vector<int> down_final_dst = down_dst[down_dst.size() - 1];
    thrust::device_vector<int> down_final_parents = down_parents[down_parents.size() - 1];
    auto first_down = thrust::make_zip_iterator(thrust::make_tuple(down_final_dst.begin(), down_final_parents.begin()));
    auto last_down = thrust::make_zip_iterator(thrust::make_tuple(down_final_dst.end(), down_final_parents.end()));
    thrust::sort(first_down, last_down, vertex_parent_compare());

    // now find intersecting vertices and their corresponding repulsive edges.
    thrust::device_vector<int> intersect_dst(max(up_final_dst.size(), down_final_dst.size()));
    thrust::device_vector<int> intersect_parents(max(up_final_dst.size(), down_final_dst.size()));
    auto first_intersect = thrust::make_zip_iterator(thrust::make_tuple(intersect_dst.begin(), intersect_parents.begin()));
    auto last_intersect = thrust::set_intersection(first_up, last_up, first_down, last_down, first_intersect, vertex_parent_compare());
    const int number_intersect = std::distance(first_intersect, last_intersect);

    intersect_dst.resize(number_intersect);
    intersect_parents.resize(number_intersect);

    thrust::device_vector<int> current_tri_v1(number_intersect);
    thrust::device_vector<int> current_tri_v2(number_intersect);

    thrust::gather(intersect_parents.begin(), intersect_parents.end(), row_ids_rep.begin(), current_tri_v1.begin());
    thrust::gather(intersect_parents.begin(), intersect_parents.end(), col_ids_rep.begin(), current_tri_v2.begin());

    tri_v1 = concatenate(tri_v1, current_tri_v1);
    tri_v2 = concatenate(tri_v2, current_tri_v2);
    tri_v3 = concatenate(tri_v3, intersect_dst);

    return {intersect_dst, intersect_parents};
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> triangulate_predecessors(const int index,
    const thrust::device_vector<int>& v_dst_select, const thrust::device_vector<int>& v_dst_select_parents,
    const device_vectors& offsets, const device_vectors& dst, const device_vectors& parents,
    const thrust::device_vector<int>& row_ids_rep, const thrust::device_vector<int>& col_ids_rep,
    thrust::device_vector<int>& tri_v1, thrust::device_vector<int>& tri_v2, thrust::device_vector<int>& tri_v3)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    if (v_dst_select.size() == 0)
        return {thrust::device_vector<int>(), thrust::device_vector<int>()};

    // first get all predecessors (stored in v_source_expanded)
    thrust::device_vector<int> v_source_degrees(offsets[index].size());
    thrust::adjacent_difference(offsets[index].begin(), offsets[index].end(), v_source_degrees.begin()); // degree for node n is at n + 1 location. 
    v_source_degrees = thrust::device_vector<int>(v_source_degrees.begin() + 1, v_source_degrees.end());

    thrust::device_vector<int> v_source = dst[index - 1];
    thrust::sort(v_source.begin(), v_source.end());

    auto last_unique = thrust::unique(v_source.begin(), v_source.end());
    v_source.resize(std::distance(v_source.begin(), last_unique));
    thrust::device_vector<int> v_source_expanded = invert_unique(v_source, v_source_degrees);
    assert(v_source_expanded.size() == dst[index].size());
    
    thrust::device_vector<int> v_dst_sorted = dst[index];
    thrust::device_vector<int> v_dst_parents_sorted = parents[index];
    auto first = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sorted.begin(), v_dst_parents_sorted.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sorted.end(), v_dst_parents_sorted.end()));
    thrust::sort_by_key(first, last, v_source_expanded.begin(), vertex_parent_compare()); // required for set_intersection_by_key

    thrust::device_vector<int> v_dst_sel_sorted = v_dst_select;
    thrust::device_vector<int> v_dst_sel_parents_sorted = v_dst_select_parents;
    auto first_sel = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sel_sorted.begin(), v_dst_sel_parents_sorted.begin()));
    auto last_sel = thrust::make_zip_iterator(thrust::make_tuple(v_dst_sel_sorted.end(), v_dst_sel_parents_sorted.end()));
    thrust::sort(first_sel, last_sel, vertex_parent_compare()); // required for set_intersection_by_key

    // get only the predecessors of the given destination vertices (in v_dst_select)
    thrust::device_vector<int> v_source_sel(v_source_expanded.size());
    thrust::device_vector<int> v_source_sel_parents(v_source_expanded.size());
    auto first_intersect = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), v_source_sel_parents.begin()));
    auto last_intersect = thrust::set_intersection_by_key(first, last, first_sel, last_sel, v_source_expanded.begin(), first_intersect, v_source_sel.begin(), vertex_parent_compare());
    const int num_source_selected = std::distance(first_intersect, last_intersect.first);
    v_source_sel.resize(num_source_selected);
    v_source_sel_parents.resize(num_source_selected);

    thrust::device_vector<int> current_tri_v1(num_source_selected);
    thrust::device_vector<int> current_tri_v2(num_source_selected);

    thrust::gather(v_source_sel_parents.begin(), v_source_sel_parents.end(), row_ids_rep.begin(), current_tri_v1.begin());
    thrust::gather(v_source_sel_parents.begin(), v_source_sel_parents.end(), col_ids_rep.begin(), current_tri_v2.begin());

    tri_v1 = concatenate(tri_v1, current_tri_v1);
    tri_v2 = concatenate(tri_v2, current_tri_v2);
    tri_v3 = concatenate(tri_v3, v_source_sel);

    return {v_source_sel, v_source_sel_parents};
}

// A should be directed thus containing same number of elements as in original problem.
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> enumerate_conflicted_cycles(const dCOO& A, const int max_cycle_length)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    // Initialize:
    thrust::device_vector<int> tri_v1, tri_v2, tri_v3;
    if (max_cycle_length < 3)
        return {tri_v1, tri_v2, tri_v3};

    dCOO A_pos_symm;
    thrust::device_vector<int> row_ids_rep, col_ids_rep;
    std::tie(A_pos_symm, row_ids_rep, col_ids_rep) = create_matrices(A); // TODO: Preprocess and remove edges from 'A_pos_symm' which are more than 'max_cycle_length' away from any repulsive edge?

    thrust::device_vector<int> edge_ids(row_ids_rep.size());
    thrust::sequence(edge_ids.begin(), edge_ids.end());

    device_vectors up_src_offsets, down_src_offsets;
    device_vectors up_expanded, down_expanded;
    device_vectors up_expanded_parent_edges, down_expanded_parent_edges;
    std::cout<<"Found "<<edge_ids.size()<< " repulsive edges\n";

    // expand rows:
    thrust::device_vector<int> up_seeds = row_ids_rep;
    thrust::device_vector<int> edge_ids_rows = edge_ids;
    thrust::sort_by_key(up_seeds.begin(), up_seeds.end(), edge_ids_rows.begin());
    thrust::device_vector<int> out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges;
    std::tie(out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges) = expand_frontier(edge_ids_rows, up_seeds, A_pos_symm, up_expanded, up_expanded_parent_edges);
    up_src_offsets.push_back(out_unique_src_offsets);
    up_expanded.push_back(expanded_dst_all);
    up_expanded_parent_edges.push_back(expanded_dst_parent_edges);

    // expand cols:
    thrust::device_vector<int> down_seeds = col_ids_rep;
    thrust::device_vector<int> edge_ids_cols = edge_ids;
    thrust::sort_by_key(down_seeds.begin(), down_seeds.end(), edge_ids_cols.begin());
    std::tie(out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges) = expand_frontier(edge_ids_cols, down_seeds, A_pos_symm, down_expanded, down_expanded_parent_edges);
    down_src_offsets.push_back(out_unique_src_offsets);
    down_expanded.push_back(expanded_dst_all);
    down_expanded_parent_edges.push_back(expanded_dst_parent_edges);

    thrust::device_vector<int> intersect_dst, intersect_parents;
    std::tie(intersect_dst, intersect_parents) = merge_paths(up_expanded, up_expanded_parent_edges,
                                                            down_expanded, down_expanded_parent_edges,
                                                            row_ids_rep, col_ids_rep, tri_v1, tri_v2, tri_v3);

    std::cout<<"3-cycles: # triangles "<<tri_v1.size()<< "\n";

    const int num_itr = (int) ceil((max_cycle_length - 3) / 2.0);
    for (int c = 0; c < num_itr; c++)
    {
        // expand up:
        thrust::device_vector<int> up_seeds = thrust::device_vector<int>(up_expanded.back().begin(), up_expanded.back().end());
        thrust::device_vector<int> edge_ids_up = thrust::device_vector<int>(up_expanded_parent_edges.back().begin(), up_expanded_parent_edges.back().end());
        thrust::sort_by_key(up_seeds.begin(), up_seeds.end(), edge_ids_up.begin());
        std::tie(out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges) = expand_frontier(edge_ids_up, up_seeds, A_pos_symm, up_expanded, up_expanded_parent_edges);
        up_src_offsets.push_back(out_unique_src_offsets);
        up_expanded.push_back(expanded_dst_all);
        up_expanded_parent_edges.push_back(expanded_dst_parent_edges);
        
        // even valued cycles:
        std::tie(intersect_dst, intersect_parents) = merge_paths(up_expanded, up_expanded_parent_edges,
                                                                down_expanded, down_expanded_parent_edges,
                                                                row_ids_rep, col_ids_rep, tri_v1, tri_v2, tri_v3);
        
        thrust::device_vector<int> current_int_dst = intersect_dst;
        thrust::device_vector<int> current_int_parents = intersect_parents;
        for (int v = up_expanded.size() - 1; v > 0; v--)
        {
            std::tie(current_int_dst, current_int_parents) = triangulate_predecessors(v, current_int_dst, current_int_parents, 
                                                                                    up_src_offsets, up_expanded, up_expanded_parent_edges,
                                                                                    row_ids_rep, col_ids_rep, tri_v1, tri_v2, tri_v3);
        }

        current_int_dst = intersect_dst;
        current_int_parents = intersect_parents;
        for (int v = down_expanded.size() - 1; v > 0; v--)
        {
            std::tie(current_int_dst, current_int_parents) = triangulate_predecessors(v, current_int_dst, current_int_parents, 
                                                                                    down_src_offsets, down_expanded, down_expanded_parent_edges,
                                                                                    row_ids_rep, col_ids_rep, tri_v1, tri_v2, tri_v3);
        }
        std::cout<<4 + 2 * c<<"-cycles: # triangles "<<tri_v1.size()<< "\n";

        if (c == num_itr - 1 && max_cycle_length % 2 == 0)
            break;

        // expand down:
        thrust::device_vector<int> down_seeds = thrust::device_vector<int>(down_expanded.back().begin(), down_expanded.back().end());
        thrust::device_vector<int> edge_ids_down = thrust::device_vector<int>(down_expanded_parent_edges.back().begin(), down_expanded_parent_edges.back().end());
        thrust::sort_by_key(down_seeds.begin(), down_seeds.end(), edge_ids_down.begin());
        std::tie(out_unique_src_offsets, expanded_dst_all, expanded_dst_parent_edges) = expand_frontier(edge_ids_down, down_seeds, A_pos_symm, down_expanded, down_expanded_parent_edges);
        down_src_offsets.push_back(out_unique_src_offsets);
        down_expanded.push_back(expanded_dst_all);
        down_expanded_parent_edges.push_back(expanded_dst_parent_edges);

        // odd valued cycles:
        std::tie(intersect_dst, intersect_parents) = merge_paths(up_expanded, up_expanded_parent_edges,
                                                                down_expanded, down_expanded_parent_edges,
                                                                row_ids_rep, col_ids_rep, tri_v1, tri_v2, tri_v3);
        current_int_dst = intersect_dst;
        current_int_parents = intersect_parents;
        for (int v = up_expanded.size() - 1; v > 0; v--)
            std::tie(current_int_dst, current_int_parents) = triangulate_predecessors(v, current_int_dst, current_int_parents, 
                                                                                    up_src_offsets, up_expanded, up_expanded_parent_edges,
                                                                                    row_ids_rep, col_ids_rep, tri_v1, tri_v2, tri_v3);

        current_int_dst = intersect_dst;
        current_int_parents = intersect_parents;
        for (int v = down_expanded.size() - 1; v > 0; v--)
            std::tie(current_int_dst, current_int_parents) = triangulate_predecessors(v, current_int_dst, current_int_parents, 
                                                                            down_src_offsets, down_expanded, down_expanded_parent_edges,
                                                                            row_ids_rep, col_ids_rep, tri_v1, tri_v2, tri_v3);

        std::cout<<5 + 2 * c<<"-cycles: # triangles "<<tri_v1.size()<< "\n";
    }
    return {tri_v1, tri_v2, tri_v3};
}