#pragma once

#include <tuple>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/equal.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#ifdef __CUDACC__
#define MST_HOST_DEVICE __host__ __device__
#else
#define MST_HOST_DEVICE
#endif

namespace MST_boruvka {

// Reduction functor: selects tuple with minimum weight, breaking ties by vertex id.
struct binop_tuple_minimum {
    typedef thrust::tuple<float, int, int> T; // (weight, destination, edge_id)
    MST_HOST_DEVICE
    T operator()(const T& a, const T& b) const {
        return (thrust::get<0>(a) == thrust::get<0>(b))
            ? ((thrust::get<1>(a) < thrust::get<1>(b)) ? a : b)
            : ((thrust::get<0>(a) < thrust::get<0>(b)) ? a : b);
    }
};

namespace detail {

// Iterative pointer-doubling path compression.
// After completion, succ[i] == root of component containing i.
template<template<typename> class VectorType>
void path_compress(VectorType<int>& succ, const int n)
{
    VectorType<int> succ_temp(n);
    for (int iter = 0; iter < 32; ++iter) {
        // succ_temp[i] = succ[succ[i]]
        thrust::gather(succ.begin(), succ.begin() + n, succ.begin(), succ_temp.begin());
        if (thrust::equal(succ.begin(), succ.begin() + n, succ_temp.begin()))
            break;
        succ.swap(succ_temp);
    }
}

} // namespace detail

// Boruvka's algorithm for maximum spanning tree.
// Input: symmetric edges (both directions: i->j and j->i with same cost).
// Output: single-direction MST edges (one per undirected MST edge).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<float>>
maximum_spanning_tree(
    const VectorType<int>& tails,
    const VectorType<int>& heads,
    const VectorType<float>& costs)
{
    const int n_directed = tails.size();
    assert(n_directed == (int)heads.size());
    assert(n_directed == (int)costs.size());

    if (n_directed == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<float>()};

    // Input is already symmetric (both directions present).
    // Assign edge IDs: for each directed edge, the ID is its index in
    // the canonical direction (tail < head). Both directions of the same
    // undirected edge get the same ID so dedup works at the end.
    VectorType<int> u(tails.begin(), tails.end());
    VectorType<int> v(heads.begin(), heads.end());
    VectorType<float> w(n_directed);
    VectorType<int> id(n_directed);

    // Negate costs for minimum spanning tree on negated weights = maximum spanning tree.
    thrust::transform(costs.begin(), costs.end(), w.begin(), thrust::negate<float>());

    // Assign canonical edge IDs: for edge (a,b), ID = index of the direction where a < b.
    // We build a lookup: sort edges by (min,max) to pair directions, then assign.
    // Simple approach: ID = own index. Dedup at end handles both directions.
    thrust::sequence(id.begin(), id.end());

    const int n_vertices_init = thrust::max(*thrust::max_element(u.begin(), u.end()) + 1, *thrust::max_element(v.begin(), v.end()) + 1);
    int n_vertices = n_vertices_init;
    int n_edges = n_directed;

    // Temp buffers
    VectorType<int> u_tmp(n_edges), v_tmp(n_edges), id_tmp(n_edges);
    VectorType<float> w_tmp(n_edges);

    VectorType<int> succ(n_vertices);
    VectorType<int> succ_id(n_vertices);
    VectorType<int> succ_input(n_vertices);  // input buffer for remove_circles
    VectorType<int> succ_temp(n_vertices);
    VectorType<int> indices(n_edges);
    VectorType<int> flags(n_edges);

    // Collected MST edge ids (at most n_vertices - 1 undirected MST edges,
    // but may collect both directions before dedup)
    VectorType<int> mst_edge_ids(n_directed);
    int n_mst = 0;

    // Reduce-by-key output buffers
    VectorType<int> rbk_keys(n_vertices);
    VectorType<float> rbk_w(n_vertices);
    VectorType<int> rbk_v(n_vertices);
    VectorType<int> rbk_id(n_vertices);

    int mst_iteration = 0;
    while (true) {
        if (n_edges == 0)
            break;

        if (n_edges == 1) {
            // Single remaining edge is always part of MST
            VectorType<int> single_id(1);
            thrust::copy(id.begin(), id.begin() + 1, single_id.begin());
            int eid = single_id[0];
            mst_edge_ids[n_mst++] = eid;
            break;
        }

        // --- Bounds checks before Step 1 ---
        assert(n_edges <= (int)u.size() && "u too small for n_edges");
        assert(n_edges <= (int)v.size() && "v too small for n_edges");
        assert(n_edges <= (int)w.size() && "w too small for n_edges");
        assert(n_edges <= (int)id.size() && "id too small for n_edges");
        assert(n_edges <= (int)indices.size() && "indices too small for n_edges");
        assert(n_edges <= (int)v_tmp.size() && "v_tmp too small for n_edges");
        assert(n_edges <= (int)w_tmp.size() && "w_tmp too small for n_edges");
        assert(n_edges <= (int)id_tmp.size() && "id_tmp too small for n_edges");

        // Step 1: Sort edges by source vertex
        thrust::sequence(indices.begin(), indices.begin() + n_edges);
        thrust::sort_by_key(u.begin(), u.begin() + n_edges, indices.begin());

        // Check: all u values should be in [0, n_vertices)
        {
            int u_max = *thrust::max_element(u.begin(), u.begin() + n_edges);
            int u_min = *thrust::min_element(u.begin(), u.begin() + n_edges);
            assert(u_min >= 0 && "u contains negative vertex id");
            assert(u_max < n_vertices && "u contains vertex id >= n_vertices");
        }

        // Check: indices are valid gather maps into v, w, id
        {
            int idx_max = *thrust::max_element(indices.begin(), indices.begin() + n_edges);
            assert(idx_max < n_edges && "sort indices out of range");
        }

        // Reorder v, w, id according to sort
        thrust::gather(indices.begin(), indices.begin() + n_edges, v.begin(), v_tmp.begin());
        thrust::gather(indices.begin(), indices.begin() + n_edges, w.begin(), w_tmp.begin());
        thrust::gather(indices.begin(), indices.begin() + n_edges, id.begin(), id_tmp.begin());
        v.swap(v_tmp);
        w.swap(w_tmp);
        id.swap(id_tmp);

        // Check: v values (destinations) in [0, n_vertices)
        {
            int v_max = *thrust::max_element(v.begin(), v.begin() + n_edges);
            int v_min = *thrust::min_element(v.begin(), v.begin() + n_edges);
            assert(v_min >= 0 && "v contains negative vertex id");
            assert(v_max < n_vertices && "v contains vertex id >= n_vertices");
        }

        // --- Bounds checks before reduce_by_key ---
        assert(n_vertices <= (int)rbk_keys.size() && "rbk_keys too small");
        assert(n_vertices <= (int)rbk_w.size() && "rbk_w too small");
        assert(n_vertices <= (int)rbk_v.size() && "rbk_v too small");
        assert(n_vertices <= (int)rbk_id.size() && "rbk_id too small");

        fprintf(stderr, "MST iter %d: n_edges=%d, n_vertices=%d, "
            "u.size=%d, v.size=%d, w.size=%d, id.size=%d, "
            "rbk_keys.size=%d\n",
            mst_iteration, n_edges, n_vertices,
            (int)u.size(), (int)v.size(), (int)w.size(), (int)id.size(),
            (int)rbk_keys.size());

        // Step 2: Find minimum-weight edge per source vertex
        auto new_last = thrust::reduce_by_key(
            u.begin(), u.begin() + n_edges,
            thrust::make_zip_iterator(thrust::make_tuple(
                w.begin(), v.begin(), id.begin())),
            rbk_keys.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(
                rbk_w.begin(), rbk_v.begin(), rbk_id.begin())),
            thrust::equal_to<int>(),
            binop_tuple_minimum());

        int n_min_edges = new_last.first - rbk_keys.begin();
        assert(n_min_edges >= 0 && n_min_edges <= n_vertices && "reduce_by_key returned invalid count");

        // Step 3: Build successor pointers
        // succ_input[vertex] = destination of min-weight edge from vertex
        // succ_id[vertex] = original edge id of that edge
        thrust::sequence(succ_input.begin(), succ_input.begin() + n_vertices);
        // Initialize succ_id to -1
        thrust::fill(succ_id.begin(), succ_id.begin() + n_vertices, -1);

        // Check: rbk_keys (scatter map) values in [0, n_vertices)
        {
            int key_max = *thrust::max_element(rbk_keys.begin(), rbk_keys.begin() + n_min_edges);
            int key_min = *thrust::min_element(rbk_keys.begin(), rbk_keys.begin() + n_min_edges);
            assert(key_min >= 0 && "rbk_keys contains negative key");
            assert(key_max < n_vertices && "rbk_keys scatter target >= n_vertices");
        }
        // Check: rbk_v (successor destinations) in [0, n_vertices)
        {
            int rv_max = *thrust::max_element(rbk_v.begin(), rbk_v.begin() + n_min_edges);
            int rv_min = *thrust::min_element(rbk_v.begin(), rbk_v.begin() + n_min_edges);
            assert(rv_min >= 0 && "rbk_v contains negative vertex");
            assert(rv_max < n_vertices && "rbk_v contains vertex >= n_vertices");
        }

        thrust::scatter(
            thrust::make_zip_iterator(thrust::make_tuple(
                rbk_v.begin(), rbk_id.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                rbk_v.begin() + n_min_edges, rbk_id.begin() + n_min_edges)),
            rbk_keys.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(
                succ_input.begin(), succ_id.begin())));

        // Step 4: Break 2-cycles
        // If vertex i points to j and j points to i, break by keeping only
        // the edge from the smaller vertex.

        // Check: succ_input values (used as indices into succ_input) in [0, n_vertices)
        {
            int si_max = *thrust::max_element(succ_input.begin(), succ_input.begin() + n_vertices);
            int si_min = *thrust::min_element(succ_input.begin(), succ_input.begin() + n_vertices);
            assert(si_min >= 0 && "succ_input contains negative value before step 4");
            assert(si_max < n_vertices && "succ_input contains value >= n_vertices before step 4");
        }
        assert(n_vertices <= (int)succ.size() && "succ too small for n_vertices");
        assert(n_vertices <= (int)succ_temp.size() && "succ_temp too small for n_vertices");

        {
            const int n = n_vertices;
            const int* si_ptr = thrust::raw_pointer_cast(succ_input.data());
            int* succ_ptr = thrust::raw_pointer_cast(succ.data());
            int* aux_ptr = thrust::raw_pointer_cast(succ_temp.data());

            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(n),
                [si_ptr, succ_ptr, aux_ptr] MST_HOST_DEVICE (int pos) {
                    int successor = si_ptr[pos];
                    int s_successor = si_ptr[successor];
                    successor = ((successor > pos) && (s_successor == pos)) ? pos : successor;
                    aux_ptr[pos] = (successor != pos) ? 1 : 0;
                    succ_ptr[pos] = successor;
                });
        }

        // Step 5: Collect MST edges
        // aux[i] == 1 means vertex i has a new MST edge
        thrust::exclusive_scan(succ_temp.begin(), succ_temp.begin() + n_vertices,
            succ_input.begin()); // reuse succ_input as scan output

        // Check: scatter targets (succ_input + n_mst) must fit in mst_edge_ids
        {
            int scan_max = *thrust::max_element(succ_input.begin(), succ_input.begin() + n_vertices);
            assert(n_mst + scan_max < (int)mst_edge_ids.size() &&
                "mst_edge_ids scatter would be out of bounds");
        }

        thrust::scatter_if(
            succ_id.begin(), succ_id.begin() + n_vertices,
            succ_input.begin(),
            succ_temp.begin(),
            mst_edge_ids.begin() + n_mst);

        {
            VectorType<int> last_scan(1), last_flag(1);
            thrust::copy(succ_input.begin() + n_vertices - 1, succ_input.begin() + n_vertices, last_scan.begin());
            thrust::copy(succ_temp.begin() + n_vertices - 1, succ_temp.begin() + n_vertices, last_flag.begin());
            n_mst += last_scan[0] + last_flag[0];
        }
        assert(n_mst <= n_directed && "collected more MST edges than input edges");

        // Step 6: Path compression to find connected components
        // succ[i] is the representative of i's component
        thrust::sequence(succ_input.begin(), succ_input.begin() + n_vertices);
        detail::path_compress<VectorType>(succ, n_vertices);

        // Step 7: Assign new contiguous component IDs
        // Sort vertices by component representative, then assign dense IDs
        thrust::sequence(succ_input.begin(), succ_input.begin() + n_vertices);
        thrust::sort_by_key(succ.begin(), succ.begin() + n_vertices, succ_input.begin());

        // Mark segment boundaries
        {
            const int n = n_vertices;
            const int* sorted_ptr = thrust::raw_pointer_cast(succ.data());
            int* marks_ptr = thrust::raw_pointer_cast(succ_temp.data());

            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(n),
                [sorted_ptr, marks_ptr, n] MST_HOST_DEVICE (int pos) {
                    marks_ptr[pos] = ((pos == n - 1) || (sorted_ptr[pos] != sorted_ptr[pos + 1])) ? 1 : 0;
                });
        }

        // new_vertices maps old vertex -> new component id
        VectorType<int>& new_vertices = succ; // reuse succ buffer
        thrust::exclusive_scan(succ_temp.begin(), succ_temp.begin() + n_vertices,
            succ_id.begin()); // reuse succ_id as scan output
        thrust::scatter(succ_id.begin(), succ_id.begin() + n_vertices,
            succ_input.begin(), new_vertices.begin());

        int new_n_vertices;
        {
            VectorType<int> last_scan(1), last_flag(1);
            thrust::copy(succ_id.begin() + n_vertices - 1, succ_id.begin() + n_vertices, last_scan.begin());
            thrust::copy(succ_temp.begin() + n_vertices - 1, succ_temp.begin() + n_vertices, last_flag.begin());
            new_n_vertices = last_scan[0] + last_flag[0];
        }

        // Step 8: Filter inter-component edges
        // Check: u and v values are valid indices into new_vertices
        {
            int u_max = *thrust::max_element(u.begin(), u.begin() + n_edges);
            int v_max = *thrust::max_element(v.begin(), v.begin() + n_edges);
            assert(u_max < (int)new_vertices.size() && "u value >= new_vertices.size() in step 8");
            assert(v_max < (int)new_vertices.size() && "v value >= new_vertices.size() in step 8");
        }

        {
            const int* nv_ptr = thrust::raw_pointer_cast(new_vertices.data());
            const int* u_ptr = thrust::raw_pointer_cast(u.data());
            const int* v_ptr_data = thrust::raw_pointer_cast(v.data());
            int* flags_ptr = thrust::raw_pointer_cast(flags.data());
            const int ne = n_edges;

            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(ne),
                [nv_ptr, u_ptr, v_ptr_data, flags_ptr] MST_HOST_DEVICE (int pos) {
                    flags_ptr[pos] = (nv_ptr[u_ptr[pos]] != nv_ptr[v_ptr_data[pos]]) ? 1 : 0;
                });
        }

        thrust::exclusive_scan(flags.begin(), flags.begin() + n_edges, indices.begin());

        int new_n_edges;
        {
            VectorType<int> last_scan(1), last_flag(1);
            thrust::copy(indices.begin() + n_edges - 1, indices.begin() + n_edges, last_scan.begin());
            thrust::copy(flags.begin() + n_edges - 1, flags.begin() + n_edges, last_flag.begin());
            new_n_edges = last_scan[0] + last_flag[0];
        }

        if (new_n_edges == 0)
            break;

        // Compact edges
        thrust::scatter_if(
            thrust::make_zip_iterator(thrust::make_tuple(
                u.begin(), v.begin(), w.begin(), id.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                u.begin() + n_edges, v.begin() + n_edges, w.begin() + n_edges, id.begin() + n_edges)),
            indices.begin(), flags.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(
                u_tmp.begin(), v_tmp.begin(), w_tmp.begin(), id_tmp.begin())));

        // Step 9: Relabel endpoints with new component IDs
        // Check: u_tmp and v_tmp values are valid indices into new_vertices
        {
            int ut_max = *thrust::max_element(u_tmp.begin(), u_tmp.begin() + new_n_edges);
            int vt_max = *thrust::max_element(v_tmp.begin(), v_tmp.begin() + new_n_edges);
            assert(ut_max < (int)new_vertices.size() && "u_tmp value >= new_vertices.size() in step 9");
            assert(vt_max < (int)new_vertices.size() && "v_tmp value >= new_vertices.size() in step 9");
        }
        {
            thrust::gather(u_tmp.begin(), u_tmp.begin() + new_n_edges, new_vertices.begin(), u_tmp.begin());
            thrust::gather(v_tmp.begin(), v_tmp.begin() + new_n_edges, new_vertices.begin(), v_tmp.begin());
        }

        // Step 10: Swap buffers and repeat
        u.swap(u_tmp);
        v.swap(v_tmp);
        w.swap(w_tmp);
        id.swap(id_tmp);

        n_vertices = new_n_vertices;
        n_edges = new_n_edges;

        // Resize temp buffers for potentially smaller problem
        if ((int)succ.size() < n_vertices) {
            succ.resize(n_vertices);
            succ_id.resize(n_vertices);
            succ_input.resize(n_vertices);
            succ_temp.resize(n_vertices);
            rbk_keys.resize(n_vertices);
            rbk_w.resize(n_vertices);
            rbk_v.resize(n_vertices);
            rbk_id.resize(n_vertices);
        }

        // Check: after relabeling, u and v should be in [0, new_n_vertices)
        {
            int u_max = *thrust::max_element(u.begin(), u.begin() + n_edges);
            int v_max = *thrust::max_element(v.begin(), v.begin() + n_edges);
            assert(u_max < n_vertices && "relabeled u >= new n_vertices at end of iteration");
            assert(v_max < n_vertices && "relabeled v >= new n_vertices at end of iteration");
        }

        mst_iteration++;
    }

    // Gather MST edges from original input
    // Check: mst_edge_ids are valid indices into tails/heads/costs
    if (n_mst > 0) {
        int eid_max = *thrust::max_element(mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst);
        int eid_min = *thrust::min_element(mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst);
        assert(eid_min >= 0 && "mst_edge_ids contains negative id");
        assert(eid_max < n_directed && "mst_edge_ids contains id >= n_directed");
    }

    VectorType<int> mst_tails(n_mst), mst_heads(n_mst);
    VectorType<float> mst_costs(n_mst);
    thrust::gather(mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst,
        tails.begin(), mst_tails.begin());
    thrust::gather(mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst,
        heads.begin(), mst_heads.begin());
    thrust::gather(mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst,
        costs.begin(), mst_costs.begin());

    // Normalize to canonical direction (tail < head) for dedup
    {
        const int nm = n_mst;
        int* t_ptr = thrust::raw_pointer_cast(mst_tails.data());
        int* h_ptr = thrust::raw_pointer_cast(mst_heads.data());

        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nm),
            [t_ptr, h_ptr] MST_HOST_DEVICE (int i) {
                if (t_ptr[i] > h_ptr[i]) {
                    int tmp = t_ptr[i];
                    t_ptr[i] = h_ptr[i];
                    h_ptr[i] = tmp;
                }
            });
    }

    // Sort by (tail, head) and deduplicate
    VectorType<int> dedup_indices(n_mst);
    thrust::sequence(dedup_indices.begin(), dedup_indices.end());
    thrust::sort_by_key(
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin(), mst_heads.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin() + n_mst, mst_heads.begin() + n_mst)),
        thrust::make_zip_iterator(thrust::make_tuple(mst_costs.begin(), dedup_indices.begin())));

    auto new_end = thrust::unique_by_key(
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin(), mst_heads.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin() + n_mst, mst_heads.begin() + n_mst)),
        thrust::make_zip_iterator(thrust::make_tuple(mst_costs.begin(), dedup_indices.begin())));

    int n_unique = new_end.first -
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin(), mst_heads.begin()));

    mst_tails.resize(n_unique);
    mst_heads.resize(n_unique);
    mst_costs.resize(n_unique);

    return {std::move(mst_tails), std::move(mst_heads), std::move(mst_costs)};
}

} // namespace MST_boruvka

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>>
MST_boruvka::maximum_spanning_tree<thrust::host_vector>(
    const thrust::host_vector<int>&, const thrust::host_vector<int>&, const thrust::host_vector<float>&);

extern template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
MST_boruvka::maximum_spanning_tree<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&);
