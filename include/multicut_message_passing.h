#pragma once

#include <cassert>
#include <algorithm>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/swap.h>
#include <thrust/count.h>
#include <thrust/tuple.h>
#include "graph.h"
#include "lifted_cut_constraints.h"
#include "time_measure_util.h"

#ifdef __CUDACC__
#define MMP_HOST_DEVICE __host__ __device__
#else
#define MMP_HOST_DEVICE
#endif

namespace mmp_detail {

MMP_HOST_DEVICE
inline float min_marginal(const float x, const float y, const float z) {
    float mm1 = thrust::min(x+y+z, thrust::min(x+y, x+z));
    float mm0 = thrust::min(0.0f, y+z);
    return mm1 - mm0;
}

template<template<typename> class VectorType>
inline void coo_sort_ij(VectorType<int>& i, VectorType<int>& j) {
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    thrust::sort(first, last);
}

template<template<typename> class VectorType>
inline void coo_sort_ijc(VectorType<int>& i, VectorType<int>& j, VectorType<float>& costs) {
    assert(i.size() == j.size());
    assert(i.size() == costs.size());
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    thrust::sort_by_key(first, last, costs.begin());
}

template<template<typename> class VectorType>
inline void coo_sort_ijk(VectorType<int>& i, VectorType<int>& j, VectorType<int>& k) {
    assert(i.size() == j.size());
    assert(i.size() == k.size());
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin(), k.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end(), k.end()));
    thrust::sort(first, last);
}

template<template<typename> class VectorType>
inline void normalize_triangles_impl(VectorType<int>& t1, VectorType<int>& t2, VectorType<int>& t3) {
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        auto new_last = thrust::remove_if(first, last,
            [] MMP_HOST_DEVICE (const thrust::tuple<int,int,int>& t) {
                return thrust::get<0>(t) == thrust::get<1>(t) ||
                    thrust::get<0>(t) == thrust::get<2>(t) ||
                    thrust::get<1>(t) == thrust::get<2>(t);
            });
        thrust::for_each(first, new_last,
            [] MMP_HOST_DEVICE (const thrust::tuple<int&,int&,int&> t) {
                int& x = thrust::get<0>(t);
                int& y = thrust::get<1>(t);
                int& z = thrust::get<2>(t);
                const int smallest = thrust::min(thrust::min(x, y), z);
                const int middle = thrust::max(thrust::min(x,y), thrust::min(thrust::max(x,y),z));
                const int largest = thrust::max(thrust::max(x, y), z);
                assert(smallest < middle && middle < largest);
                x = smallest;
                y = middle;
                z = largest;
            });
        t1.resize(std::distance(first, new_last));
        t2.resize(std::distance(first, new_last));
        t3.resize(std::distance(first, new_last));
    }
    {
        coo_sort_ijk<VectorType>(t1, t2, t3);
        assert(thrust::is_sorted(t1.begin(), t1.end()));
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        auto new_last = thrust::unique(first, last);
        t1.resize(std::distance(first, new_last));
        t2.resize(std::distance(first, new_last));
        t3.resize(std::distance(first, new_last));
    }
}

template<template<typename> class VectorType>
inline VectorType<int> invert_unique_impl(const VectorType<int>& values, const VectorType<int>& counts) {
    VectorType<int> counts_sum(counts.size() + 1);
    counts_sum[0] = 0;
    thrust::inclusive_scan(counts.begin(), counts.end(), counts_sum.begin() + 1);

    int out_size = counts_sum.back();
    VectorType<int> output_indices(out_size, 0);

    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + values.size(),
                    counts_sum.begin(), output_indices.begin());

    thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin());
    thrust::transform(output_indices.begin(), output_indices.end(),
                      thrust::make_constant_iterator(1), output_indices.begin(), thrust::minus<int>());

    VectorType<int> out_values(out_size);
    thrust::gather(output_indices.begin(), output_indices.end(), values.begin(), out_values.begin());

    return out_values;
}

} // namespace mmp_detail

// --- Class declaration ---

template<template<typename> class VectorType>
class multicut_message_passing {
public:
    multicut_message_passing(
            const Graph<VectorType>& A,
            VectorType<int>&& _t1,
            VectorType<int>&& _t2,
            VectorType<int>&& _t3,
            const bool verbose = true);

    multicut_message_passing(const Graph<VectorType>& A, const bool verbose = true);

    int add_triangles(VectorType<int>&& new_t1, VectorType<int>&& new_t2, VectorType<int>&& new_t3);

    Graph<VectorType> reparametrized_graph() const;

    void send_messages_to_triplets();
    void send_messages_to_edges();

    double lower_bound();

    void iteration();

    std::tuple<const VectorType<int>&, const VectorType<int>&, const VectorType<float>&>
        reparametrized_edge_costs() const;

    double edge_lower_bound();
    double triangle_lower_bound();

    int add_cut_factors(const LiftedCutFactors<VectorType>& factors,
                        const Graph<VectorType>& base_G,
                        const Graph<VectorType>& lifted_G);

    double cut_factor_lower_bound();

    static std::tuple<VectorType<int>, VectorType<int>, VectorType<float>>
        extract_directed_edges(const Graph<VectorType>& A);

    // Public for nvcc: __host__ __device__ lambdas require public access.
    void send_messages_to_cut_factors();
    void send_from_cut_factors_to_edges();

private:
    void compute_triangle_edge_correspondence(const VectorType<int>&, const VectorType<int>&,
        VectorType<int>&, VectorType<int>&);

    VectorType<int> i;
    VectorType<int> j;

    VectorType<int> t1;
    VectorType<int> t2;
    VectorType<int> t3;

    VectorType<float> edge_costs;

    VectorType<float> t12_costs;
    VectorType<float> t13_costs;
    VectorType<float> t23_costs;

    VectorType<int> triangle_correspondence_12;
    VectorType<int> triangle_correspondence_13;
    VectorType<int> triangle_correspondence_23;
    VectorType<int> edge_counter;

    int num_nodes_;

    // Cut factor data (CSR format)
    int num_cut_factors_ = 0;
    VectorType<int> cf_base_offsets;
    VectorType<int> cf_base_mp_idx;      // MP edge indices for base edges
    VectorType<float> cf_base_costs;
    VectorType<int> cf_lifted_offsets;
    VectorType<int> cf_lifted_mp_idx;    // MP edge indices for lifted edges
    VectorType<float> cf_lifted_costs;
};

// --- Implementation ---

template<template<typename> class VectorType>
void multicut_message_passing<VectorType>::compute_triangle_edge_correspondence(
    const VectorType<int>& ta, const VectorType<int>& tb,
    VectorType<int>& edge_counter, VectorType<int>& triangle_correspondence_ab)
{
    VectorType<int> t_sort_order(ta.size());
    thrust::sequence(t_sort_order.begin(), t_sort_order.end());
    VectorType<int> ta_unique(ta.size());
    VectorType<int> tb_unique(tb.size());
    VectorType<int> t_counts(ta.size());
    {
        VectorType<int> ta_sorted = ta;
        VectorType<int> tb_sorted = tb;
        auto first = thrust::make_zip_iterator(thrust::make_tuple(ta_sorted.begin(), tb_sorted.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(ta_sorted.end(), tb_sorted.end()));
        thrust::sort_by_key(first, last, t_sort_order.begin());

        auto first_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.begin(), tb_unique.begin()));
        auto last_reduce = thrust::reduce_by_key(first, last, thrust::make_constant_iterator(1), first_unique, t_counts.begin());
        int num_unique = std::distance(first_unique, last_reduce.first);
        ta_unique.resize(num_unique);
        tb_unique.resize(num_unique);
        t_counts.resize(num_unique);
    }

    auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    assert(thrust::is_sorted(first_edge, last_edge));
    assert(std::distance(first_edge, thrust::unique(first_edge, last_edge)) == i.size());

    auto first_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.begin(), tb_unique.begin()));
    auto last_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.end(), tb_unique.end()));
    VectorType<int> unique_correspondence(ta_unique.size());
    auto last_edge_int = thrust::set_intersection_by_key(first_edge, last_edge, first_unique, last_unique,
                                    thrust::make_counting_iterator(0), thrust::make_discard_iterator(),
                                    unique_correspondence.begin());

    unique_correspondence.resize(std::distance(unique_correspondence.begin(), last_edge_int.second));
    assert(unique_correspondence.size() == ta_unique.size()); // all triangle edges should be present.

    VectorType<int> correspondence_sorted = mmp_detail::invert_unique_impl<VectorType>(unique_correspondence, t_counts);
    thrust::scatter(correspondence_sorted.begin(), correspondence_sorted.end(), t_sort_order.begin(), triangle_correspondence_ab.begin());

    VectorType<int> edge_increment(i.size(), 0);
    thrust::scatter(t_counts.begin(), t_counts.end(), unique_correspondence.begin(), edge_increment.begin());
    thrust::transform(edge_counter.begin(), edge_counter.end(), edge_increment.begin(), edge_counter.begin(), thrust::plus<int>());
}

template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<float>>
multicut_message_passing<VectorType>::extract_directed_edges(const Graph<VectorType>& A)
{
    const auto& tails = A.get_tails();
    const auto& heads = A.get_heads();
    const auto& costs = A.get_costs();

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(tails.begin(), heads.begin(), costs.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(tails.end(), heads.end(), costs.end()));
    auto is_directed = [] MMP_HOST_DEVICE (const thrust::tuple<int,int,float>& t) -> bool {
        return thrust::get<0>(t) < thrust::get<1>(t);
    };
    size_t n_dir = thrust::count_if(begin, end, is_directed);

    VectorType<int> dir_i(n_dir);
    VectorType<int> dir_j(n_dir);
    VectorType<float> dir_costs(n_dir);

    auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(dir_i.begin(), dir_j.begin(), dir_costs.begin()));
    thrust::copy_if(begin, end, out_begin, is_directed);

    return {std::move(dir_i), std::move(dir_j), std::move(dir_costs)};
}

template<template<typename> class VectorType>
multicut_message_passing<VectorType>::multicut_message_passing(
        const Graph<VectorType>& A, const bool verbose)
    : num_nodes_(A.num_nodes())
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    std::tie(i, j, edge_costs) = extract_directed_edges(A);
    mmp_detail::coo_sort_ijc<VectorType>(i, j, edge_costs);
    if (verbose)
        std::cout << "edges size = " << i.size() << "\n";
    edge_counter = VectorType<int>(i.size(), 0);
}

template<template<typename> class VectorType>
multicut_message_passing<VectorType>::multicut_message_passing(
        const Graph<VectorType>& A,
        VectorType<int>&& _t1,
        VectorType<int>&& _t2,
        VectorType<int>&& _t3,
        const bool verbose)
    : multicut_message_passing(A, verbose)
{
    if (verbose)
        std::cout << "triangle size = " << _t1.size() << "\n";
    add_triangles(std::move(_t1), std::move(_t2), std::move(_t3));
}

template<template<typename> class VectorType>
int multicut_message_passing<VectorType>::add_triangles(
        VectorType<int>&& new_t1, VectorType<int>&& new_t2, VectorType<int>&& new_t3)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    assert(new_t1.size() == new_t2.size() && new_t1.size() == new_t3.size());
    if (new_t1.empty()) return 0;

    // 1. Normalize new triangles (sort vertices, remove degenerate, deduplicate)
    mmp_detail::normalize_triangles_impl<VectorType>(new_t1, new_t2, new_t3);
    if (new_t1.empty()) return 0;

    // 2. Deduplicate against existing triangles
    if (!t1.empty()) {
        VectorType<int> diff_t1(new_t1.size()), diff_t2(new_t1.size()), diff_t3(new_t1.size());

        auto existing_first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto existing_last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        auto new_first = thrust::make_zip_iterator(thrust::make_tuple(new_t1.begin(), new_t2.begin(), new_t3.begin()));
        auto new_last = thrust::make_zip_iterator(thrust::make_tuple(new_t1.end(), new_t2.end(), new_t3.end()));
        auto diff_first = thrust::make_zip_iterator(thrust::make_tuple(diff_t1.begin(), diff_t2.begin(), diff_t3.begin()));

        auto diff_end = thrust::set_difference(new_first, new_last, existing_first, existing_last, diff_first);
        int num_diff = std::distance(diff_first, diff_end);
        diff_t1.resize(num_diff);
        diff_t2.resize(num_diff);
        diff_t3.resize(num_diff);

        new_t1 = std::move(diff_t1);
        new_t2 = std::move(diff_t2);
        new_t3 = std::move(diff_t3);
    }

    const int num_new = new_t1.size();
    if (num_new == 0) return 0;

    // 3. Add any missing edges from new triangles to the edge set.
    //    Triangulated cycles (quadrangles, pentagons) introduce diagonal
    //    edges that may not exist in the original graph.
    {
        VectorType<int> tri_i(3 * num_new);
        VectorType<int> tri_j(3 * num_new);
        thrust::copy(new_t1.begin(), new_t1.end(), tri_i.begin());
        thrust::copy(new_t2.begin(), new_t2.end(), tri_j.begin());
        thrust::copy(new_t1.begin(), new_t1.end(), tri_i.begin() + num_new);
        thrust::copy(new_t3.begin(), new_t3.end(), tri_j.begin() + num_new);
        thrust::copy(new_t2.begin(), new_t2.end(), tri_i.begin() + 2 * num_new);
        thrust::copy(new_t3.begin(), new_t3.end(), tri_j.begin() + 2 * num_new);

        mmp_detail::coo_sort_ij<VectorType>(tri_i, tri_j);
        {
            auto first = thrust::make_zip_iterator(thrust::make_tuple(tri_i.begin(), tri_j.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(tri_i.end(), tri_j.end()));
            auto new_last = thrust::unique(first, last);
            tri_i.resize(std::distance(first, new_last));
            tri_j.resize(std::distance(first, new_last));
        }

        // Merge: existing edges keep their reparametrized costs,
        // new edges (diagonals from triangulated cycles) enter with cost 0.
        VectorType<float> tri_costs(tri_i.size(), 0.0f);

        auto first_existing = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
        auto last_existing = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
        auto first_tri = thrust::make_zip_iterator(thrust::make_tuple(tri_i.begin(), tri_j.begin()));
        auto last_tri = thrust::make_zip_iterator(thrust::make_tuple(tri_i.end(), tri_j.end()));

        VectorType<int> merged_i(i.size() + tri_i.size());
        VectorType<int> merged_j(j.size() + tri_j.size());
        VectorType<float> merged_costs(edge_costs.size() + tri_costs.size());
        auto first_merged = thrust::make_zip_iterator(thrust::make_tuple(merged_i.begin(), merged_j.begin()));

        auto merged_last = thrust::set_union_by_key(
            first_existing, last_existing, first_tri, last_tri,
            edge_costs.begin(), tri_costs.begin(),
            first_merged, merged_costs.begin());

        int num_merged = std::distance(first_merged, merged_last.first);

        // Skip swap if no new edges were actually added
        if (num_merged != (int)i.size())
        {
            merged_i.resize(num_merged);
            merged_j.resize(num_merged);
            merged_costs.resize(num_merged);

            // After swap: merged_i/j = old edges, i/j = new (expanded) edges.
            thrust::swap(i, merged_i);
            thrust::swap(j, merged_j);
            thrust::swap(edge_costs, merged_costs);

            // Remap cut factor edge indices to the new (expanded) edge array.
            if (num_cut_factors_ > 0)
            {
                auto new_first = thrust::make_zip_iterator(
                    thrust::make_tuple(i.begin(), j.begin()));
                auto new_last = thrust::make_zip_iterator(
                    thrust::make_tuple(i.end(), j.end()));

                auto remap = [&](VectorType<int>& mp_idx) {
                    if (mp_idx.empty()) return;
                    // Gather old (i,j) keys for each cut factor entry.
                    VectorType<int> keys_i(mp_idx.size()), keys_j(mp_idx.size());
                    thrust::gather(mp_idx.begin(), mp_idx.end(),
                                   merged_i.begin(), keys_i.begin());
                    thrust::gather(mp_idx.begin(), mp_idx.end(),
                                   merged_j.begin(), keys_j.begin());
                    // Find new positions via lower_bound in expanded edge array.
                    auto keys_first = thrust::make_zip_iterator(
                        thrust::make_tuple(keys_i.begin(), keys_j.begin()));
                    thrust::lower_bound(new_first, new_last,
                        keys_first, keys_first + mp_idx.size(),
                        mp_idx.begin());
                };
                remap(cf_base_mp_idx);
                remap(cf_lifted_mp_idx);
            }
        }
    }

    // 4. Concatenate new triangles to existing
    const int old_size = t1.size();
    const int total_size = old_size + num_new;

    t1.resize(total_size);
    t2.resize(total_size);
    t3.resize(total_size);
    thrust::copy(new_t1.begin(), new_t1.end(), t1.begin() + old_size);
    thrust::copy(new_t2.begin(), new_t2.end(), t2.begin() + old_size);
    thrust::copy(new_t3.begin(), new_t3.end(), t3.begin() + old_size);

    // Extend cost arrays with zeros for new triangles
    t12_costs.resize(total_size, 0.0f);
    t13_costs.resize(total_size, 0.0f);
    t23_costs.resize(total_size, 0.0f);

    // 4. Sort combined triangles by (t1,t2,t3), carrying cost arrays via permutation
    {
        VectorType<int> perm(total_size);
        thrust::sequence(perm.begin(), perm.end());

        auto key_first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto key_last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        thrust::sort_by_key(key_first, key_last, perm.begin());

        VectorType<float> tmp12(total_size), tmp13(total_size), tmp23(total_size);
        thrust::gather(perm.begin(), perm.end(), t12_costs.begin(), tmp12.begin());
        thrust::gather(perm.begin(), perm.end(), t13_costs.begin(), tmp13.begin());
        thrust::gather(perm.begin(), perm.end(), t23_costs.begin(), tmp23.begin());
        thrust::swap(t12_costs, tmp12);
        thrust::swap(t13_costs, tmp13);
        thrust::swap(t23_costs, tmp23);
    }

    // 5. Recompute all triangle-edge correspondences and edge_counter from scratch
    triangle_correspondence_12 = VectorType<int>(total_size);
    triangle_correspondence_13 = VectorType<int>(total_size);
    triangle_correspondence_23 = VectorType<int>(total_size);
    edge_counter = VectorType<int>(i.size(), 0);

    compute_triangle_edge_correspondence(t1, t2, edge_counter, triangle_correspondence_12);
    compute_triangle_edge_correspondence(t1, t3, edge_counter, triangle_correspondence_13);
    compute_triangle_edge_correspondence(t2, t3, edge_counter, triangle_correspondence_23);

    // Re-add cut factor contributions lost by the edge_counter reset above.
    if (num_cut_factors_ > 0)
    {
        int total = (int)cf_base_mp_idx.size() + (int)cf_lifted_mp_idx.size();
        VectorType<int> all_idx(total);
        thrust::copy(cf_base_mp_idx.begin(), cf_base_mp_idx.end(), all_idx.begin());
        thrust::copy(cf_lifted_mp_idx.begin(), cf_lifted_mp_idx.end(),
                     all_idx.begin() + cf_base_mp_idx.size());

        VectorType<int> sorted_idx(all_idx);
        thrust::sort(sorted_idx.begin(), sorted_idx.end());

        VectorType<int> unique_idx(total), counts(total);
        auto end = thrust::reduce_by_key(sorted_idx.begin(), sorted_idx.end(),
            thrust::make_constant_iterator(1), unique_idx.begin(), counts.begin());
        int num_unique = (int)std::distance(unique_idx.begin(), end.first);

        VectorType<int> increments(edge_counter.size(), 0);
        thrust::scatter(counts.begin(), counts.begin() + num_unique,
                        unique_idx.begin(), increments.begin());
        thrust::transform(edge_counter.begin(), edge_counter.end(), increments.begin(),
                          edge_counter.begin(), thrust::plus<int>());
    }

    return num_new;
}

template<template<typename> class VectorType>
Graph<VectorType> multicut_message_passing<VectorType>::reparametrized_graph() const
{
    VectorType<int> tails(i);
    VectorType<int> heads(j);
    VectorType<float> costs(edge_costs);
    return Graph<VectorType>(num_nodes_, std::move(tails), std::move(heads), std::move(costs));
}

template<template<typename> class VectorType>
double multicut_message_passing<VectorType>::edge_lower_bound()
{
    return thrust::transform_reduce(edge_costs.begin(), edge_costs.end(),
        [] MMP_HOST_DEVICE (const float x) -> float { return x < 0.0 ? x : 0.0; },
        0.0, thrust::plus<double>());
}

template<template<typename> class VectorType>
double multicut_message_passing<VectorType>::triangle_lower_bound()
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(t12_costs.begin(), t13_costs.begin(), t23_costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(t12_costs.end(), t13_costs.end(), t23_costs.end()));
    return thrust::transform_reduce(first, last,
        [] MMP_HOST_DEVICE (const thrust::tuple<float,float,float> x) -> float {
            const float c12 = thrust::get<0>(x);
            const float c13 = thrust::get<1>(x);
            const float c23 = thrust::get<2>(x);
            float lb = 0.0;
            lb = thrust::min(lb, c12 + c13);
            lb = thrust::min(lb, c12 + c23);
            lb = thrust::min(lb, c13 + c23);
            lb = thrust::min(lb, c12 + c13 + c23);
            return lb;
        },
        0.0, thrust::plus<double>());
}

template<template<typename> class VectorType>
double multicut_message_passing<VectorType>::lower_bound()
{
    return edge_lower_bound() + triangle_lower_bound() + cut_factor_lower_bound();
}

template<template<typename> class VectorType>
void multicut_message_passing<VectorType>::send_messages_to_triplets()
{
    float* ec_ptr = thrust::raw_pointer_cast(edge_costs.data());
    int* cnt_ptr = thrust::raw_pointer_cast(edge_counter.data());

    auto increase_triangle_costs = [ec_ptr, cnt_ptr] MMP_HOST_DEVICE (const thrust::tuple<int,float&> t) {
        const int edge_idx = thrust::get<0>(t);
        float& triangle_cost = thrust::get<1>(t);
        assert(cnt_ptr[edge_idx] > 0);
        triangle_cost += ec_ptr[edge_idx] / float(cnt_ptr[edge_idx]);
    };

    // send costs to triangles
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.begin(), t12_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.end(), t12_costs.end()));
        thrust::for_each(first, last, increase_triangle_costs);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.begin(), t13_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.end(), t13_costs.end()));
        thrust::for_each(first, last, increase_triangle_costs);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.begin(), t23_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.end(), t23_costs.end()));
        thrust::for_each(first, last, increase_triangle_costs);
    }

    // set costs of edges to zero (if edge participates in a triangle)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.begin(), edge_counter.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.end(), edge_counter.end()));
        thrust::for_each(first, last,
            [] MMP_HOST_DEVICE (const thrust::tuple<float&,int> x) {
                float& cost = thrust::get<0>(x);
                int counter = thrust::get<1>(x);
                if (counter > 0)
                    cost = 0.0;
            });
    }
}

template<template<typename> class VectorType>
void multicut_message_passing<VectorType>::send_messages_to_edges()
{
    const int num_triangles = t1.size();
    if (num_triangles == 0) return;

    // Phase 1: per-triangle — update triangle costs in-place, store edge diffs.
    // Fully parallel, no atomics needed since triangles are independent.
    VectorType<float> d12(num_triangles), d13(num_triangles), d23(num_triangles);
    {
        float* t12_ptr = thrust::raw_pointer_cast(t12_costs.data());
        float* t13_ptr = thrust::raw_pointer_cast(t13_costs.data());
        float* t23_ptr = thrust::raw_pointer_cast(t23_costs.data());
        float* d12_ptr = thrust::raw_pointer_cast(d12.data());
        float* d13_ptr = thrust::raw_pointer_cast(d13.data());
        float* d23_ptr = thrust::raw_pointer_cast(d23.data());
        thrust::for_each(thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(num_triangles),
            [t12_ptr, t13_ptr, t23_ptr, d12_ptr, d13_ptr, d23_ptr]
            MMP_HOST_DEVICE (const int t) {
                float& t12c = t12_ptr[t];
                float& t13c = t13_ptr[t];
                float& t23c = t23_ptr[t];
                float e12_diff = 0.0, e13_diff = 0.0, e23_diff = 0.0;
                { const float mm = mmp_detail::min_marginal(t12c, t13c, t23c); t12c -= 1.0/3.0*mm; e12_diff += 1.0/3.0*mm; }
                { const float mm = mmp_detail::min_marginal(t13c, t12c, t23c); t13c -= 1.0/2.0*mm; e13_diff += 1.0/2.0*mm; }
                { const float mm = mmp_detail::min_marginal(t23c, t12c, t13c); t23c -= mm;          e23_diff += mm; }
                { const float mm = mmp_detail::min_marginal(t12c, t13c, t23c); t12c -= 1.0/2.0*mm; e12_diff += 1.0/2.0*mm; }
                { const float mm = mmp_detail::min_marginal(t13c, t12c, t23c); t13c -= mm;          e13_diff += mm; }
                { const float mm = mmp_detail::min_marginal(t12c, t13c, t23c); t12c -= mm;          e12_diff += mm; }
                d12_ptr[t] = e12_diff;
                d13_ptr[t] = e13_diff;
                d23_ptr[t] = e23_diff;
            });
    }

    // Phase 2: build flat (edge_idx, delta) pairs — 3 per triangle.
    const int num_pairs = 3 * num_triangles;
    VectorType<int> all_idx(num_pairs);
    VectorType<float> all_deltas(num_pairs);
    thrust::copy(triangle_correspondence_12.begin(), triangle_correspondence_12.end(), all_idx.begin());
    thrust::copy(triangle_correspondence_13.begin(), triangle_correspondence_13.end(), all_idx.begin() + num_triangles);
    thrust::copy(triangle_correspondence_23.begin(), triangle_correspondence_23.end(), all_idx.begin() + 2 * num_triangles);
    thrust::copy(d12.begin(), d12.end(), all_deltas.begin());
    thrust::copy(d13.begin(), d13.end(), all_deltas.begin() + num_triangles);
    thrust::copy(d23.begin(), d23.end(), all_deltas.begin() + 2 * num_triangles);

    // Phase 3: sort by edge index so reduce_by_key can aggregate per edge.
    thrust::sort_by_key(all_idx.begin(), all_idx.end(), all_deltas.begin());

    // Phase 4: sum all deltas that belong to the same edge.
    VectorType<int> reduced_keys(num_pairs);
    VectorType<float> reduced_vals(num_pairs);
    auto reduce_end = thrust::reduce_by_key(
        all_idx.begin(), all_idx.end(), all_deltas.begin(),
        reduced_keys.begin(), reduced_vals.begin());
    const int num_unique = std::distance(reduced_keys.begin(), reduce_end.first);

    // Phase 5: scatter-add the per-edge sums into edge_costs.
    VectorType<float> increments(edge_costs.size(), 0.0f);
    thrust::scatter(reduced_vals.begin(), reduced_vals.begin() + num_unique,
                    reduced_keys.begin(), increments.begin());
    thrust::transform(edge_costs.begin(), edge_costs.end(), increments.begin(),
                      edge_costs.begin(), thrust::plus<float>());
}

template<template<typename> class VectorType>
void multicut_message_passing<VectorType>::iteration()
{
    send_messages_to_cut_factors();
    send_messages_to_triplets();
    send_messages_to_edges();
    send_from_cut_factors_to_edges();
}

template<template<typename> class VectorType>
std::tuple<const VectorType<int>&, const VectorType<int>&, const VectorType<float>&>
multicut_message_passing<VectorType>::reparametrized_edge_costs() const
{
    return {i, j, edge_costs};
}

// --- Cut factor methods ---

template<template<typename> class VectorType>
int multicut_message_passing<VectorType>::add_cut_factors(
    const LiftedCutFactors<VectorType>& factors,
    const Graph<VectorType>& base_G,
    const Graph<VectorType>& lifted_G)
{
    if (factors.num_factors == 0) return 0;

    const int old_num_factors = num_cut_factors_;
    const int old_num_base = (int)cf_base_mp_idx.size();
    const int old_num_lifted = (int)cf_lifted_mp_idx.size();

    const int new_num_base_entries = (int)factors.base_edge_idx.size();
    const int new_num_lifted_entries = (int)factors.lifted_edge_idx.size();
    const int num_mp_edges = (int)i.size();

    // Encode MP edges as long long keys for binary search
    VectorType<long long> mp_keys(num_mp_edges);
    {
        const int* i_ptr = thrust::raw_pointer_cast(i.data());
        const int* j_ptr = thrust::raw_pointer_cast(j.data());
        long long* k_ptr = thrust::raw_pointer_cast(mp_keys.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_mp_edges),
            [i_ptr, j_ptr, k_ptr] MMP_HOST_DEVICE (int e) {
                k_ptr[e] = ((long long)i_ptr[e] << 32) | (long long)(unsigned int)j_ptr[e];
            });
    }

    // Map new base edge indices (in base_G) to MP edge indices
    VectorType<int> new_base_mp_idx(new_num_base_entries);
    {
        const int* bi_ptr = thrust::raw_pointer_cast(factors.base_edge_idx.data());
        const int* bt_ptr = base_G.get_tails_ptr();
        const int* bh_ptr = base_G.get_heads_ptr();
        const long long* mk_ptr = thrust::raw_pointer_cast(mp_keys.data());
        int* out_ptr = thrust::raw_pointer_cast(new_base_mp_idx.data());
        int nm = num_mp_edges;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(new_num_base_entries),
            [bi_ptr, bt_ptr, bh_ptr, mk_ptr, out_ptr, nm] MMP_HOST_DEVICE (int k) {
                int idx = bi_ptr[k];
                int t = bt_ptr[idx], h = bh_ptr[idx];
                if (t > h) { int tmp = t; t = h; h = tmp; }
                long long key = ((long long)t << 32) | (long long)(unsigned int)h;
                int lo = 0, hi = nm;
                while (lo < hi) { int mid = (lo + hi) / 2; if (mk_ptr[mid] < key) lo = mid + 1; else hi = mid; }
                out_ptr[k] = lo;
            });
    }

    // Map new lifted edge indices (in lifted_G) to MP edge indices
    VectorType<int> new_lifted_mp_idx(new_num_lifted_entries);
    {
        const int* li_ptr = thrust::raw_pointer_cast(factors.lifted_edge_idx.data());
        const int* lt_ptr = lifted_G.get_tails_ptr();
        const int* lh_ptr = lifted_G.get_heads_ptr();
        const long long* mk_ptr = thrust::raw_pointer_cast(mp_keys.data());
        int* out_ptr = thrust::raw_pointer_cast(new_lifted_mp_idx.data());
        int nm = num_mp_edges;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(new_num_lifted_entries),
            [li_ptr, lt_ptr, lh_ptr, mk_ptr, out_ptr, nm] MMP_HOST_DEVICE (int k) {
                int idx = li_ptr[k];
                int t = lt_ptr[idx], h = lh_ptr[idx];
                if (t > h) { int tmp = t; t = h; h = tmp; }
                long long key = ((long long)t << 32) | (long long)(unsigned int)h;
                int lo = 0, hi = nm;
                while (lo < hi) { int mid = (lo + hi) / 2; if (mk_ptr[mid] < key) lo = mid + 1; else hi = mid; }
                out_ptr[k] = lo;
            });
    }

    // Append to CSR offsets (shift new offsets by old totals)
    {
        VectorType<int> merged_base_offsets(old_num_factors + factors.num_factors + 1);
        if (old_num_factors > 0)
            thrust::copy(cf_base_offsets.begin(), cf_base_offsets.end(), merged_base_offsets.begin());
        else
            merged_base_offsets[0] = 0;
        const int base_shift = old_num_base;
        const int* new_off = thrust::raw_pointer_cast(factors.base_offsets.data());
        int* out = thrust::raw_pointer_cast(merged_base_offsets.data()) + old_num_factors + 1;
        int nf = factors.num_factors;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nf),
            [new_off, out, base_shift] MMP_HOST_DEVICE (int f) {
                out[f] = new_off[f + 1] + base_shift;
            });
        cf_base_offsets = std::move(merged_base_offsets);

        VectorType<int> merged_lifted_offsets(old_num_factors + factors.num_factors + 1);
        if (old_num_factors > 0)
            thrust::copy(cf_lifted_offsets.begin(), cf_lifted_offsets.end(), merged_lifted_offsets.begin());
        else
            merged_lifted_offsets[0] = 0;
        const int lifted_shift = old_num_lifted;
        const int* new_loff = thrust::raw_pointer_cast(factors.lifted_offsets.data());
        int* lout = thrust::raw_pointer_cast(merged_lifted_offsets.data()) + old_num_factors + 1;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nf),
            [new_loff, lout, lifted_shift] MMP_HOST_DEVICE (int f) {
                lout[f] = new_loff[f + 1] + lifted_shift;
            });
        cf_lifted_offsets = std::move(merged_lifted_offsets);
    }

    // Append to mp_idx and cost arrays
    cf_base_mp_idx.resize(old_num_base + new_num_base_entries);
    thrust::copy(new_base_mp_idx.begin(), new_base_mp_idx.end(),
                 cf_base_mp_idx.begin() + old_num_base);

    cf_lifted_mp_idx.resize(old_num_lifted + new_num_lifted_entries);
    thrust::copy(new_lifted_mp_idx.begin(), new_lifted_mp_idx.end(),
                 cf_lifted_mp_idx.begin() + old_num_lifted);

    cf_base_costs.resize(old_num_base + new_num_base_entries, 0.0f);
    cf_lifted_costs.resize(old_num_lifted + new_num_lifted_entries, 0.0f);

    num_cut_factors_ = old_num_factors + factors.num_factors;

    // Update edge_counter for new entries only
    {
        int total = new_num_base_entries + new_num_lifted_entries;
        VectorType<int> all_idx(total);
        thrust::copy(new_base_mp_idx.begin(), new_base_mp_idx.end(), all_idx.begin());
        thrust::copy(new_lifted_mp_idx.begin(), new_lifted_mp_idx.end(),
                     all_idx.begin() + new_num_base_entries);

        VectorType<int> sorted_idx(all_idx);
        thrust::sort(sorted_idx.begin(), sorted_idx.end());

        VectorType<int> unique_idx(total), counts(total);
        auto end = thrust::reduce_by_key(sorted_idx.begin(), sorted_idx.end(),
            thrust::make_constant_iterator(1), unique_idx.begin(), counts.begin());
        int num_unique = (int)std::distance(unique_idx.begin(), end.first);

        VectorType<int> increments(edge_counter.size(), 0);
        thrust::scatter(counts.begin(), counts.begin() + num_unique,
                        unique_idx.begin(), increments.begin());
        thrust::transform(edge_counter.begin(), edge_counter.end(), increments.begin(),
                          edge_counter.begin(), thrust::plus<int>());
    }

    return factors.num_factors;
}

template<template<typename> class VectorType>
void multicut_message_passing<VectorType>::send_messages_to_cut_factors()
{
    if (num_cut_factors_ == 0) return;

    // Distribute edge cost share to each cut factor base/lifted entry
    const float* ec_ptr = thrust::raw_pointer_cast(edge_costs.data());
    const int* cnt_ptr = thrust::raw_pointer_cast(edge_counter.data());

    {
        const int* idx_ptr = thrust::raw_pointer_cast(cf_base_mp_idx.data());
        float* bc_ptr = thrust::raw_pointer_cast(cf_base_costs.data());
        int n = (int)cf_base_mp_idx.size();
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(n),
            [ec_ptr, cnt_ptr, idx_ptr, bc_ptr] MMP_HOST_DEVICE (int k) {
                int e = idx_ptr[k];
                if (cnt_ptr[e] > 0)
                    bc_ptr[k] += ec_ptr[e] / float(cnt_ptr[e]);
            });
    }
    {
        const int* idx_ptr = thrust::raw_pointer_cast(cf_lifted_mp_idx.data());
        float* lc_ptr = thrust::raw_pointer_cast(cf_lifted_costs.data());
        int n = (int)cf_lifted_mp_idx.size();
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(n),
            [ec_ptr, cnt_ptr, idx_ptr, lc_ptr] MMP_HOST_DEVICE (int k) {
                int e = idx_ptr[k];
                if (cnt_ptr[e] > 0)
                    lc_ptr[k] += ec_ptr[e] / float(cnt_ptr[e]);
            });
    }
}

template<template<typename> class VectorType>
void multicut_message_passing<VectorType>::send_from_cut_factors_to_edges()
{
    if (num_cut_factors_ == 0) return;

    const int num_base_entries = (int)cf_base_mp_idx.size();
    const int num_lifted_entries = (int)cf_lifted_mp_idx.size();

    VectorType<float> b_deltas(num_base_entries, 0.0f);
    VectorType<float> l_deltas(num_lifted_entries, 0.0f);

    // Per-factor sequential relaxation: compute min-marginals and update costs
    {
        const int* bo_ptr = thrust::raw_pointer_cast(cf_base_offsets.data());
        const int* lo_ptr = thrust::raw_pointer_cast(cf_lifted_offsets.data());
        float* bc_ptr = thrust::raw_pointer_cast(cf_base_costs.data());
        float* lc_ptr = thrust::raw_pointer_cast(cf_lifted_costs.data());
        float* bd_ptr = thrust::raw_pointer_cast(b_deltas.data());
        float* ld_ptr = thrust::raw_pointer_cast(l_deltas.data());

        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_cut_factors_),
            [bo_ptr, lo_ptr, bc_ptr, lc_ptr, bd_ptr, ld_ptr] MMP_HOST_DEVICE (int f) {
                const int bs = bo_ptr[f], be = bo_ptr[f + 1];
                const int ls = lo_ptr[f], le = lo_ptr[f + 1];
                const int nb = be - bs, nl = le - ls;
                if (nb == 0 || nl == 0) return;

                for (int round = 0; round < 3; round++) {
                    const float damping = (round < 2) ? 0.5f : 1.0f;

                    // Base edges
                    for (int bi = bs; bi < be; bi++) {
                        float sb = 0, pb = 0, max_b = -1e30f, second_max_b = -1e30f;
                        bool all_neg = true;
                        int max_idx = bs;
                        for (int k = bs; k < be; k++) {
                            float c = bc_ptr[k];
                            if (c < 0) sb += c;
                            pb += c;
                            if (c >= 0) all_neg = false;
                            if (c > max_b) { second_max_b = max_b; max_b = c; max_idx = k; }
                            else if (c > second_max_b) second_max_b = c;
                        }
                        float sl = 0, pl = 0;
                        int num_nonneg_l = 0;
                        for (int k = ls; k < le; k++) {
                            float c = lc_ptr[k];
                            if (c < 0) sl += c;
                            pl += c;
                            if (c >= 0) num_nonneg_l++;
                        }

                        float c_bj = bc_ptr[bi];
                        float min0_bj = (c_bj < 0) ? c_bj : 0.0f;
                        float val0 = sb - min0_bj + sl;

                        float val1;
                        if (nb == 1) {
                            val1 = c_bj + pl;
                        } else {
                            float sub_sb = sb - min0_bj;
                            float sub_pb = pb - c_bj;
                            float sub_max = (bi == max_idx) ? second_max_b : max_b;
                            bool sub_all_neg = (bi == max_idx) ? (second_max_b < 0) : all_neg;
                            bool any_nonneg_l = (num_nonneg_l > 0);

                            if (!(sub_all_neg && any_nonneg_l)) {
                                val1 = c_bj + sub_sb + sl;
                            } else {
                                float a = sub_sb - sub_max + sl;
                                float b = sub_pb + pl;
                                val1 = c_bj + ((a < b) ? a : b);
                            }
                        }
                        float mm = damping * (val1 - val0);
                        bd_ptr[bi] += mm;
                        bc_ptr[bi] -= mm;
                    }

                    // Lifted edges
                    for (int li = ls; li < le; li++) {
                        float sb = 0, pb = 0, max_b = -1e30f;
                        bool all_neg = true;
                        for (int k = bs; k < be; k++) {
                            float c = bc_ptr[k];
                            if (c < 0) sb += c;
                            pb += c;
                            if (c >= 0) all_neg = false;
                            if (c > max_b) max_b = c;
                        }
                        float sl = 0, pl = 0;
                        int num_nonneg_l = 0;
                        for (int k = ls; k < le; k++) {
                            float c = lc_ptr[k];
                            if (c < 0) sl += c;
                            pl += c;
                            if (c >= 0) num_nonneg_l++;
                        }

                        float c_lj = lc_ptr[li];
                        float min0_lj = (c_lj < 0) ? c_lj : 0.0f;

                        float val0;
                        if (!all_neg) {
                            val0 = sb + sl - min0_lj;
                        } else {
                            val0 = sb - max_b + sl - min0_lj;
                        }

                        float sub_sl = sl - min0_lj;
                        float sub_pl = pl - c_lj;
                        int sub_nonneg = num_nonneg_l - (c_lj >= 0.0f ? 1 : 0);

                        float val1;
                        if (!(all_neg && sub_nonneg > 0)) {
                            val1 = c_lj + sb + sub_sl;
                        } else {
                            float a = sb - max_b + sub_sl;
                            float b = pb + sub_pl;
                            val1 = c_lj + ((a < b) ? a : b);
                        }
                        float mm = damping * (val1 - val0);
                        ld_ptr[li] += mm;
                        lc_ptr[li] -= mm;
                    }
                }
            });
    }

    // Scatter-add deltas back to edge_costs
    const int total_deltas = num_base_entries + num_lifted_entries;
    VectorType<int> all_idx(total_deltas);
    VectorType<float> all_deltas(total_deltas);
    thrust::copy(cf_base_mp_idx.begin(), cf_base_mp_idx.end(), all_idx.begin());
    thrust::copy(cf_lifted_mp_idx.begin(), cf_lifted_mp_idx.end(),
                 all_idx.begin() + num_base_entries);
    thrust::copy(b_deltas.begin(), b_deltas.end(), all_deltas.begin());
    thrust::copy(l_deltas.begin(), l_deltas.end(), all_deltas.begin() + num_base_entries);

    thrust::sort_by_key(all_idx.begin(), all_idx.end(), all_deltas.begin());

    VectorType<int> reduced_keys(total_deltas);
    VectorType<float> reduced_vals(total_deltas);
    auto reduce_end = thrust::reduce_by_key(
        all_idx.begin(), all_idx.end(), all_deltas.begin(),
        reduced_keys.begin(), reduced_vals.begin());
    int num_unique = (int)std::distance(reduced_keys.begin(), reduce_end.first);

    VectorType<float> increments(edge_costs.size(), 0.0f);
    thrust::scatter(reduced_vals.begin(), reduced_vals.begin() + num_unique,
                    reduced_keys.begin(), increments.begin());
    thrust::transform(edge_costs.begin(), edge_costs.end(), increments.begin(),
                      edge_costs.begin(), thrust::plus<float>());
}

template<template<typename> class VectorType>
double multicut_message_passing<VectorType>::cut_factor_lower_bound()
{
    if (num_cut_factors_ == 0) return 0.0;

    const int* bo_ptr = thrust::raw_pointer_cast(cf_base_offsets.data());
    const int* lo_ptr = thrust::raw_pointer_cast(cf_lifted_offsets.data());
    const float* bc_ptr = thrust::raw_pointer_cast(cf_base_costs.data());
    const float* lc_ptr = thrust::raw_pointer_cast(cf_lifted_costs.data());

    return thrust::transform_reduce(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_cut_factors_),
        [bo_ptr, lo_ptr, bc_ptr, lc_ptr] MMP_HOST_DEVICE (int f) -> float {
            const int bs = bo_ptr[f], be = bo_ptr[f + 1];
            const int ls = lo_ptr[f], le = lo_ptr[f + 1];

            float sb = 0, pb = 0, max_b = -1e30f;
            bool all_neg = true;
            for (int k = bs; k < be; k++) {
                float c = bc_ptr[k];
                if (c < 0) sb += c;
                pb += c;
                if (c >= 0) all_neg = false;
                if (c > max_b) max_b = c;
            }
            float sl = 0, pl = 0;
            bool any_nonneg_l = false;
            for (int k = ls; k < le; k++) {
                float c = lc_ptr[k];
                if (c < 0) sl += c;
                pl += c;
                if (c >= 0) any_nonneg_l = true;
            }

            if (!(all_neg && any_nonneg_l))
                return sb + sl;

            float opt_a = sb - max_b + sl;
            float opt_b = pb + pl;
            return (opt_a < opt_b) ? opt_a : opt_b;
        },
        0.0, thrust::plus<double>());
}

// Explicit instantiation declarations.
extern template class multicut_message_passing<thrust::host_vector>;
extern template class multicut_message_passing<thrust::device_vector>;