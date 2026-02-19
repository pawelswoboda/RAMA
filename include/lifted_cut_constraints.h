#pragma once

#include "graph.h"
#include "connected_components.h"
#include "rama_utils.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#ifdef __CUDACC__
#define LCC_HOST_DEVICE __host__ __device__
#else
#define LCC_HOST_DEVICE
#endif

namespace lifted_cut_detail {

LCC_HOST_DEVICE inline int uf_find(int* parent, int x)
{
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}

LCC_HOST_DEVICE inline unsigned int hash32(unsigned int x)
{
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
    return x;
}

// One Karger contraction trial for a given (s,t) pair.
// Processes Q-edges in permutation order, never merging s and t.
// Writes the resulting cut size to cut_size[pair_idx].
struct karger_trial_functor {
    int* parent;
    int* cut_size;
    const int* perm;
    const int* q_src;
    const int* q_dst;
    const int* pair_s;
    const int* pair_t;
    int E, N;

    LCC_HOST_DEVICE void operator()(int pair_idx) const
    {
        int s = pair_s[pair_idx];
        int t = pair_t[pair_idx];
        int* my_parent = parent + pair_idx * N;

        for (int i = 0; i < N; i++) my_parent[i] = i;

        for (int k = 0; k < E; k++)
        {
            int e = perm[k];
            int u = uf_find(my_parent, q_src[e]);
            int v = uf_find(my_parent, q_dst[e]);
            if (u == v) continue;
            int su = uf_find(my_parent, s);
            int sv = uf_find(my_parent, t);
            if ((u == su && v == sv) || (u == sv && v == su)) continue;
            my_parent[v] = u;
        }

        int su = uf_find(my_parent, s);
        int sv = uf_find(my_parent, t);
        int count = 0;
        for (int e = 0; e < E; e++)
        {
            int u = uf_find(my_parent, q_src[e]);
            int v = uf_find(my_parent, q_dst[e]);
            if ((u == su && v == sv) || (u == sv && v == su))
                count++;
        }
        cut_size[pair_idx] = count;
    }
};

// Keep the better of current vs best trial per pair.
struct update_best_functor {
    int* best_parent;
    int* best_cut_size;
    const int* cur_parent;
    const int* cur_cut_size;
    int N;

    LCC_HOST_DEVICE void operator()(int pair_idx) const
    {
        if (cur_cut_size[pair_idx] < best_cut_size[pair_idx])
        {
            best_cut_size[pair_idx] = cur_cut_size[pair_idx];
            const int* src = cur_parent + pair_idx * N;
            int* dst = best_parent + pair_idx * N;
            for (int i = 0; i < N; i++) dst[i] = src[i];
        }
    }
};

} // namespace lifted_cut_detail

// A violated lifted cut constraint: a lifted edge whose endpoints are in different
// components of the non-negative base subgraph, together with a small s-t cut in the
// quotient graph found by Karger's algorithm.
struct LiftedCutConstraint {
    int lifted_fwd_idx;                         // directed edge index in lifted_G (tail < head)
    float lifted_cost;                          // cost of the lifted edge (positive)
    std::set<std::pair<int,int>> cut_comp_pairs; // component pairs whose inter-component base edges form the cut
    int cut_size;                               // number of undirected base edges in the cut
    float min_abs_base_cost;                    // minimum |cost| among those base edges
};

// Find violated lifted cut constraints and small cuts for reparametrization.
//
// A lifted edge (s,t) with positive cost is "violated" when s and t are in different
// connected components of the non-negative base subgraph. This means the LP relaxation
// can set x_{st}=0 while cutting base edges on every s-t path -- violating the
// cut constraint x_{st} >= sum_{e in C} x_e - (|C|-1).
//
// For each violation, builds a quotient graph Q (nodes = components, edges = negative
// inter-component base edges) and runs Karger's randomized min s-t cut algorithm to
// find a small cut suitable for Lagrangian reparametrization.
template<template<typename> class VectorType>
std::vector<LiftedCutConstraint> find_lifted_cut_constraints(
    const Graph<VectorType>& base_G,
    const Graph<VectorType>& lifted_G,
    bool verbose = false)
{
    std::vector<LiftedCutConstraint> result;

    if (lifted_G.num_directed_edges() == 0)
        return result;

    const int num_nodes = base_G.num_nodes();
    const int num_base_dir = (int)base_G.num_directed_edges();
    const int num_lifted_dir = (int)lifted_G.num_directed_edges();

    // ---- Phase 1: CC on non-negative base subgraph ----

    auto is_nonneg = [] LCC_HOST_DEVICE (float c) { return c >= 0.0f; };
    int num_nonneg = thrust::count_if(
        base_G.get_costs().begin(), base_G.get_costs().end(), is_nonneg);

    VectorType<int> attr_t(num_nonneg), attr_h(num_nonneg);
    thrust::copy_if(base_G.get_tails().begin(), base_G.get_tails().begin() + num_base_dir,
                    base_G.get_costs().begin(), attr_t.begin(), is_nonneg);
    thrust::copy_if(base_G.get_heads().begin(), base_G.get_heads().begin() + num_base_dir,
                    base_G.get_costs().begin(), attr_h.begin(), is_nonneg);

    VectorType<int> comp =
        connected_components::compute_cc<VectorType>(num_nodes, attr_t, attr_h);

    int max_label = thrust::reduce(comp.begin(), comp.end(), 0, thrust::maximum<int>());
    comp = compress_label_sequence<VectorType>(comp, max_label);
    int num_comp = thrust::reduce(comp.begin(), comp.end(), 0, thrust::maximum<int>()) + 1;

    if (num_comp <= 1)
        return result;

    // ---- Phase 2: Build quotient graph Q ----
    // One Q-edge per undirected negative inter-component base edge (comp_tail < comp_head).

    VectorType<int> base_ct(num_base_dir), base_ch(num_base_dir);
    thrust::gather(base_G.get_tails().begin(), base_G.get_tails().begin() + num_base_dir,
                   comp.begin(), base_ct.begin());
    thrust::gather(base_G.get_heads().begin(), base_G.get_heads().begin() + num_base_dir,
                   comp.begin(), base_ch.begin());

    VectorType<int> q_mask(num_base_dir);
    {
        const float* bc_ptr = base_G.get_costs_ptr();
        const int* ct_ptr = thrust::raw_pointer_cast(base_ct.data());
        const int* ch_ptr = thrust::raw_pointer_cast(base_ch.data());
        int* mask_ptr = thrust::raw_pointer_cast(q_mask.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_base_dir),
            [bc_ptr, ct_ptr, ch_ptr, mask_ptr] LCC_HOST_DEVICE (int e) {
                mask_ptr[e] = (bc_ptr[e] < 0.0f && ct_ptr[e] < ch_ptr[e]) ? 1 : 0;
            });
    }

    int num_q = thrust::reduce(q_mask.begin(), q_mask.end(), 0);
    if (num_q == 0)
        return result;

    auto is_one = [] LCC_HOST_DEVICE (int x) { return x == 1; };
    VectorType<int> q_src(num_q), q_dst(num_q);
    thrust::copy_if(base_ct.begin(), base_ct.end(), q_mask.begin(), q_src.begin(), is_one);
    thrust::copy_if(base_ch.begin(), base_ch.end(), q_mask.begin(), q_dst.begin(), is_one);

    // ---- Phase 3: Find violations ----
    // Positive lifted edges (tail < head) with endpoints in different components.

    VectorType<int> lift_ct(num_lifted_dir), lift_ch(num_lifted_dir);
    thrust::gather(lifted_G.get_tails().begin(), lifted_G.get_tails().begin() + num_lifted_dir,
                   comp.begin(), lift_ct.begin());
    thrust::gather(lifted_G.get_heads().begin(), lifted_G.get_heads().begin() + num_lifted_dir,
                   comp.begin(), lift_ch.begin());

    VectorType<int> v_mask(num_lifted_dir);
    {
        const int* lt_ptr = lifted_G.get_tails_ptr();
        const int* lh_ptr = lifted_G.get_heads_ptr();
        const float* lc_ptr = lifted_G.get_costs_ptr();
        const int* lct_ptr = thrust::raw_pointer_cast(lift_ct.data());
        const int* lch_ptr = thrust::raw_pointer_cast(lift_ch.data());
        int* vm_ptr = thrust::raw_pointer_cast(v_mask.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_lifted_dir),
            [lt_ptr, lh_ptr, lc_ptr, lct_ptr, lch_ptr, vm_ptr] LCC_HOST_DEVICE (int e) {
                vm_ptr[e] = (lt_ptr[e] < lh_ptr[e] && lc_ptr[e] > 0.0f &&
                             lct_ptr[e] != lch_ptr[e]) ? 1 : 0;
            });
    }

    int num_v = thrust::reduce(v_mask.begin(), v_mask.end(), 0);
    if (num_v == 0)
        return result;

    // Gather violation data
    VectorType<int> lift_cmin(num_lifted_dir), lift_cmax(num_lifted_dir);
    {
        const int* lct_ptr = thrust::raw_pointer_cast(lift_ct.data());
        const int* lch_ptr = thrust::raw_pointer_cast(lift_ch.data());
        int* cmin_ptr = thrust::raw_pointer_cast(lift_cmin.data());
        int* cmax_ptr = thrust::raw_pointer_cast(lift_cmax.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_lifted_dir),
            [lct_ptr, lch_ptr, cmin_ptr, cmax_ptr] LCC_HOST_DEVICE (int e) {
                int a = lct_ptr[e], b = lch_ptr[e];
                cmin_ptr[e] = (a < b) ? a : b;
                cmax_ptr[e] = (a < b) ? b : a;
            });
    }

    VectorType<int> v_src(num_v), v_dst(num_v), v_idx(num_v);
    VectorType<float> v_cost(num_v);
    thrust::copy_if(lift_cmin.begin(), lift_cmin.end(), v_mask.begin(), v_src.begin(), is_one);
    thrust::copy_if(lift_cmax.begin(), lift_cmax.end(), v_mask.begin(), v_dst.begin(), is_one);

    VectorType<int> iota(num_lifted_dir);
    thrust::sequence(iota.begin(), iota.end(), 0);
    thrust::copy_if(iota.begin(), iota.end(), v_mask.begin(), v_idx.begin(), is_one);
    thrust::copy_if(lifted_G.get_costs().begin(), lifted_G.get_costs().end(),
                    v_mask.begin(), v_cost.begin(), is_one);

    if (verbose)
        std::cout << "lifted cut constraints: " << num_v
                  << " violations, Q: " << num_comp << " nodes, "
                  << num_q << " edges\n";

    // ---- Phase 4: Unique (s,t) component pairs ----

    VectorType<long long> pair_keys(num_v);
    {
        const int* vs_ptr = thrust::raw_pointer_cast(v_src.data());
        const int* vd_ptr = thrust::raw_pointer_cast(v_dst.data());
        long long* pk_ptr = thrust::raw_pointer_cast(pair_keys.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_v),
            [vs_ptr, vd_ptr, pk_ptr] LCC_HOST_DEVICE (int i) {
                pk_ptr[i] = ((long long)vs_ptr[i] << 32) | (long long)(unsigned int)vd_ptr[i];
            });
    }

    VectorType<long long> unique_keys(pair_keys);
    thrust::sort(unique_keys.begin(), unique_keys.end());
    auto new_end = thrust::unique(unique_keys.begin(), unique_keys.end());
    int num_pairs = (int)(new_end - unique_keys.begin());
    unique_keys.resize(num_pairs);

    VectorType<int> pair_s(num_pairs), pair_t(num_pairs);
    {
        const long long* uk_ptr = thrust::raw_pointer_cast(unique_keys.data());
        int* ps_ptr = thrust::raw_pointer_cast(pair_s.data());
        int* pt_ptr = thrust::raw_pointer_cast(pair_t.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_pairs),
            [uk_ptr, ps_ptr, pt_ptr] LCC_HOST_DEVICE (int i) {
                ps_ptr[i] = (int)(uk_ptr[i] >> 32);
                pt_ptr[i] = (int)(uk_ptr[i] & 0xFFFFFFFF);
            });
    }

    // ---- Phase 5: Karger trials ----
    // For each unique (s,t) pair, run M random contraction trials and keep the smallest cut.
    // Each trial uses a random permutation of Q-edges (generated via hash-based keys + sort).
    // Parallelism is across pairs within each trial round.

    const int M = std::max(10, std::min(num_comp * num_comp / 2, 100));

    VectorType<int> best_parent(num_pairs * num_comp);
    VectorType<int> best_cut(num_pairs);
    thrust::fill(best_cut.begin(), best_cut.end(), std::numeric_limits<int>::max());

    VectorType<int> cur_parent(num_pairs * num_comp);
    VectorType<int> cur_cut(num_pairs);
    VectorType<unsigned int> perm_keys_buf(num_q);
    VectorType<int> perm(num_q);

    for (int m = 0; m < M; m++)
    {
        // Generate random permutation via hash-keyed sort
        {
            const unsigned int seed = (unsigned int)m * 0x9E3779B9u + 42u;
            unsigned int* pk = thrust::raw_pointer_cast(perm_keys_buf.data());
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_q),
                [pk, seed] LCC_HOST_DEVICE (int e) {
                    pk[e] = lifted_cut_detail::hash32(seed + (unsigned int)e);
                });
        }
        thrust::sequence(perm.begin(), perm.end(), 0);
        thrust::sort_by_key(perm_keys_buf.begin(), perm_keys_buf.end(), perm.begin());

        // Run one trial for each unique pair
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_pairs),
            lifted_cut_detail::karger_trial_functor{
                thrust::raw_pointer_cast(cur_parent.data()),
                thrust::raw_pointer_cast(cur_cut.data()),
                thrust::raw_pointer_cast(perm.data()),
                thrust::raw_pointer_cast(q_src.data()),
                thrust::raw_pointer_cast(q_dst.data()),
                thrust::raw_pointer_cast(pair_s.data()),
                thrust::raw_pointer_cast(pair_t.data()),
                num_q, num_comp});

        // Update best where current trial found a smaller cut
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_pairs),
            lifted_cut_detail::update_best_functor{
                thrust::raw_pointer_cast(best_parent.data()),
                thrust::raw_pointer_cast(best_cut.data()),
                thrust::raw_pointer_cast(cur_parent.data()),
                thrust::raw_pointer_cast(cur_cut.data()),
                num_comp});

        // Early termination: stop if all best cuts are <= 1
        auto gt_one = [] LCC_HOST_DEVICE (int x) { return x > 1; };
        if (thrust::count_if(best_cut.begin(), best_cut.end(), gt_one) == 0)
            break;
    }

    // ---- Phase 6: Extract results on host ----

    thrust::host_vector<int> h_best_parent(best_parent);
    thrust::host_vector<int> h_best_cut(best_cut);
    thrust::host_vector<int> h_pair_s(pair_s);
    thrust::host_vector<int> h_pair_t(pair_t);
    thrust::host_vector<long long> h_unique_keys(unique_keys);
    thrust::host_vector<int> h_v_src(v_src);
    thrust::host_vector<int> h_v_dst(v_dst);
    thrust::host_vector<int> h_v_idx(v_idx);
    thrust::host_vector<float> h_v_cost(v_cost);
    thrust::host_vector<int> h_q_src(q_src);
    thrust::host_vector<int> h_q_dst(q_dst);
    thrust::host_vector<int> h_bt(base_G.get_tails());
    thrust::host_vector<int> h_bh(base_G.get_heads());
    thrust::host_vector<float> h_bc(base_G.get_costs());
    thrust::host_vector<int> h_comp(comp);

    // Map encoded pair keys to pair indices
    std::map<long long, int> key_to_pair;
    for (int i = 0; i < num_pairs; i++)
        key_to_pair[h_unique_keys[i]] = i;

    auto host_find = [](int* p, int x) {
        while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
        return x;
    };

    for (int vi = 0; vi < num_v; vi++)
    {
        long long key = ((long long)h_v_src[vi] << 32) | (long long)(unsigned int)h_v_dst[vi];
        int pi = key_to_pair[key];

        if (h_best_cut[pi] <= 0 || h_best_cut[pi] == std::numeric_limits<int>::max())
            continue;

        // Extract cut component pairs from the best parent array for this (s,t) pair
        int* par = h_best_parent.data() + pi * num_comp;
        int su = host_find(par, h_pair_s[pi]);
        int sv = host_find(par, h_pair_t[pi]);

        std::set<std::pair<int,int>> cut_pairs;
        for (int e = 0; e < num_q; e++)
        {
            int u = host_find(par, h_q_src[e]);
            int v = host_find(par, h_q_dst[e]);
            if ((u == su && v == sv) || (u == sv && v == su))
                cut_pairs.insert({h_q_src[e], h_q_dst[e]});
        }

        // Count undirected base edges in the cut and find min |cost|
        int actual_cut = 0;
        float min_abs = std::numeric_limits<float>::max();
        for (int e = 0; e < num_base_dir; e++)
        {
            if (h_bc[e] >= 0.0f) continue;
            int ct = h_comp[h_bt[e]], ch = h_comp[h_bh[e]];
            auto p = std::make_pair(std::min(ct, ch), std::max(ct, ch));
            if (cut_pairs.count(p) && h_bt[e] < h_bh[e])
            {
                actual_cut++;
                min_abs = std::min(min_abs, std::abs(h_bc[e]));
            }
        }

        if (actual_cut == 0) continue;

        result.push_back({h_v_idx[vi], h_v_cost[vi], std::move(cut_pairs),
                          actual_cut, min_abs});

        if (verbose)
            std::cout << "  cut (" << h_v_src[vi] << "," << h_v_dst[vi]
                      << "): base-edges=" << actual_cut << "\n";
    }

    return result;
}
