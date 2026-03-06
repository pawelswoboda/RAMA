#pragma once

#include "graph.h"
#include "connected_components.h"
#include "rama_utils.h"

#include <algorithm>
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
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/tuple.h>

#ifdef __CUDACC__
#define LCC_HOST_DEVICE __host__ __device__
#else
#define LCC_HOST_DEVICE
#endif

// GPU-friendly representation of multiple cut factors in CSR format.
// Each factor groups one cut (a set of base edges) with all lifted edges
// separated by that cut. Edge indices refer to directed (tail < head) positions
// in the respective graph's edge list.
template<template<typename> class VectorType>
struct LiftedCutFactors {
    int num_factors = 0;

    // Base edges per factor (CSR)
    VectorType<int> base_offsets;     // [num_factors + 1]
    VectorType<int> base_edge_idx;    // flat: directed base edge indices (tail < head)

    // Lifted edges per factor (CSR)
    VectorType<int> lifted_offsets;   // [num_factors + 1]
    VectorType<int> lifted_edge_idx;  // flat: directed lifted edge indices (tail < head)
};

namespace lifted_cut_detail {

LCC_HOST_DEVICE inline int uf_find(int* parent, int x)
{
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}

LCC_HOST_DEVICE inline int uf_find_readonly(const int* parent, int x)
{
    while (parent[x] != x) x = parent[x];
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

// Find violated lifted cut constraints and build GPU-friendly cut factors.
//
// Contracts base edges with cost > -tau into super-nodes. The remaining edges
// (cost <= -tau, strongly repulsive) form the contracted graph Q.
// For each lifted edge with cost >= tau whose endpoints are in different
// super-nodes and reachable through Q, finds a min-cut via Karger's algorithm.
//
// At tau=0: contracts cost > 0, Q has cost <= 0 — close to the standard split.
// Cascading from high tau to low finds strongest violations first.
//
// Returns LiftedCutFactors in CSR format.
template<template<typename> class VectorType>
LiftedCutFactors<VectorType> find_lifted_cut_constraints(
    const Graph<VectorType>& base_G,
    const Graph<VectorType>& lifted_G,
    bool verbose = false,
    float tau = 0.0f)
{
    LiftedCutFactors<VectorType> result;

    if (lifted_G.num_directed_edges() == 0)
        return result;

    const int num_nodes = base_G.num_nodes();
    const int num_base_dir = (int)base_G.num_directed_edges();
    const int num_lifted_dir = (int)lifted_G.num_directed_edges();

    // ---- Phase 1: Contract base edges with cost > -tau ----
    // Edges with cost > -tau (attractive + neutral) are contracted into super-nodes.
    // Only strongly negative edges (cost <= -tau) remain between super-nodes.

    const float tau_val = tau;
    const float neg_tau = -tau;
    auto is_contracted = [neg_tau] LCC_HOST_DEVICE (float c) { return c > neg_tau; };
    int num_contracted = thrust::count_if(
        base_G.get_costs().begin(), base_G.get_costs().end(), is_contracted);

    VectorType<int> attr_t(num_contracted), attr_h(num_contracted);
    thrust::copy_if(base_G.get_tails().begin(), base_G.get_tails().begin() + num_base_dir,
                    base_G.get_costs().begin(), attr_t.begin(), is_contracted);
    thrust::copy_if(base_G.get_heads().begin(), base_G.get_heads().begin() + num_base_dir,
                    base_G.get_costs().begin(), attr_h.begin(), is_contracted);

    VectorType<int> comp =
        connected_components::compute_cc<VectorType>(num_nodes, attr_t, attr_h);

    int max_label = thrust::reduce(comp.begin(), comp.end(), 0, thrust::maximum<int>());
    comp = compress_label_sequence<VectorType>(comp, max_label);
    int num_comp = thrust::reduce(comp.begin(), comp.end(), 0, thrust::maximum<int>()) + 1;

    if (num_comp <= 1)
        return result;

    // ---- Phase 2: Build quotient graph Q ----
    // One Q-edge per undirected strongly repulsive inter-component base edge (cost <= -tau).

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
            [bc_ptr, ct_ptr, ch_ptr, mask_ptr, neg_tau] LCC_HOST_DEVICE (int e) {
                mask_ptr[e] = (bc_ptr[e] <= neg_tau && ct_ptr[e] < ch_ptr[e]) ? 1 : 0;
            });
    }

    int num_q_all = thrust::reduce(q_mask.begin(), q_mask.end(), 0);
    if (num_q_all == 0)
        return result;

    auto is_one = [] LCC_HOST_DEVICE (int x) { return x == 1; };

    // Compute internal cost threshold = 0.1 * mean(|cost|) of Q-edges.
    // Further tighten Q if this is stricter than -tau.
    float internal_tau;
    {
        VectorType<float> q_costs_tmp(num_q_all);
        thrust::copy_if(base_G.get_costs().begin(), base_G.get_costs().begin() + num_base_dir,
                        q_mask.begin(), q_costs_tmp.begin(), is_one);
        float sum_abs = -thrust::reduce(q_costs_tmp.begin(), q_costs_tmp.end(), 0.0f);
        internal_tau = 0.1f * sum_abs / num_q_all;

        const float q_thresh = std::min(neg_tau, -internal_tau);
        if (q_thresh < neg_tau)
        {
            const float* bc_ptr2 = base_G.get_costs_ptr();
            const int* ct_ptr2 = thrust::raw_pointer_cast(base_ct.data());
            const int* ch_ptr2 = thrust::raw_pointer_cast(base_ch.data());
            int* mask_ptr2 = thrust::raw_pointer_cast(q_mask.data());
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_base_dir),
                [bc_ptr2, ct_ptr2, ch_ptr2, mask_ptr2, q_thresh] LCC_HOST_DEVICE (int e) {
                    mask_ptr2[e] = (bc_ptr2[e] <= q_thresh && ct_ptr2[e] < ch_ptr2[e]) ? 1 : 0;
                });
        }
    }

    int num_q = thrust::reduce(q_mask.begin(), q_mask.end(), 0);
    if (num_q == 0)
        return result;

    VectorType<int> q_src(num_q), q_dst(num_q);
    VectorType<float> q_costs(num_q);
    thrust::copy_if(base_ct.begin(), base_ct.end(), q_mask.begin(), q_src.begin(), is_one);
    thrust::copy_if(base_ch.begin(), base_ch.end(), q_mask.begin(), q_dst.begin(), is_one);
    thrust::copy_if(base_G.get_costs().begin(), base_G.get_costs().begin() + num_base_dir,
                    q_mask.begin(), q_costs.begin(), is_one);

    // ---- Phase 2b: Q-CC prefilter ----
    // Compute CC of Q to filter violations to pairs reachable through strongly negative edges.
    // Q edges are undirected: add both directions for CC computation.

    VectorType<int> q_cc_t(2 * num_q), q_cc_h(2 * num_q);
    thrust::copy(q_src.begin(), q_src.end(), q_cc_t.begin());
    thrust::copy(q_dst.begin(), q_dst.end(), q_cc_t.begin() + num_q);
    thrust::copy(q_dst.begin(), q_dst.end(), q_cc_h.begin());
    thrust::copy(q_src.begin(), q_src.end(), q_cc_h.begin() + num_q);

    VectorType<int> q_comp =
        connected_components::compute_cc<VectorType>(num_comp, q_cc_t, q_cc_h);

    // ---- Phase 3: Find violations ----
    // Lifted edges with cost > tau, endpoints in different CCs (strong base subgraph),
    // and endpoints in the same Q-CC (reachable through strongly negative edges).

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
        const int* qc_ptr = thrust::raw_pointer_cast(q_comp.data());
        int* vm_ptr = thrust::raw_pointer_cast(v_mask.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_lifted_dir),
            [lt_ptr, lh_ptr, lc_ptr, lct_ptr, lch_ptr, qc_ptr, vm_ptr, tau_val] LCC_HOST_DEVICE (int e) {
                int ct = lct_ptr[e], ch = lch_ptr[e];
                vm_ptr[e] = (lt_ptr[e] < lh_ptr[e] && lc_ptr[e] >= tau_val &&
                             ct != ch && qc_ptr[ct] == qc_ptr[ch]) ? 1 : 0;
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
    thrust::copy_if(lift_cmin.begin(), lift_cmin.end(), v_mask.begin(), v_src.begin(), is_one);
    thrust::copy_if(lift_cmax.begin(), lift_cmax.end(), v_mask.begin(), v_dst.begin(), is_one);

    VectorType<int> iota(num_lifted_dir);
    thrust::sequence(iota.begin(), iota.end(), 0);
    thrust::copy_if(iota.begin(), iota.end(), v_mask.begin(), v_idx.begin(), is_one);

    // ---- Phase 4: Unique (s,t) component pairs, filtered by sum(L) > τ ----

    // Gather violated lifted edge costs
    VectorType<float> v_costs(num_v);
    thrust::gather(v_idx.begin(), v_idx.end(), lifted_G.get_costs().begin(), v_costs.begin());

    // Build pair keys for each violation
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

    // Sort violations by pair key, then reduce to get sum(L) per pair
    VectorType<long long> sorted_pair_keys(pair_keys);
    VectorType<float> sorted_v_costs(v_costs);
    thrust::sort_by_key(sorted_pair_keys.begin(), sorted_pair_keys.end(), sorted_v_costs.begin());

    VectorType<long long> reduced_keys(num_v);
    VectorType<float> reduced_sums(num_v);
    auto reduce_end_pairs = thrust::reduce_by_key(
        sorted_pair_keys.begin(), sorted_pair_keys.end(),
        sorted_v_costs.begin(),
        reduced_keys.begin(), reduced_sums.begin());
    int num_pairs_all = (int)std::distance(reduced_keys.begin(), reduce_end_pairs.first);
    reduced_keys.resize(num_pairs_all);
    reduced_sums.resize(num_pairs_all);

    // Filter to pairs with sum(L) > internal_tau
    VectorType<long long> unique_keys(num_pairs_all);
    {
        auto in_first = thrust::make_zip_iterator(
            thrust::make_tuple(reduced_keys.begin(), reduced_sums.begin()));
        auto in_last = thrust::make_zip_iterator(
            thrust::make_tuple(reduced_keys.end(), reduced_sums.end()));
        auto out_first = thrust::make_zip_iterator(
            thrust::make_tuple(unique_keys.begin(), thrust::make_discard_iterator()));
        auto filt_end = thrust::copy_if(in_first, in_last, reduced_sums.begin(), out_first,
            [internal_tau] LCC_HOST_DEVICE (float s) { return s > internal_tau; });
        int num_kept = (int)std::distance(
            thrust::make_zip_iterator(thrust::make_tuple(unique_keys.begin(), thrust::make_discard_iterator())),
            filt_end);
        unique_keys.resize(num_kept);
    }

    int num_pairs = (int)unique_keys.size();
    if (num_pairs == 0)
        return result;

    // Filter violations to only those belonging to surviving pairs
    {
        const long long* uk_ptr = thrust::raw_pointer_cast(unique_keys.data());
        const long long* pk_ptr = thrust::raw_pointer_cast(pair_keys.data());
        int np = num_pairs;
        VectorType<int> keep_mask(num_v);
        int* km_ptr = thrust::raw_pointer_cast(keep_mask.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_v),
            [pk_ptr, uk_ptr, km_ptr, np] LCC_HOST_DEVICE (int i) {
                long long key = pk_ptr[i];
                int lo = 0, hi = np;
                while (lo < hi) { int mid = (lo + hi) / 2; if (uk_ptr[mid] < key) lo = mid + 1; else hi = mid; }
                km_ptr[i] = (lo < np && uk_ptr[lo] == key) ? 1 : 0;
            });

        int new_num_v = thrust::reduce(keep_mask.begin(), keep_mask.end(), 0);
        VectorType<int> new_v_src(new_num_v), new_v_dst(new_num_v), new_v_idx(new_num_v);
        thrust::copy_if(v_src.begin(), v_src.end(), keep_mask.begin(), new_v_src.begin(), is_one);
        thrust::copy_if(v_dst.begin(), v_dst.end(), keep_mask.begin(), new_v_dst.begin(), is_one);
        thrust::copy_if(v_idx.begin(), v_idx.end(), keep_mask.begin(), new_v_idx.begin(), is_one);
        v_src = std::move(new_v_src);
        v_dst = std::move(new_v_dst);
        v_idx = std::move(new_v_idx);
        num_v = new_num_v;
    }

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

    if (verbose)
        std::cout << "cut factor search: tau=" << tau
                  << ", num_comp=" << num_comp
                  << ", Q-edges " << num_q_all << " -> " << num_q
                  << ", pairs " << num_pairs_all << " -> " << num_pairs
                  << ", violations " << num_v << "\n";

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
    VectorType<float> perm_keys_buf(num_q);
    VectorType<int> perm(num_q);

    for (int m = 0; m < M; m++)
    {
        // Generate cost-weighted random permutation via hash-keyed sort.
        // Key = hash * |cost|: strongly negative edges get large keys,
        // are sorted late, and thus contracted late — surviving into the cut.
        {
            const unsigned int seed = (unsigned int)m * 0x9E3779B9u + 42u;
            float* pk = thrust::raw_pointer_cast(perm_keys_buf.data());
            const float* qc = thrust::raw_pointer_cast(q_costs.data());
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_q),
                [pk, qc, seed] LCC_HOST_DEVICE (int e) {
                    float u = (float)lifted_cut_detail::hash32(seed + (unsigned int)e);
                    pk[e] = u * (-qc[e]);
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

    // ---- Phase 6: Build LiftedCutFactors on GPU ----

    // 6a: Collect strongly repulsive forward inter-component base edge indices.
    //     These are the base edges that CAN participate in cuts (cost <= -tau).
    VectorType<int> nfb_mask(num_base_dir);
    {
        const float* bc_ptr = base_G.get_costs_ptr();
        const int* bt_ptr = base_G.get_tails_ptr();
        const int* bh_ptr = base_G.get_heads_ptr();
        const int* ct_ptr = thrust::raw_pointer_cast(base_ct.data());
        const int* ch_ptr = thrust::raw_pointer_cast(base_ch.data());
        int* m_ptr = thrust::raw_pointer_cast(nfb_mask.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_base_dir),
            [bc_ptr, bt_ptr, bh_ptr, ct_ptr, ch_ptr, m_ptr, neg_tau] LCC_HOST_DEVICE (int e) {
                m_ptr[e] = (bc_ptr[e] <= neg_tau && bt_ptr[e] < bh_ptr[e] &&
                            ct_ptr[e] != ch_ptr[e]) ? 1 : 0;
            });
    }

    int num_nfb = thrust::reduce(nfb_mask.begin(), nfb_mask.end(), 0);
    if (num_nfb == 0)
        return result;

    VectorType<int> nfb_idx(num_nfb);
    {
        VectorType<int> seq(num_base_dir);
        thrust::sequence(seq.begin(), seq.end());
        thrust::copy_if(seq.begin(), seq.end(), nfb_mask.begin(), nfb_idx.begin(), is_one);
    }

    // Gather component ids for the negative forward base edges
    VectorType<int> nfb_ct(num_nfb), nfb_ch(num_nfb);
    thrust::gather(nfb_idx.begin(), nfb_idx.end(), base_ct.begin(), nfb_ct.begin());
    thrust::gather(nfb_idx.begin(), nfb_idx.end(), base_ch.begin(), nfb_ch.begin());

    // 6b: Compute partition sides for each (pair, component node).
    //     side = 0 if on same side as s, 1 if on same side as t.
    VectorType<int> sides(num_pairs * num_comp);
    {
        const int* bp = thrust::raw_pointer_cast(best_parent.data());
        const int* ps = thrust::raw_pointer_cast(pair_s.data());
        int* s_ptr = thrust::raw_pointer_cast(sides.data());
        int nc = num_comp;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_pairs * num_comp),
            [bp, ps, s_ptr, nc] LCC_HOST_DEVICE (int idx) {
                int pi = idx / nc;
                int node = idx % nc;
                const int* parent = bp + pi * nc;
                int root_s = lifted_cut_detail::uf_find_readonly(parent, ps[pi]);
                int root_n = lifted_cut_detail::uf_find_readonly(parent, node);
                s_ptr[idx] = (root_n == root_s) ? 0 : 1;
            });
    }

    // 6c: For each (pair, neg_fwd_base_edge), check if it crosses the partition.
    //     Parallel over num_pairs * num_nfb.
    int total_pxb = num_pairs * num_nfb;
    VectorType<int> cross_flags(total_pxb);
    {
        const int* s_ptr = thrust::raw_pointer_cast(sides.data());
        const int* ct_ptr = thrust::raw_pointer_cast(nfb_ct.data());
        const int* ch_ptr = thrust::raw_pointer_cast(nfb_ch.data());
        const int* bc_ptr = thrust::raw_pointer_cast(best_cut.data());
        int* cf_ptr = thrust::raw_pointer_cast(cross_flags.data());
        int nc = num_comp;
        int nb = num_nfb;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(total_pxb),
            [s_ptr, ct_ptr, ch_ptr, bc_ptr, cf_ptr, nc, nb] LCC_HOST_DEVICE (int idx) {
                int pi = idx / nb;
                int bi = idx % nb;
                // Skip pairs with no valid Karger result
                if (bc_ptr[pi] <= 0 || bc_ptr[pi] >= 0x7FFFFFFF) { cf_ptr[idx] = 0; return; }
                int side_tail = s_ptr[pi * nc + ct_ptr[bi]];
                int side_head = s_ptr[pi * nc + ch_ptr[bi]];
                cf_ptr[idx] = (side_tail != side_head) ? 1 : 0;
            });
    }

    int num_cross = thrust::reduce(cross_flags.begin(), cross_flags.end(), 0);
    if (num_cross == 0)
        return result;

    // Extract crossing (pair_idx, base_edge_idx) pairs
    VectorType<int> cross_pair(num_cross), cross_base(num_cross);
    {
        VectorType<int> flat_pair(total_pxb), flat_base(total_pxb);
        int nb = num_nfb;
        const int* nfb_ptr = thrust::raw_pointer_cast(nfb_idx.data());
        int* fp_ptr = thrust::raw_pointer_cast(flat_pair.data());
        int* fb_ptr = thrust::raw_pointer_cast(flat_base.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(total_pxb),
            [fp_ptr, fb_ptr, nfb_ptr, nb] LCC_HOST_DEVICE (int idx) {
                fp_ptr[idx] = idx / nb;
                fb_ptr[idx] = nfb_ptr[idx % nb];
            });

        auto in_first = thrust::make_zip_iterator(thrust::make_tuple(flat_pair.begin(), flat_base.begin()));
        auto out_first = thrust::make_zip_iterator(thrust::make_tuple(cross_pair.begin(), cross_base.begin()));
        thrust::copy_if(in_first, in_first + total_pxb, cross_flags.begin(), out_first, is_one);
    }

    // 6d: Sort by (pair, base_edge), deduplicate.
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(cross_pair.begin(), cross_base.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(cross_pair.end(), cross_base.end()));
        thrust::sort(first, last);
        auto new_last = thrust::unique(first, last);
        int n = (int)std::distance(first, new_last);
        cross_pair.resize(n);
        cross_base.resize(n);
        num_cross = n;
    }

    // 6e: Build base CSR from (pair_idx, base_edge_idx) pairs.
    //     reduce_by_key on pair_idx to get counts, then scan for offsets.
    VectorType<int> base_factor_ids(num_cross), base_counts(num_cross);
    int num_base_factors;
    {
        auto end = thrust::reduce_by_key(
            cross_pair.begin(), cross_pair.end(),
            thrust::make_constant_iterator(1),
            base_factor_ids.begin(), base_counts.begin());
        num_base_factors = (int)std::distance(base_factor_ids.begin(), end.first);
        base_factor_ids.resize(num_base_factors);
        base_counts.resize(num_base_factors);
    }

    VectorType<int> base_offsets(num_base_factors + 1);
    base_offsets[0] = 0;
    thrust::inclusive_scan(base_counts.begin(), base_counts.end(), base_offsets.begin() + 1);

    // 6f: Map violations to pair indices using binary search in unique_keys.
    VectorType<int> v_pair(num_v);
    {
        const long long* uk_ptr = thrust::raw_pointer_cast(unique_keys.data());
        const int* vs_ptr = thrust::raw_pointer_cast(v_src.data());
        const int* vd_ptr = thrust::raw_pointer_cast(v_dst.data());
        int* vp_ptr = thrust::raw_pointer_cast(v_pair.data());
        int np = num_pairs;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_v),
            [uk_ptr, vs_ptr, vd_ptr, vp_ptr, np] LCC_HOST_DEVICE (int vi) {
                long long key = ((long long)vs_ptr[vi] << 32) |
                                (long long)(unsigned int)vd_ptr[vi];
                int lo = 0, hi = np;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (uk_ptr[mid] < key) lo = mid + 1;
                    else hi = mid;
                }
                vp_ptr[vi] = lo;
            });
    }

    // Map violation pair indices to factor indices using binary search in base_factor_ids.
    // A violation belongs to a factor only if its pair has base edges in a cut.
    VectorType<int> v_factor(num_v);
    {
        const int* bf_ptr = thrust::raw_pointer_cast(base_factor_ids.data());
        const int* vp_ptr = thrust::raw_pointer_cast(v_pair.data());
        int* vf_ptr = thrust::raw_pointer_cast(v_factor.data());
        int nbf = num_base_factors;
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_v),
            [bf_ptr, vp_ptr, vf_ptr, nbf] LCC_HOST_DEVICE (int vi) {
                int pair = vp_ptr[vi];
                int lo = 0, hi = nbf;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (bf_ptr[mid] < pair) lo = mid + 1;
                    else hi = mid;
                }
                vf_ptr[vi] = (lo < nbf && bf_ptr[lo] == pair) ? lo : -1;
            });
    }

    // Filter violations with valid factor assignment, sort by factor
    VectorType<int> valid_v_mask(num_v);
    thrust::transform(v_factor.begin(), v_factor.end(), valid_v_mask.begin(),
        [] LCC_HOST_DEVICE (int f) { return f >= 0 ? 1 : 0; });
    int num_valid_v = thrust::reduce(valid_v_mask.begin(), valid_v_mask.end(), 0);

    if (num_valid_v == 0)
        return result;

    VectorType<int> vv_factor(num_valid_v), vv_idx(num_valid_v);
    thrust::copy_if(v_factor.begin(), v_factor.end(), valid_v_mask.begin(), vv_factor.begin(), is_one);
    thrust::copy_if(v_idx.begin(), v_idx.end(), valid_v_mask.begin(), vv_idx.begin(), is_one);

    // Sort violations by factor index
    thrust::sort_by_key(vv_factor.begin(), vv_factor.end(), vv_idx.begin());

    // Build lifted CSR: compute counts per factor via reduce_by_key
    VectorType<int> lifted_factor_ids(num_valid_v), lifted_counts(num_valid_v);
    int num_lifted_factors;
    {
        auto end = thrust::reduce_by_key(
            vv_factor.begin(), vv_factor.end(),
            thrust::make_constant_iterator(1),
            lifted_factor_ids.begin(), lifted_counts.begin());
        num_lifted_factors = (int)std::distance(lifted_factor_ids.begin(), end.first);
        lifted_factor_ids.resize(num_lifted_factors);
        lifted_counts.resize(num_lifted_factors);
    }

    // 6g: Intersect base and lifted factor sets to get final factors.
    //     A valid factor must have both base edges and lifted edges.
    VectorType<int> final_factor_ids(std::min(num_base_factors, num_lifted_factors));
    {
        auto end = thrust::set_intersection(
            base_factor_ids.begin(), base_factor_ids.end(),
            lifted_factor_ids.begin(), lifted_factor_ids.end(),
            final_factor_ids.begin());
        int n = (int)std::distance(final_factor_ids.begin(), end);
        final_factor_ids.resize(n);
    }

    int num_final = (int)final_factor_ids.size();
    if (num_final == 0)
        return result;

    // 6h: Extract final base CSR.
    //     For each final factor, gather the base edges from cross_base.
    VectorType<int> final_base_offsets(num_final + 1);
    {
        // Map final factor ids to positions in base_factor_ids via lower_bound
        VectorType<int> base_pos(num_final);
        thrust::lower_bound(base_factor_ids.begin(), base_factor_ids.end(),
                            final_factor_ids.begin(), final_factor_ids.end(),
                            base_pos.begin());

        // Gather base offsets for final factors
        VectorType<int> starts(num_final), ends(num_final);
        thrust::gather(base_pos.begin(), base_pos.end(), base_offsets.begin(), starts.begin());
        {
            VectorType<int> pos_plus1(num_final);
            thrust::transform(base_pos.begin(), base_pos.end(),
                              thrust::make_constant_iterator(1),
                              pos_plus1.begin(), thrust::plus<int>());
            thrust::gather(pos_plus1.begin(), pos_plus1.end(), base_offsets.begin(), ends.begin());
        }

        // Compute sizes and final offsets
        VectorType<int> sizes(num_final);
        thrust::transform(ends.begin(), ends.end(), starts.begin(), sizes.begin(), thrust::minus<int>());
        final_base_offsets[0] = 0;
        thrust::inclusive_scan(sizes.begin(), sizes.end(), final_base_offsets.begin() + 1);

        // Gather base edge indices
        int total_base = final_base_offsets[num_final];
        VectorType<int> final_base_idx(total_base);

        const int* s_ptr = thrust::raw_pointer_cast(starts.data());
        const int* fbo_ptr = thrust::raw_pointer_cast(final_base_offsets.data());
        const int* cb_ptr = thrust::raw_pointer_cast(cross_base.data());
        int* out_ptr = thrust::raw_pointer_cast(final_base_idx.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_final),
            [s_ptr, fbo_ptr, cb_ptr, out_ptr] LCC_HOST_DEVICE (int f) {
                int src = s_ptr[f];
                int dst = fbo_ptr[f];
                int len = fbo_ptr[f + 1] - dst;
                for (int k = 0; k < len; k++)
                    out_ptr[dst + k] = cb_ptr[src + k];
            });

        result.base_offsets = std::move(final_base_offsets);
        result.base_edge_idx = std::move(final_base_idx);
    }

    // 6i: Extract final lifted CSR.
    VectorType<int> final_lifted_offsets(num_final + 1);
    {
        VectorType<int> lifted_pos(num_final);
        thrust::lower_bound(lifted_factor_ids.begin(), lifted_factor_ids.end(),
                            final_factor_ids.begin(), final_factor_ids.end(),
                            lifted_pos.begin());

        // Build lifted offsets from lifted_counts via the same pattern
        VectorType<int> lifted_off_full(num_lifted_factors + 1);
        lifted_off_full[0] = 0;
        thrust::inclusive_scan(lifted_counts.begin(), lifted_counts.end(),
                               lifted_off_full.begin() + 1);

        VectorType<int> starts(num_final), ends(num_final);
        thrust::gather(lifted_pos.begin(), lifted_pos.end(), lifted_off_full.begin(), starts.begin());
        {
            VectorType<int> pos_plus1(num_final);
            thrust::transform(lifted_pos.begin(), lifted_pos.end(),
                              thrust::make_constant_iterator(1),
                              pos_plus1.begin(), thrust::plus<int>());
            thrust::gather(pos_plus1.begin(), pos_plus1.end(), lifted_off_full.begin(), ends.begin());
        }

        VectorType<int> sizes(num_final);
        thrust::transform(ends.begin(), ends.end(), starts.begin(), sizes.begin(), thrust::minus<int>());
        final_lifted_offsets[0] = 0;
        thrust::inclusive_scan(sizes.begin(), sizes.end(), final_lifted_offsets.begin() + 1);

        int total_lifted = final_lifted_offsets[num_final];
        VectorType<int> final_lifted_idx(total_lifted);

        const int* s_ptr = thrust::raw_pointer_cast(starts.data());
        const int* flo_ptr = thrust::raw_pointer_cast(final_lifted_offsets.data());
        const int* vi_ptr = thrust::raw_pointer_cast(vv_idx.data());
        int* out_ptr = thrust::raw_pointer_cast(final_lifted_idx.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_final),
            [s_ptr, flo_ptr, vi_ptr, out_ptr] LCC_HOST_DEVICE (int f) {
                int src = s_ptr[f];
                int dst = flo_ptr[f];
                int len = flo_ptr[f + 1] - dst;
                for (int k = 0; k < len; k++)
                    out_ptr[dst + k] = vi_ptr[src + k];
            });

        result.lifted_offsets = std::move(final_lifted_offsets);
        result.lifted_edge_idx = std::move(final_lifted_idx);
    }

    result.num_factors = num_final;

    if (verbose)
        std::cout << "lifted cut factors: " << num_final << " factors, "
                  << result.base_edge_idx.size() << " base edges, "
                  << result.lifted_edge_idx.size() << " lifted edges\n";

    return result;
}

// Explicit instantiation declarations.
extern template
LiftedCutFactors<thrust::host_vector>
find_lifted_cut_constraints<thrust::host_vector>(
    const Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&, bool, float);

extern template
LiftedCutFactors<thrust::device_vector>
find_lifted_cut_constraints<thrust::device_vector>(
    const Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&, bool, float);
