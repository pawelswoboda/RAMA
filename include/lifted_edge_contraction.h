#pragma once

#include "graph.h"
#include "connected_components.h"
#include "rama_utils.h"

#include <iostream>

#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>

#ifdef __CUDACC__
#define LEC_HOST_DEVICE __host__ __device__
#else
#define LEC_HOST_DEVICE
#endif

// For each lifted edge (u,v) with positive reparametrized cost, find the
// shortest base-graph path from u to v using only edges with non-negative
// reparametrized cost.  If such a path exists, merge all nodes along it.
// Returns (node_mapping, num_contracted_edges).
//
// Uses batched multi-source BFS: all lifted edges in a batch share a single
// level-synchronous BFS, reducing kernel launches from O(S * diameter) to
// O(diameter) per batch.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, int>
find_lifted_contraction_mapping(
    const Graph<VectorType>& base_G,
    const Graph<VectorType>& lifted_G,
    bool verbose = true)
{
    const int N = base_G.num_nodes();

    if (lifted_G.num_directed_edges() == 0 || base_G.num_directed_edges() == 0)
    {
        VectorType<int> node_mapping(N);
        thrust::sequence(node_mapping.begin(), node_mapping.end());
        return {std::move(node_mapping), 0};
    }

    const int E = (int)base_G.num_directed_edges();
    const int* base_t = base_G.get_tails_ptr();
    const int* base_h = base_G.get_heads_ptr();
    const float* base_c = base_G.get_costs_ptr();

    VectorType<int> node_mapping(N);
    thrust::sequence(node_mapping.begin(), node_mapping.end());
    int* nm_ptr = thrust::raw_pointer_cast(node_mapping.data());

    // Filter positive forward lifted edges (tail < head, cost > 0)
    const int L = (int)lifted_G.num_directed_edges();
    VectorType<int> fwd_mask(L);
    {
        const int* lt = lifted_G.get_tails_ptr();
        const int* lh = lifted_G.get_heads_ptr();
        const float* lc = lifted_G.get_costs_ptr();
        int* m = thrust::raw_pointer_cast(fwd_mask.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(L),
            [lt, lh, lc, m] LEC_HOST_DEVICE (int e) {
                m[e] = (lt[e] < lh[e] && lc[e] > 0.0f) ? 1 : 0;
            });
    }

    int S = thrust::reduce(fwd_mask.begin(), fwd_mask.end(), 0);
    if (S == 0)
        return {std::move(node_mapping), 0};

    auto is_one = [] LEC_HOST_DEVICE (int x) { return x == 1; };

    // CC prefilter: discard lifted edges whose endpoints are in different
    // connected components of the non-negative base subgraph (no path exists).
    {
        auto is_nonneg = [] LEC_HOST_DEVICE (float c) { return c >= 0.0f; };
        int num_nonneg = thrust::count_if(
            base_G.get_costs().begin(), base_G.get_costs().end(), is_nonneg);
        VectorType<int> nn_t(num_nonneg), nn_h(num_nonneg);
        thrust::copy_if(base_G.get_tails().begin(), base_G.get_tails().begin() + E,
                        base_G.get_costs().begin(), nn_t.begin(), is_nonneg);
        thrust::copy_if(base_G.get_heads().begin(), base_G.get_heads().begin() + E,
                        base_G.get_costs().begin(), nn_h.begin(), is_nonneg);

        VectorType<int> comp =
            connected_components::compute_cc<VectorType>(N, nn_t, nn_h);
        const int* comp_ptr = thrust::raw_pointer_cast(comp.data());

        // Tighten mask: require same component
        const int* lt = lifted_G.get_tails_ptr();
        const int* lh = lifted_G.get_heads_ptr();
        int* m = thrust::raw_pointer_cast(fwd_mask.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(L),
            [m, lt, lh, comp_ptr] LEC_HOST_DEVICE (int e) {
                if (m[e] == 1 && comp_ptr[lt[e]] != comp_ptr[lh[e]])
                    m[e] = 0;
            });

        const int S_before = S;
        S = thrust::reduce(fwd_mask.begin(), fwd_mask.end(), 0);
        if (verbose)
            std::cout << "  lifted BFS: " << S << "/" << S_before
                      << " positive lifted edges reachable (CC prefilter)\n";
        if (S == 0)
            return {std::move(node_mapping), 0};
    }

    VectorType<int> src_all(S), dst_all(S);
    thrust::copy_if(lifted_G.get_tails().begin(), lifted_G.get_tails().end(),
                    fwd_mask.begin(), src_all.begin(), is_one);
    thrust::copy_if(lifted_G.get_heads().begin(), lifted_G.get_heads().end(),
                    fwd_mask.begin(), dst_all.begin(), is_one);

    const int* src_all_ptr = thrust::raw_pointer_cast(src_all.data());
    const int* dst_all_ptr = thrust::raw_pointer_cast(dst_all.data());

    // Batch size: limit dist+parent arrays to ~128MB total
    const int max_batch = std::max(1,
        (int)std::min((long long)S, 16LL * 1024 * 1024 / std::max((long long)N, 1LL)));

    if (verbose)
        std::cout << "  lifted BFS: " << S << " sources, batch size " << max_batch
                  << ", " << N << " nodes, " << E << " edges\n";

    int total_path_edges = 0;

    for (int b_start = 0; b_start < S; b_start += max_batch)
    {
        const int B = std::min(max_batch, S - b_start);
        const long long BN = (long long)B * N;
        const long long BE = (long long)B * E;
        const int* b_src = src_all_ptr + b_start;
        const int* b_dst = dst_all_ptr + b_start;

        // Flat arrays: dist[s * N + node], parent[s * N + node]
        VectorType<int> dist((size_t)BN, N);
        VectorType<int> parent((size_t)BN, -1);
        int* d = thrust::raw_pointer_cast(dist.data());
        int* p = thrust::raw_pointer_cast(parent.data());

        // Initialize source nodes
        {
            const int n = N;
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [d, p, b_src, n] LEC_HOST_DEVICE (int s) {
                    const long long off = (long long)s * n + b_src[s];
                    d[off] = 0;
                    p[off] = b_src[s];
                });
        }

        // Level-synchronous BFS over all sources in batch
        for (int level = 0; level < N; ++level)
        {
            const int n = N;
            const int e_count = E;
            thrust::for_each(
                thrust::make_counting_iterator<long long>(0),
                thrust::make_counting_iterator<long long>(BE),
                [base_t, base_h, base_c, d, p, n, e_count, level]
                LEC_HOST_DEVICE (long long idx) {
                    const int e = (int)(idx % e_count);
                    if (base_c[e] < 0.0f) return;
                    const int s = (int)(idx / e_count);
                    const int t = base_t[e];
                    const int h = base_h[e];
                    const long long off = (long long)s * n;
                    if (d[off + t] == level && d[off + h] > level + 1)
                    {
                        d[off + h] = level + 1;
                        p[off + h] = t;
                    }
                });

            // All targets reached?
            {
                const int n = N;
                int unreached = thrust::count_if(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(B),
                    [d, b_dst, n] LEC_HOST_DEVICE (int s) {
                        return d[(long long)s * n + b_dst[s]] >= n;
                    });
                if (unreached == 0) break;
            }

            // Frontier empty? (no more progress possible)
            {
                const int next = level + 1;
                int frontier = thrust::count_if(
                    thrust::make_counting_iterator<long long>(0),
                    thrust::make_counting_iterator<long long>(BN),
                    [d, next] LEC_HOST_DEVICE (long long idx) {
                        return d[idx] == next;
                    });
                if (frontier == 0) break;
            }
        }

        // Sum path lengths for found paths
        {
            const int n = N;
            VectorType<int> path_lens(B);
            int* pl = thrust::raw_pointer_cast(path_lens.data());
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [pl, d, b_dst, n] LEC_HOST_DEVICE (int s) {
                    const int dv = d[(long long)s * n + b_dst[s]];
                    pl[s] = (dv < n) ? dv : 0;
                });
            total_path_edges += thrust::reduce(path_lens.begin(), path_lens.end(), 0);
        }

        // Trace paths backwards and write node_mapping
        VectorType<int> cur(B);
        int* cur_ptr = thrust::raw_pointer_cast(cur.data());

        // Initialize: cur[s] = dst[s] if found, -1 if not
        {
            const int n = N;
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [cur_ptr, d, b_dst, n] LEC_HOST_DEVICE (int s) {
                    cur_ptr[s] = (d[(long long)s * n + b_dst[s]] < n) ? b_dst[s] : -1;
                });
        }

        // Walk backwards: cur -> parent(cur), writing nm[cur] = parent
        for (int hop = 0; hop < N; ++hop)
        {
            int active = thrust::count_if(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [cur_ptr, b_src] LEC_HOST_DEVICE (int s) {
                    return cur_ptr[s] >= 0 && cur_ptr[s] != b_src[s];
                });
            if (active == 0) break;

            const int n = N;
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [cur_ptr, p, nm_ptr, b_src, n] LEC_HOST_DEVICE (int s) {
                    const int c = cur_ptr[s];
                    if (c < 0 || c == b_src[s]) return;
                    const int par = p[(long long)s * n + c];
                    nm_ptr[c] = par;
                    cur_ptr[s] = par;
                });
        }
    }

    if (total_path_edges == 0)
    {
        thrust::sequence(node_mapping.begin(), node_mapping.end());
        return {std::move(node_mapping), 0};
    }

    // Pointer jumping to flatten the mapping
    for (int iter = 0; iter < 32; ++iter)
    {
        int changed = thrust::count_if(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(N),
            [nm_ptr] LEC_HOST_DEVICE (int i) {
                return nm_ptr[i] != nm_ptr[nm_ptr[i]];
            });
        if (changed == 0) break;

        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(N),
            [nm_ptr] LEC_HOST_DEVICE (int i) {
                nm_ptr[i] = nm_ptr[nm_ptr[i]];
            });
    }

    if (verbose)
        std::cout << "lifted path contraction: " << total_path_edges
                  << " path edges contracted\n";

    return {compress_label_sequence<VectorType>(node_mapping, N - 1),
            total_path_edges};
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, int>
find_lifted_contraction_mapping<thrust::host_vector>(
    const Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&, bool);

extern template
std::tuple<thrust::device_vector<int>, int>
find_lifted_contraction_mapping<thrust::device_vector>(
    const Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&, bool);