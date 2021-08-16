#pragma once

#include"dCOO.h"
#include <thrust/device_vector.h>

typedef std::vector<thrust::device_vector<int>> device_vectors;
class conflicted_cycles {
    public:
        conflicted_cycles(const dCOO&, const int);

        std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> enumerate_conflicted_cycles();

    private:
        std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> 
            expand_frontier(const thrust::device_vector<int>&, const thrust::device_vector<int>&, const device_vectors&, const device_vectors&);

    std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> filter_self_intersecting_paths(
        const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<int>&,
        const thrust::device_vector<int>&, const thrust::device_vector<int>&, const device_vectors&, const device_vectors&);
        
        void expand_merge_triangulate(device_vectors&, device_vectors&, device_vectors&);
        void expand_merge_triangulate(const thrust::device_vector<int>&, const thrust::device_vector<int>&,
            device_vectors&, device_vectors&, device_vectors&);

        std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> merge_paths() const;

        void add_to_triangulation(const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<int>&);
        void triangulate_intersection(const thrust::device_vector<int>&, const thrust::device_vector<int>&);

        thrust::device_vector<int> tri_v1;
        thrust::device_vector<int> tri_v2;
        thrust::device_vector<int> tri_v3;

        void init(const dCOO& A);

        dCOO A_pos_symm;
        int max_cycle_length;

        thrust::device_vector<int> row_ids_rep;
        thrust::device_vector<int> col_ids_rep;

        device_vectors up_src_offsets, down_src_offsets;
        device_vectors up_dst, down_dst;
        device_vectors up_dst_parent_edges, down_dst_parent_edges;
};