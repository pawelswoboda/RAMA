#pragma once

#include <thrust/device_vector.h>
#include "dCOO.h"

class frontier
{
    public:
        frontier() {}
        
        frontier(const thrust::device_vector<int>& seeds) : 
            nodes(seeds), 
            parent_nodes(seeds.size(), -1), 
            rep_edges(seeds.size()), 
            bottleneck_indices(seeds.size(), 0), 
            bottleneck_values(seeds.size(), std::numeric_limits<float>::max())
        {
            thrust::sequence(rep_edges.begin(), rep_edges.end());
        }

        frontier(thrust::device_vector<int>&& _nodes, thrust::device_vector<int>&& _parent_nodes, thrust::device_vector<int>&& _rep_edges,
            thrust::device_vector<int>&& _bottleneck_indices, thrust::device_vector<float>&& _bottleneck_values) : 
        nodes(std::move(_nodes)), 
        parent_nodes(std::move(_parent_nodes)), 
        rep_edges(std::move(_rep_edges)),
        bottleneck_indices(std::move(_bottleneck_indices)), 
        bottleneck_values(std::move(_bottleneck_values)) { }

        // Remove all expansions done by 'rep_edges_to_remove'
        void filter_by_rep_edges(const thrust::device_vector<int>& rep_edges_to_remove);
        
        // Keep all bottleneck indices in 'mst_edges_to_keep'
        void filter_by_mst_edges(const thrust::device_vector<int>& mst_edges_to_keep);

        // Change indices of get_bottleneck_indices in valid_mst_indices to 1:size(valid_mst_indices) - 1. 
        // Rest will be removed by 'filter_by_rep_edges' operation.
        void reassign_mst_indices(const thrust::device_vector<int>& valid_mst_indices, const int prev_mst_size);

        thrust::device_vector<int>& get_nodes() { return nodes; }
        thrust::device_vector<int>& get_parent_nodes() { return parent_nodes; }
        thrust::device_vector<int>& get_rep_edges() { return rep_edges; }
        thrust::device_vector<int>& get_bottleneck_indices() { return bottleneck_indices; }
        thrust::device_vector<float>& get_bottleneck_values() { return bottleneck_values; }

        const thrust::device_vector<int>& get_nodes() const { return nodes; }
        const thrust::device_vector<int>& get_parent_nodes() const { return parent_nodes; }
        const thrust::device_vector<int>& get_rep_edges() const { return rep_edges; }
        const thrust::device_vector<int>& get_bottleneck_indices() const { return bottleneck_indices; }
        const thrust::device_vector<float>& get_bottleneck_values() const { return bottleneck_values; }

    private:
        void restrict_to_indices(const thrust::device_vector<int>& indices_to_keep);

        thrust::device_vector<int> nodes;
        thrust::device_vector<int> parent_nodes;
        thrust::device_vector<int> rep_edges;
        thrust::device_vector<int> bottleneck_indices;
        thrust::device_vector<float> bottleneck_values;
};

class edge_contractions_woc_thrust
{
    public:
        edge_contractions_woc_thrust(const dCOO& A);
        edge_contractions_woc_thrust(const int _num_nodes,
                            const thrust::device_vector<int>&& _mst_row_ids, 
                            const thrust::device_vector<int>&& _mst_col_ids, 
                            const thrust::device_vector<float>&& _mst_data, 
                            const thrust::device_vector<int>&& _rep_row_ids,
                            const thrust::device_vector<int>&& _rep_col_ids,
                            const thrust::device_vector<int>&& _cc_labels) : 
        num_nodes(_num_nodes), mst_row_ids(std::move(_mst_row_ids)), mst_col_ids(std::move(_mst_col_ids)), mst_data(std::move(_mst_data)),
        rep_row_ids(std::move(_rep_row_ids)), rep_col_ids(std::move(_rep_col_ids)), cc_labels(std::move(_cc_labels)) { }

        std::tuple<thrust::device_vector<int>, int> find_contraction_mapping();

    private:
        void expand_frontier(frontier& f);
        bool filter_cycles();
        int filter_by_cc();

        const int num_nodes;

        frontier row_frontier, col_frontier;
        thrust::device_vector<int> mst_row_ids, mst_col_ids;
        thrust::device_vector<float> mst_data;
        thrust::device_vector<int> rep_row_ids, rep_col_ids;

        thrust::device_vector<int> cc_labels;
};
