#pragma once

#include <thrust/device_vector.h>
#include "dCOO.h"

class edge_contractions_woc
{
    public:
        edge_contractions_woc(const dCOO& A, const bool _verbose = true);
        std::tuple<thrust::device_vector<int>, int> find_contraction_mapping();

    private:
        bool check_triangles();
        bool check_quadrangles();
        bool check_pentagons();

        bool remove_mst_by_mask(thrust::device_vector<bool>& edge_valid_mask);
        void filter_by_cc();

        const int num_nodes;

        thrust::device_vector<int> mst_row_ids, mst_col_ids;
        thrust::device_vector<float> mst_data;
        thrust::device_vector<int> mst_row_offsets;
        thrust::device_vector<int> rep_row_ids, rep_col_ids;

        thrust::device_vector<int> cc_labels;
        bool verbose;
};
