#pragma once

#include <vector>
#include <tuple>
#include "dCOO.h"
#include "multicut_solver_options.h"

class iterative_rama_cuda {
    public:
        iterative_rama_cuda() {}
        iterative_rama_cuda(thrust::device_vector<int>&& i, thrust::device_vector<int>&& j, thrust::device_vector<float>&& costs, const multicut_solver_options& opts, const int device);
        std::tuple<bool, int> try_do_primal_step(int* out_node_mapping_ptr);
        void get_node_mapping(int* ptr) const;
        void set_edge_costs(const float* const ptr);
        dCOO A_; // is public to allow access to public methods of dCOO to caller of current class.
        bool try_edges_to_contract_by_maximum_matching_ = true;
    private:
        thrust::device_vector<int> sanitized_node_ids_;
        thrust::device_vector<int> node_mapping_;
        double lb_;
        multicut_solver_options opts_;
        int max_num_dual_itr_ = 0;
};

void sort_edge_list_dCOO(thrust::device_ptr<int> i, thrust::device_ptr<int> j, const int num_edges);
