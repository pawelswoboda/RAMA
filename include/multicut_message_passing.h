#pragma once
#include <thrust/device_vector.h>
#include"dCOO.h"

class multicut_message_passing {
    public:
        multicut_message_passing(
                const dCOO& A,
                thrust::device_vector<int>&& _t1,
                thrust::device_vector<int>&& _t2,
                thrust::device_vector<int>&& _t3,
                const bool verbose = true
                );

        void send_messages_to_triplets();
        void send_messages_to_edges();

        double lower_bound();

        void iteration(const bool use_nn);

        std::tuple<const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&>
            reparametrized_edge_costs() const;

        std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_triangles() const {
            std::vector<int> h_t1(t1.size()), h_t2(t2.size()), h_t3(t3.size());
            thrust::copy(t1.begin(), t1.end(), h_t1.begin());
            thrust::copy(t2.begin(), t2.end(), h_t2.begin());
            thrust::copy(t3.begin(), t3.end(), h_t3.begin());
            return {h_t1, h_t2, h_t3};
        }

        std::vector<float> get_lagrange_multipliers() const {
            std::vector<float> h_lagrange(edge_costs.size());
            thrust::copy(edge_costs.begin(), edge_costs.end(), h_lagrange.begin());
            return h_lagrange;
        }

        std::tuple<std::vector<int>, std::vector<int>, std::vector<float>> get_edges() const {
            std::vector<int> h_i(i.size()), h_j(j.size());
            std::vector<float> h_cost(edge_costs.size());
            thrust::copy(i.begin(), i.end(), h_i.begin());
            thrust::copy(j.begin(), j.end(), h_j.begin());
            thrust::copy(edge_costs.begin(), edge_costs.end(), h_cost.begin());
            return {h_i, h_j, h_cost};
        }

        void update_lagrange_via_nn();  


    private:
        void compute_triangle_edge_correspondence(const thrust::device_vector<int>&, const thrust::device_vector<int>&, 
            thrust::device_vector<int>&, thrust::device_vector<int>&);

        double edge_lower_bound();
        double triangle_lower_bound();

        thrust::device_vector<int> i;
        thrust::device_vector<int> j;

        thrust::device_vector<int> t1;
        thrust::device_vector<int> t2;
        thrust::device_vector<int> t3;

        thrust::device_vector<float> edge_costs;

        thrust::device_vector<float> t12_costs;
        thrust::device_vector<float> t13_costs;
        thrust::device_vector<float> t23_costs;

        thrust::device_vector<int> triangle_correspondence_12;
        thrust::device_vector<int> triangle_correspondence_13;
        thrust::device_vector<int> triangle_correspondence_23;
        thrust::device_vector<int> edge_counter;
};

