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
        std::vector<int> get_i() const {
            std::vector<int> h_i(i.size());
            thrust::copy(i.begin(), i.end(), h_i.begin());
            return h_i;
        }

        std::vector<int> get_j() const {
            std::vector<int> h_j(j.size());
            thrust::copy(j.begin(), j.end(), h_j.begin());
            return h_j;
        }

        std::vector<float> get_edge_costs() const {
            std::vector<float> h_edge(edge_costs.size());
            thrust::copy(edge_costs.begin(), edge_costs.end(), h_edge.begin());
            return h_edge;
        }

        std::vector<float> get_t12_costs() const {
            std::vector<float> h(t12_costs.size());
            thrust::copy(t12_costs.begin(), t12_costs.end(), h.begin());
            return h;
        }
        std::vector<float> get_t13_costs() const {
            std::vector<float> h(t13_costs.size());
            thrust::copy(t13_costs.begin(), t13_costs.end(), h.begin());
            return h;
        }
        std::vector<float> get_t23_costs() const {
            std::vector<float> h(t23_costs.size());
            thrust::copy(t23_costs.begin(), t23_costs.end(), h.begin());
            return h;
        }

        std::vector<int> get_tri_corr_12() const {
            std::vector<int> h(triangle_correspondence_12.size());
            thrust::copy(triangle_correspondence_12.begin(), triangle_correspondence_12.end(), h.begin());
            return h;
        }
        std::vector<int> get_tri_corr_13() const {
            std::vector<int> h(triangle_correspondence_13.size());
            thrust::copy(triangle_correspondence_13.begin(), triangle_correspondence_13.end(), h.begin());
            return h;
        }
        std::vector<int> get_tri_corr_23() const {
            std::vector<int> h(triangle_correspondence_23.size());
            thrust::copy(triangle_correspondence_23.begin(), triangle_correspondence_23.end(), h.begin());
            return h;
        }

        std::vector<int> get_edge_counter() const {
            std::vector<int> h(edge_counter.size());
            thrust::copy(edge_counter.begin(), edge_counter.end(), h.begin());
            return h;
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

