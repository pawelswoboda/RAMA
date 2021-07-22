#pragma once
#include <thrust/device_vector.h>

class multicut_message_passing {
    public:
        multicut_message_passing(
                thrust::device_vector<int>& _i,
                thrust::device_vector<int>& _j,
                thrust::device_vector<float>& _edge_costs,
                thrust::device_vector<int>& _t1,
                thrust::device_vector<int>& _t2,
                thrust::device_vector<int>& _t3
                );

        void send_messages_to_triplets();
        void send_messages_to_edges();

        double lower_bound();

        void iteration();

        std::tuple<const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>>
            reparametrized_edge_costs() const;

        void compute_triangle_edge_correspondence(const thrust::device_vector<int>&, const thrust::device_vector<int>&, 
            thrust::device_vector<int>&, thrust::device_vector<int>&);

    private:
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

