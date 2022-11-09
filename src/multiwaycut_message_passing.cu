#include "multiwaycut_message_passing.h"


multiwaycut_message_passing::multiwaycut_message_passing(
        const dCOO &A,
        const int _n_nodes,
        const int _n_classes,
        thrust::device_vector<int> &&_t1,
        thrust::device_vector<int> &&_t2,
        thrust::device_vector<int> &&_t3,
        const bool verbose
        )
        : multicut_message_passing(A, std::move(_t1), std::move(_t2), std::move(_t3), verbose),
        n_classes(_n_classes),
        n_nodes(_n_nodes),
        class_costs(thrust::device_vector<float>(_n_nodes * _n_classes))  // Zero initialize summation constraints
        {
}

struct class_lower_bound_parallel {
    int k;  // number of classes
    int N;  // number of elements in class_costs
    float* class_costs;
    float* result;
    __host__ __device__ void operator() (int node) {
        size_t offset_start = node * k;
        size_t offset_end = offset_start + k - 1;
        assert(offset_end < N);

        // k should be quite small compared to N/k so summing using the for loop
        // should not be an issue
        float largest = class_costs[0];
        for (size_t i = offset_start; i <= offset_end; ++i) {
            *result += class_costs[i];
            if (class_costs[i] > largest)
                largest = class_costs[i];
        }
        *result -= largest;
    }
};
double multiwaycut_message_passing::class_lower_bound()
{
    // The minimal class-cost-configuration for a node will always be given as
    // the sum of the elements in the cost vector with its largest element removed.
    // Hence instead of calculating the dot product for each of the K possible
    // configuration, we can sum up all elements and subtract the largest cost.

    // As typically N >> K it makes more sense to parallelize over N, we can
    // then simply use a for loop (with only K iterations) to calculate the dot
    // product
    thrust::counting_iterator<int> iter(0);  // Iterates over the node indices
    thrust::device_vector<float> res(1);
    class_lower_bound_parallel f({
        n_classes, n_nodes * n_classes,
        thrust::raw_pointer_cast(class_costs.data()),
        thrust::raw_pointer_cast(res.data())
    });
    thrust::for_each(iter, iter + n_nodes, f);

    return res[0];
}

double multiwaycut_message_passing::lower_bound()
{
    if (n_classes > 0) {
        return multicut_message_passing::lower_bound() + class_lower_bound();
    } else {
        std::cout << "No classes provided. Defaulting to multicut\n";
        return multicut_message_passing::lower_bound();
    }
}

// Adapted from multicut_message_passing to check if an edge is a class-node edge
struct increase_triangle_costs_func_mwc {
    int n_nodes;     // Needed to check if an edge is a class-node edge
    int n_classes;   // For calculating the index in the class_costs_matrix
    int *edge_sources;  // For calculating the index in the class_costs matrix
    int *edge_dests;    // The destination tells us if the edge is a class nod edge

    float* class_costs;
    float* edge_costs;
    int* edge_counter;

    /**
     * Checks if the given destination node is a class node
     * @param dest End node of the edge
     * @return True if the index of the destination node is bigger than any "regular" node, this is the way
     *  a class node is encoded by the `mwc_to_coo` function
     */
    __host__ __device__ bool is_class_edge(int dest) const
    {
        return dest >= n_nodes;
    }

    __device__ void operator()(const thrust::tuple<int,float&> t) const
    {
        const int edge_idx = thrust::get<0>(t);
        float& triangle_cost = thrust::get<1>(t);

        if (is_class_edge(edge_dests[edge_idx])) {
            // edge_counter may be zero but that indicates not part of a triangle?

            float update = edge_costs[edge_idx]/(1.0f + float(edge_counter[edge_idx]));
            triangle_cost += update;
            // update class costs for this edge
            int node = edge_sources[edge_idx];
            int klass = edge_dests[edge_idx];
            // The ith class is encoded as n_nodes + i
            int class_idx = (klass - static_cast<int>(n_nodes));
            assert((node * n_classes + class_idx) < (n_nodes * n_classes));
            atomicAdd(&class_costs[node * n_classes + class_idx], -update);
        } else {
            assert(edge_counter[edge_idx] > 0);
            triangle_cost += edge_costs[edge_idx] / float(edge_counter[edge_idx]);
        }
    }
};


// Direct copy from multicut_message_passing
struct decrease_edge_costs_func {
      __host__ __device__ void operator()(const thrust::tuple<float&,int> x) const
      {
          float& cost = thrust::get<0>(x);
          int counter = thrust::get<1>(x);
          if(counter > 0)
              cost = 0.0;
      }
};


void multiwaycut_message_passing::send_messages_to_triplets()
{
    // Most of this is duplicated from the multicut_message_passing::send_messages_to_triplets method
    // as there is no way with the current interface to override the update function.
    // It might be worth to consider changing the interface to accept a function pointer
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.begin(), t12_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.end(), t12_costs.end()));
        increase_triangle_costs_func_mwc func({
            n_nodes,
            n_classes,
            thrust::raw_pointer_cast(i.data()),
            thrust::raw_pointer_cast(j.data()),
            thrust::raw_pointer_cast(class_costs.data()),
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data())
        });
        thrust::for_each(first, last, func);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.begin(), t13_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.end(), t13_costs.end()));
        increase_triangle_costs_func_mwc func({
            n_nodes,
            n_classes,
            thrust::raw_pointer_cast(i.data()),
            thrust::raw_pointer_cast(j.data()),
            thrust::raw_pointer_cast(class_costs.data()),
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data())
        });
        thrust::for_each(first, last, func);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.begin(), t23_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.end(), t23_costs.end()));
        increase_triangle_costs_func_mwc func({
            n_nodes,
            n_classes,
            thrust::raw_pointer_cast(i.data()),
            thrust::raw_pointer_cast(j.data()),
            thrust::raw_pointer_cast(class_costs.data()),
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data())
        });
        thrust::for_each(first, last, func);
    }

    // Direct copy from multicut_message_passing
    // set costs of edges to zero (if edge participates in a triangle)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.begin(), edge_counter.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.end(), edge_counter.end()));
        thrust::for_each(first, last, decrease_edge_costs_func());
    }


}


struct sum_to_edges_func {
    // Basically the number of nodes, edge_costs should first contain the
    // costs for the normal node-node edges and then the costs for the
    // node-class edges
    int class_cost_start;
    int classes;
    float *edge_costs;
    float *class_costs;

    __host__ __device__
    float get_class_cost(int node, int klass) {
        int offset = node * classes + klass;
        return edge_costs[class_cost_start + offset];
    }

    __device__ void operator() (int node) {

        float update = 0.0f;

        switch (classes) {
            case 1: // Only one class
                update = 0.5f * get_class_cost(node, 0);
                break;
            case 2:
                update = 0.5f * (get_class_cost(node, 0) + get_class_cost(node, 1));
                break;
            default:
                // Find the two largest class costs
                float first = get_class_cost(node, 0);
                float second = get_class_cost(node, 1);
                for (int klass = 2; klass < classes; ++klass) {
                    float val = get_class_cost(node, klass);
                    if (val > first) {
                        second  = first;
                        first = val;
                    } else if (second < val && val <= first) {
                        second = val;
                    }
                }
                update = 0.5f * (first + second);
                break;
        }

        // Update all class edges of this node
        for (int k = 0; k < classes; ++k) {
            int offset = node * classes + k;
            class_costs[offset] -= get_class_cost(node, k) + update;
            //Update the re-parameterized costs
            atomicAdd(&edge_costs[class_cost_start + offset], get_class_cost(node, k) + update);
        }
    }
};


void multiwaycut_message_passing::send_messages_from_sum_to_edges()
{
        thrust::counting_iterator<int> iter(0);  // Iterates over the node indices
        sum_to_edges_func func({
            // The node-class edges are the last n_nodes*n_classes edges
            static_cast<int>(edge_costs.size()) - (n_nodes * n_classes),
            n_classes,
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(class_costs.data())
        });
        thrust::for_each(iter, iter + n_classes, func);
}


void multiwaycut_message_passing::iteration()
{
    send_messages_to_triplets();
    send_messages_to_edges();
    send_messages_from_sum_to_edges();
}

