#include "multiwaycut_message_passing.h"
#include "rama_utils.h"

// An edge is a node class-edge iff the destination node value is larger than the largest base node
// which are assigned values from [0...n_nodes]
#define IS_CLASS_EDGE(dest) (dest >= n_nodes)

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


std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> multiwaycut_message_passing::get_node_partition()
{
    thrust::device_vector<int> nodes(n_nodes);
    thrust::device_vector<int> sizes(n_nodes);
    // After sorting i has the keys in consecutive, ascending order we can now
    // sum up the number of times the key exists in the array and thus find
    // the start of the partition that only contains the edges with this node
    thrust::reduce_by_key(
        i.begin(), i.end(),
        thrust::make_constant_iterator(1),
        nodes.begin(), sizes.begin()
    );


    // Sum up the sizes to find the starting index
    thrust::device_vector<int> starts(n_nodes);
    thrust::exclusive_scan(sizes.begin(), sizes.end(), starts.begin());

    return {nodes, starts, sizes};
}

struct class_lower_bound_parallel {
    int k;  // number of classes
    int N;  // number of elements in class_costs
    float* class_costs;
    float* result;
    __host__ __device__ void operator() (int node) {
        int offset_start = node * k;
        int offset_end = offset_start + k - 1;
        assert(offset_end < N);

        // k should be quite small compared to N/k so summing using the for loop
        // should not be an issue
        float largest = class_costs[offset_start];
        for (int i = offset_start; i <= offset_end; ++i) {
            result[node] += class_costs[i];
            largest = max(largest, class_costs[i]);
        }
        result[node] -= largest;
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
    thrust::device_vector<float> res(n_nodes);
    class_lower_bound_parallel f({
        n_classes, n_nodes * n_classes,
        thrust::raw_pointer_cast(class_costs.data()),
        thrust::raw_pointer_cast(res.data())
    });
    thrust::for_each(iter, iter + n_nodes, f);
    return thrust::reduce(res.begin(), res.end(), 0.0);
}

double multiwaycut_message_passing::lower_bound()
{
    if (n_classes > 0) {
        double clb = class_lower_bound();
        double res = edge_lower_bound() + triangle_lower_bound() + clb;
        printf("%f+%f+%f = %f\n", edge_lower_bound(), triangle_lower_bound(), clb, res);
        return res;
    } else {
        std::cout << "No classes provided. Defaulting to multicut\n";
        return multicut_message_passing::lower_bound();
    }
}

// Adapted from multicut_message_passing to check if an edge is a class-node edge
struct increase_triangle_costs_func_mwc {
    int n_nodes;     // Needed to check if an edge is a class-node edge
    int *edge_dests;    // The destination tells us if the edge is a class nod edge

    float* edge_costs;
    int* edge_counter;

    __device__ void operator()(const thrust::tuple<int,float&> t) const
    {
        const int edge_idx = thrust::get<0>(t);
        float& triangle_cost = thrust::get<1>(t);

        if (IS_CLASS_EDGE(edge_dests[edge_idx])) {
            float update = edge_costs[edge_idx]/(1.0f + float(edge_counter[edge_idx]));
            triangle_cost += update;
        } else {
            assert(edge_counter[edge_idx] > 0);
            triangle_cost += edge_costs[edge_idx] / float(edge_counter[edge_idx]);
        }
    }
};

struct increase_class_costs_func {
    int n_nodes;
    int n_classes;
    float *class_costs;


    __device__ void operator()(const thrust::tuple<int, int, int, float> t) {
        int source = thrust::get<0>(t);
        int dest = thrust::get<1>(t);
        int edge_count = thrust::get<2>(t);
        float edge_cost = thrust::get<3>(t);

        // Check if it is class edge
        if (!IS_CLASS_EDGE(dest)) return;

        // the ith class is encoded as n_nodes + i
        int class_idx = source * n_classes + (dest - n_nodes);
        assert(class_idx < n_nodes * n_classes);
        class_costs[class_idx] += edge_cost / (1.0f + float(edge_count));
    }
};


// Direct copy from multicut_message_passing
struct decrease_edge_costs_func {
    int n_nodes;
    __host__ __device__ void operator()(const thrust::tuple<float&,int, int> x) const
    {
        float& cost = thrust::get<0>(x);
        int counter = thrust::get<1>(x);
        int dest = thrust::get<2>(x);
        if(counter > 0 || IS_CLASS_EDGE(dest)) {
            cost = 0.0;  // Participates in a triangle or is a class edge
        }
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
            thrust::raw_pointer_cast(j.data()),
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
            thrust::raw_pointer_cast(j.data()),
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
            thrust::raw_pointer_cast(j.data()),
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data())
        });
        thrust::for_each(first, last, func);
    }

    // Send messages to summation constraints
    {
        increase_class_costs_func func({n_nodes, n_classes, thrust::raw_pointer_cast(class_costs.data())});
        auto first = thrust::make_zip_iterator(thrust::make_tuple(
            i.begin(), j.begin(), edge_counter.begin(), edge_costs.begin()
        ));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(
            i.end(), j.end(), edge_counter.end(), edge_costs.end()
        ));
        thrust::for_each(first, last, func);
    }

    // set costs of edges to zero (if edge participates in a triangle or is a class edge)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.begin(), edge_counter.begin(), j.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.end(), edge_counter.end(), j.end()));
        thrust::for_each(first, last, decrease_edge_costs_func({n_nodes}));
    }

    print_vector(edge_costs, "edge_costs after msg to triplets");
    print_vector(class_costs, "class_costs after msg to triplets");
    print_vector(t12_costs, "t12 after msg to triplets");
    print_vector(t13_costs, "t13 after msg to triplets");
    print_vector(t23_costs, "t23 after msg to triplets");
}


struct sum_to_edges_func {
    int classes;
    float *edge_costs;
    float *class_costs;

    __device__ void operator() (thrust::tuple<int, int, int> t) {
        int node = thrust::get<0>(t);
        int start = thrust::get<1>(t);
        int size = thrust::get<2>(t);

        float update = 0.0f;

        int cstart = node * classes;

        switch (classes) {
            case 1: // Only one class
                update = 0.5f * class_costs[cstart];
                break;
            case 2:
                update = 0.5f * (class_costs[cstart] + class_costs[cstart + 1]);
                break;
            default:
                // Find the two largest class costs
                float first = class_costs[cstart];
                float second = class_costs[cstart + 1];
                for (int k = 2; k < classes; ++k) {
                    float cost = class_costs[cstart + k];
                    if (cost > first) {
                        second  = first;
                        first = cost;
                    } else if (second < cost && cost <= first) {
                        second = cost;
                    }
                }
                update = 0.5f * (first + second);
                break;
        }

        // Update all class edges of this node
        for (int k = 0; k < classes; ++k) {
            int offset = node * classes + k;

            //Update the re-parameterized costs for node-class edges
            atomicAdd(&edge_costs[start + size - classes + k], -update+class_costs[offset]);
            class_costs[offset] -= class_costs[offset] - update;
        }
    }
};


void multiwaycut_message_passing::send_messages_from_sum_to_edges()
{
    thrust::device_vector<int> node;
    thrust::device_vector<int> start;
    thrust::device_vector<int> size;
    std::tie(node, start, size) = get_node_partition();
    sum_to_edges_func func({
        n_classes,
        thrust::raw_pointer_cast(edge_costs.data()),
        thrust::raw_pointer_cast(class_costs.data())
    });
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(node.begin(), start.begin(), size.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(node.end(), start.end(), size.end())),
        func
    );
    print_vector(edge_costs, "edge_costs after msg to edges 2");
    print_vector(class_costs, "class_costs after msg to edges 2");
    print_vector(t12_costs, "t12 after msg to edges 2");
    print_vector(t13_costs, "t13 after msg to edges 2");
    print_vector(t23_costs, "t23 after msg to edges 2");
}


void multiwaycut_message_passing::iteration()
{
    send_messages_to_triplets();
    send_messages_to_edges();
    send_messages_from_sum_to_edges();
}
void multiwaycut_message_passing::send_messages_to_edges()
{
    multicut_message_passing::send_messages_to_edges();
    print_vector(edge_costs, "edge_costs after msg to edges");
    print_vector(class_costs, "class_costs after msg to edges");
    print_vector(t12_costs, "t12 after msg to edges ");
    print_vector(t13_costs, "t13 after msg to edges ");
    print_vector(t23_costs, "t23 after msg to edges ");
}

