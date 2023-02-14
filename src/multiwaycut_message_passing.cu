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
        class_costs(thrust::device_vector<float>(_n_nodes * _n_classes)),  // Zero initialize summation constraints
        is_class_edge(i.size(), false)
        {
    // Populate is_class_edge lookup table
    for (int idx = 0; idx < i.size(); ++idx) {
        is_class_edge[idx] = IS_CLASS_EDGE(j[idx]);
    }

    print_vector(i, "sources");
    print_vector(j, "dest   ");
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

/**
 * Triangle lower bound adapted to the case of 2 classes.
 * We dont allow all edges to be cut if a triangle contains class edges
 */
struct triangle_lower_bound_2_classes_func {
    bool* is_class_edge;

    __host__ __device__ float operator() (const thrust::tuple<float,float,float, int, int, int> x) {
        const float c12 = thrust::get<0>(x);
        const float c13 = thrust::get<1>(x);
        const float c23 = thrust::get<2>(x);
        const int t12 = thrust::get<3>(x);
        const int t13 = thrust::get<4>(x);
        const int t23 = thrust::get<5>(x);

        float lb = 0.0;
        lb = min(lb, c12 + c13);
        lb = min(lb, c12 + c23);
        lb = min(lb, c13 + c23);

        // No class in triangle => all edges can be cut
        bool includes_class_edge = (is_class_edge[t12]
                || is_class_edge[t13]
                || is_class_edge[t23]);
        if (includes_class_edge) {
            lb = min(lb, c12 + c13 + c23);
        }
        return lb;
    }
};
double multiwaycut_message_passing::triangle_lower_bound_2_classes() {
    auto first = thrust::make_zip_iterator(thrust::make_tuple(
        t12_costs.begin(), t13_costs.begin(), t23_costs.begin(),
        triangle_correspondence_12.begin(), triangle_correspondence_13.begin(), triangle_correspondence_23.begin()
    ));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(
        t12_costs.end(), t13_costs.end(), t23_costs.end(),
        triangle_correspondence_12.end(), triangle_correspondence_13.end(), triangle_correspondence_23.end()
    ));
    triangle_lower_bound_2_classes_func f = triangle_lower_bound_2_classes_func({
        thrust::raw_pointer_cast(is_class_edge.data())
    });
    return thrust::transform_reduce(first, last, f, 0.0, thrust::plus<float>());
}


double multiwaycut_message_passing::lower_bound()
{
    if (n_classes > 0) {
        double clb = class_lower_bound();

        double tlb;
        if (n_classes <= 2) {
            tlb = triangle_lower_bound_2_classes();
        } else {
            tlb = triangle_lower_bound();
        }
        double res = edge_lower_bound() + tlb + clb;
        printf("%f+%f+%f = %f\n", edge_lower_bound(), tlb, clb, res);
        return res;
    } else {
        std::cout << "No classes provided. Defaulting to multicut\n";
        return multicut_message_passing::lower_bound();
    }
}

// Adapted from multicut_message_passing to check if an edge is a class-node edge
struct increase_triangle_costs_func_mwc {
    float* edge_costs;
    int* edge_counter;

    bool* is_class_edge;

    __device__ void operator()(const thrust::tuple<int,float&> t) const
    {
        const int edge_idx = thrust::get<0>(t);
        float& triangle_cost = thrust::get<1>(t);

        if (is_class_edge[edge_idx]) {
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
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data()),
            thrust::raw_pointer_cast(is_class_edge.data())
        });
        thrust::for_each(first, last, func);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.begin(), t13_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.end(), t13_costs.end()));
        increase_triangle_costs_func_mwc func({
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data()),
            thrust::raw_pointer_cast(is_class_edge.data())
        });
        thrust::for_each(first, last, func);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.begin(), t23_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.end(), t23_costs.end()));
        increase_triangle_costs_func_mwc func({
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data()),
            thrust::raw_pointer_cast(is_class_edge.data())
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


struct decrease_triangle_costs_2_classes_func {
    float* edge_costs;
    bool* is_class_edge;

    /**
     * Calculate the min marginal to edge with costs x, i.e. the order of the parameters defines the min marginal
     */
    __host__ __device__
        float min_marginal(const float x, const float y, const float z, const bool includes_class_edge) const
        {
            float mm1 = min(x+y, x+z);
            if (includes_class_edge) {
                mm1 = min(x+y+z, mm1);
            }
            float mm0 = min(0.0, y+z);
            printf("[%d](%f, %f, %f): %f - %f = %f\n", includes_class_edge, x, y, z, mm1, mm0, mm1-mm0);
            return mm1-mm0;
        }

    __device__
        void operator()(const thrust::tuple<int,int,int,float&,float&,float&,int,int,int> t) const
        {
            const int t12 = thrust::get<0>(t);
            const int t13 = thrust::get<1>(t);
            const int t23 = thrust::get<2>(t);
            float& t12_costs = thrust::get<3>(t);
            float& t13_costs = thrust::get<4>(t);
            float& t23_costs = thrust::get<5>(t);
            const int t12_correspondence = thrust::get<6>(t);
            const int t13_correspondence = thrust::get<7>(t);
            const int t23_correspondence = thrust::get<8>(t);

            // If this triangle includes a class edge we have to adapt our min marginal
            // such that not all edges can be cut.
            bool includes_class_edge = (is_class_edge[t12_correspondence]
                || is_class_edge[t13_correspondence]
                || is_class_edge[t23_correspondence]);

            float e12_diff = 0.0;
            float e13_diff = 0.0;
            float e23_diff = 0.0;

            {
                printf("(%d, %d, %d) \t", t12_correspondence, t13_correspondence, t23_correspondence);
                const float mm12 = min_marginal(t12_costs, t13_costs, t23_costs, includes_class_edge);
                t12_costs -= 1.0/3.0*mm12;
                e12_diff += 1.0/3.0*mm12;
            }
            {
                printf("(%d, %d, %d) \t", t13_correspondence, t12_correspondence, t23_correspondence);
                const float mm13 = min_marginal(t13_costs, t12_costs, t23_costs, includes_class_edge);
                t13_costs -= 1.0/2.0*mm13;
                e13_diff += 1.0/2.0*mm13;
            }
            {
                printf("(%d, %d, %d) \t", t23_correspondence, t12_correspondence, t13_correspondence);
                const float mm23 = min_marginal(t23_costs, t12_costs, t13_costs, includes_class_edge);
                t23_costs -= mm23;
                e23_diff += mm23;
            }
            {
                printf("(%d, %d, %d) \t", t12_correspondence, t13_correspondence, t23_correspondence);
                const float mm12 = min_marginal(t12_costs, t13_costs, t23_costs, includes_class_edge);
                t12_costs -= 1.0/2.0*mm12;
                e12_diff += 1.0/2.0*mm12;
            }
            {
                printf("(%d, %d, %d) \t", t13_correspondence, t12_correspondence, t23_correspondence);
                const float mm13 = min_marginal(t13_costs, t12_costs, t23_costs, includes_class_edge);
                t13_costs -= mm13;
                e13_diff += mm13;
            }
            {
                printf("(%d, %d, %d) \t", t12_correspondence, t13_correspondence, t23_correspondence);
                const float mm12 = min_marginal(t12_costs, t13_costs, t23_costs, includes_class_edge);
                t12_costs -= mm12;
                e12_diff += mm12;
            }

//            {
//                printf("(%d, %d, %d) \t", t12_correspondence, t13_correspondence, t23_correspondence);
//                const float mm13 = min_marginal(t13_costs, t12_costs, t23_costs, includes_class_edge);
//                t13_costs -= mm13;
//                e13_diff += mm13;
//            }

            atomicAdd(&edge_costs[t12_correspondence], e12_diff);
            atomicAdd(&edge_costs[t13_correspondence], e13_diff);
            atomicAdd(&edge_costs[t23_correspondence], e23_diff);
        }
};

void multiwaycut_message_passing::send_messages_to_edges()
{
    if (n_classes <= 2) {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(
            t1.begin(), t2.begin(), t3.begin(),
            t12_costs.begin(), t13_costs.begin(), t23_costs.begin(),
            triangle_correspondence_12.begin(), triangle_correspondence_13.begin(), triangle_correspondence_23.begin()
        ));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(
            t1.end(), t2.end(), t3.end(),
            t12_costs.end(), t13_costs.end(), t23_costs.end(),
            triangle_correspondence_12.end(), triangle_correspondence_13.end(), triangle_correspondence_23.end()
        ));
        decrease_triangle_costs_2_classes_func f = decrease_triangle_costs_2_classes_func({
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(is_class_edge.data())
        });
        thrust::for_each(first, last, f);
    } else {
        multicut_message_passing::send_messages_to_edges();
    }
    print_vector(edge_costs, "edge_costs after msg to edges");
    print_vector(class_costs, "class_costs after msg to edges");
    print_vector(t12_costs, "t12 after msg to edges ");
    print_vector(t13_costs, "t13 after msg to edges ");
    print_vector(t23_costs, "t23 after msg to edges ");
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

