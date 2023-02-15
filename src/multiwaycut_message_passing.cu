#include "multiwaycut_message_passing.h"
#include "rama_utils.h"
#include <algorithm>
#include <numeric>
#include <thrust/inner_product.h>

// An edge is a node class-edge iff the destination node value is larger than the largest base node
// which are assigned values from [0...n_nodes]
// TODO: Replace all occurrences of the macro with the is_class_edge vector
#define IS_CLASS_EDGE(dest) (dest >= n_nodes)


multiwaycut_message_passing::multiwaycut_message_passing(
        const dCOO &A,
        const int _n_nodes,
        const int _n_classes,
        thrust::device_vector<int> &&_t1,
        thrust::device_vector<int> &&_t2,
        thrust::device_vector<int> &&_t3,
        const MWCOptions _options
        )
        :
        multicut_message_passing(A, std::move(_t1), std::move(_t2), std::move(_t3), _options & MWCOptions::VERBOSE),
        options(_options),
        n_classes(_n_classes),
        n_nodes(_n_nodes),
        class_costs(_n_nodes * _n_classes, 0),  // Zero initialize summation constraints
        is_class_edge(create_class_edge_mask(i, j, [_n_nodes](int source, int dest) {return dest >= _n_nodes;} )),
        cdtf_counter(edge_counter.size(), 0),
        cdtf_costs(0, 0.0)  // Size is increased when adding new class dependent triangle factors
        {
    print_vector(i, "sources");
    print_vector(j, "dest   ");


    // We pretend triangles don't exist but multicut initializes edge_counter
    // Which is used even in the non triangle message, e.g. to the classes
    // We simply set the edge_counter to 0
    if (options & MWCOptions::IGNORE_TRIANGLES) {
        for (int idx = 0; idx < edge_counter.size(); ++idx) {
            edge_counter[idx] = 0;
        }
    }
}

/***************************** Lower bounds **********************************/

/**
 * Functor to calculate the summation constraint lb for a single node, i.e.
 * min{cost(node) * y | y in (0, 1, ..., 1), (1, 0, ..., 1), ... }
 */
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

/**
 * Calculates the lower bound for the summation constraint subproblems
 * @return Lower bound of summation constraints
 */
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
 * Functor to calculate the lower bound for a single triangle
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

/**
 * Calculates the lower bound of the triangle subproblems in the case of 2 classes.
 * In this case we don't allow all edges in the base graph to be cut as this would imply that each node
 * is part of a distinct class.
 */
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

/**
 * Calculates the overall lower bound for multiway cut, i.e. the sum of all subproblem lower bounds
 */
struct cdtf_lower_bound_func{
    float* results;
    float* costs;
    const float ys[37][9] = {
         {0, 0, 0, 1, 0, 1, 0, 1, 0},
         {0, 0, 0, 0, 1, 0, 1, 0, 1},
         {1, 1, 0, 0, 1, 1, 0, 1, 0},
         {1, 1, 0, 1, 0, 0, 1, 0, 1},
         {1, 0, 1, 1, 0, 0, 1, 1, 0},
         {1, 0, 1, 0, 1, 1, 0, 0, 1},
         {0, 1, 1, 1, 0, 1, 0, 0, 1},
         {0, 1, 1, 0, 1, 0, 1, 1, 0},
         {1, 1, 0, 1, 1, 1, 0, 1, 0},
         {1, 1, 0, 1, 1, 0, 1, 0, 1},
         {1, 0, 1, 1, 0, 1, 1, 1, 0},
         {1, 0, 1, 0, 1, 1, 1, 0, 1},
         {0, 1, 1, 1, 0, 1, 0, 1, 1},
         {0, 1, 1, 0, 1, 0, 1, 1, 1},
         {0, 0, 0, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 0, 0, 1},
         {1, 1, 1, 0, 1, 1, 1, 1, 0},
         {1, 1, 1, 1, 0, 1, 1, 0, 1},
         {1, 1, 1, 0, 1, 1, 0, 1, 1},
         {1, 1, 1, 1, 0, 0, 1, 1, 1},
         {0, 1, 1, 1, 1, 1, 1, 1, 0},
         {1, 0, 1, 1, 1, 1, 0, 1, 1},
         {1, 1, 0, 1, 0, 1, 1, 1, 1},
         {0, 1, 1, 1, 1, 1, 1, 0, 1},
         {1, 0, 1, 1, 1, 0, 1, 1, 1},
         {1, 1, 0, 0, 1, 1, 1, 1, 1},
         {0, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 1, 1, 1, 1, 1, 1, 1},
         {1, 1, 0, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 0, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 0, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 0, 1, 1, 1},
         {1, 1, 1, 1, 1, 1, 0, 1, 1},
         {1, 1, 1, 1, 1, 1, 1, 0, 1},
         {1, 1, 1, 1, 1, 1, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1}
     };
    const int rows = 37;
    const int cols = 9;

    __device__ void operator() (const int cdtf_idx) const {
        // 9 values for each problem
        int offset = cdtf_idx * 9;
        float best = 0.0;
        for (int r = 0; r < rows; ++r) {
            float result = 0.0;
            const float *row = ys[r];
            for (int i = 0; i < cols; ++i) {
                result += row[i] * costs[offset + i];
            }
            best = min(result, best);
        }
        results[cdtf_idx] = best;
    }
};
double multiwaycut_message_passing::cdtf_lower_bound() {
    // With no triangles we have no lower bound
    if (cdtf_costs.empty()) return 0.0;

    auto dv = std::div(cdtf_costs.size(), 9);
    assert(dv.rem == 0);  // We should always have chunks of 9 costs, one for each subproblem
    int nr_cdtf_problems = dv.quot;

    // We parallelize over the class-dependent triangle factors
    thrust::device_vector<float> results(nr_cdtf_problems);
    auto idx = thrust::make_counting_iterator<int>(0);  // Index of the cdtf
    cdtf_lower_bound_func func({
        thrust::raw_pointer_cast(results.data()), 
        thrust::raw_pointer_cast(cdtf_costs.data())
    });
    thrust::for_each(idx, idx + nr_cdtf_problems, func);
    return thrust::reduce(results.begin(), results.end(), 0.0);
}

double multiwaycut_message_passing::lower_bound()
{
    if (n_classes > 0) {
        double clb = class_lower_bound();

        double tlb;
        if (options ^ MWCOptions::IGNORE_TRIANGLES) {
            if (n_classes <= 2) {
                tlb = triangle_lower_bound_2_classes();
            } else {
                tlb = triangle_lower_bound();
            }
        } else {
            tlb = 0.0;
        }

        double cdtflb = cdtf_lower_bound();
        double res = edge_lower_bound() + tlb + clb + cdtflb;
        printf("%f+%f+%f+%f = %f\n", edge_lower_bound(), tlb, clb, cdtflb, res);
        return res;
    } else {
        std::cout << "No classes provided. Defaulting to multicut\n";
        return multicut_message_passing::lower_bound();
    }
}

/***************************** Message passing *******************************/

/**
 * Message passing functor from edges to triplets
 */
struct increase_triangle_costs_func_mwc {
    float* edge_costs;
    int* edge_counter;
    int* cdtf_counter;

    bool* is_class_edge;

    __device__ void operator()(const thrust::tuple<int,float&> t) const
    {
        const int edge_idx = thrust::get<0>(t);
        float& triangle_cost = thrust::get<1>(t);

        if (is_class_edge[edge_idx]) {
            float update = edge_costs[edge_idx] / (
                1.0f + float(edge_counter[edge_idx]) + float(cdtf_counter[edge_idx])
            );
            triangle_cost += update;
        } else {
            assert(edge_counter[edge_idx] + cdtf_counter[edge_idx] > 0);
            triangle_cost += edge_costs[edge_idx] / (
                float(edge_counter[edge_idx]) + float(cdtf_counter[edge_idx])
            );
        }
    }
};

 // TODO: If we rewrite the summation constraint problems in the same form we could use a single message passing
 //  functor for the messages from the edges to all subproblems, just changing the costs
struct increase_cdtf_costs_func{
  float* edge_costs;
  int* edge_counter;
  int* cdtf_counter;
  bool* is_class_edge;

  __device__ void operator()(const thrust::tuple<int, float&> t) const
  {
    const int edge_idx = thrust::get<0>(t);
    float& cdtf_cost = thrust::get<1>(t);

    if (is_class_edge[edge_idx]) {
        float update = edge_costs[edge_idx] / (
            1.0f + float(edge_counter[edge_idx]) + float(cdtf_counter[edge_idx])
        );
        cdtf_cost += update;
    } else {
        assert(edge_counter[edge_idx] + cdtf_counter[edge_idx] > 0);
        cdtf_cost += edge_costs[edge_idx] / (
            float(edge_counter[edge_idx]) + float(cdtf_counter[edge_idx])
        );
    }
  }
};

/**
 * Message passing functor from edges to summation constraints
 */
struct increase_class_costs_func {
    int n_nodes;
    int n_classes;
    float *class_costs;


    __device__ void operator()(const thrust::tuple<int, int, int, float, int> t) {
        int source = thrust::get<0>(t);
        int dest = thrust::get<1>(t);
        int edge_count = thrust::get<2>(t);
        float edge_cost = thrust::get<3>(t);
        int cdtf_count = thrust::get<4>(t);

        // Check if it is class edge
        if (!IS_CLASS_EDGE(dest)) return;

        // the ith class is encoded as n_nodes + i
        int class_idx = source * n_classes + (dest - n_nodes);
        assert(class_idx < n_nodes * n_classes);
        class_costs[class_idx] += edge_cost / (1.0f + float(edge_count) + float(cdtf_count));
    }
};

/**
 * Cost update functor after the message passing from the edges
 * sets all edges that participate in any subproblem to 0
 */
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

/**
 * Message passing from edges to triplets and summation constraints
 * First sends messages to all edges in triangle subproblems then to all summation constraints
 * In the end sets all edges that are party of any subproblem to 0
 */
void multiwaycut_message_passing::send_messages_to_triplets()
{
    // Most of this is duplicated from the multicut_message_passing::send_messages_to_triplets method
    // as there is no way with the current interface to override the update function.
    // It might be worth to consider changing the interface to accept a function pointer
    if (options & MWCOptions::NO_MESSAGES_FROM_EDGES) {
        throw std::logic_error("options indicate that no messages from the edges should be send");
    }

    // Messages to triangles
    // If we ignore triangles we don't want to send messages to them as we pretend they don't exist
    if (options ^ MWCOptions::IGNORE_TRIANGLES) {

        {
            auto first =
                thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.begin(), t12_costs.begin()));
            auto
                last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.end(), t12_costs.end()));
            increase_triangle_costs_func_mwc func({
                                                      thrust::raw_pointer_cast(edge_costs.data()),
                                                      thrust::raw_pointer_cast(edge_counter.data()),
                                                      thrust::raw_pointer_cast(cdtf_counter.data()),
                                                      thrust::raw_pointer_cast(is_class_edge.data()),
                                                  });
            thrust::for_each(first, last, func);
        }
        {
            auto first =
                thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.begin(), t13_costs.begin()));
            auto
                last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.end(), t13_costs.end()));
            increase_triangle_costs_func_mwc func({
                                                      thrust::raw_pointer_cast(edge_costs.data()),
                                                      thrust::raw_pointer_cast(edge_counter.data()),
                                                      thrust::raw_pointer_cast(cdtf_counter.data()),
                                                      thrust::raw_pointer_cast(is_class_edge.data())
                                                  });
            thrust::for_each(first, last, func);
        }
        {
            auto first =
                thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.begin(), t23_costs.begin()));
            auto
                last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.end(), t23_costs.end()));
            increase_triangle_costs_func_mwc func({
                                                      thrust::raw_pointer_cast(edge_costs.data()),
                                                      thrust::raw_pointer_cast(edge_counter.data()),
                                                      thrust::raw_pointer_cast(cdtf_counter.data()),
                                                      thrust::raw_pointer_cast(is_class_edge.data())
                                                  });
            thrust::for_each(first, last, func);
        }
    }

    // Messages to summation constraints
    {
        increase_class_costs_func func({n_nodes, n_classes, thrust::raw_pointer_cast(class_costs.data())});
        auto first = thrust::make_zip_iterator(thrust::make_tuple(
            i.begin(), j.begin(), edge_counter.begin(), edge_costs.begin(), cdtf_counter.end()
        ));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(
            i.end(), j.end(), edge_counter.end(), edge_costs.end(), cdtf_counter.end()
        ));
        thrust::for_each(first, last, func);
    }

    // Send messaged to class dependent triangle factors
    {
        auto first = thrust::make_zip_iterator(
            thrust::make_tuple(cdtf_correspondence.begin(), cdtf_costs.begin())
        );
        auto last = thrust::make_zip_iterator(
            thrust::make_tuple(cdtf_correspondence.end(), cdtf_costs.end())
        );
        increase_cdtf_costs_func func({
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data()),
            thrust::raw_pointer_cast(cdtf_counter.data()),
            thrust::raw_pointer_cast(is_class_edge.data()),
        });
        thrust::for_each(first, last, func);
    }

    // set costs of edges to zero (if edge participates in a triangle or is a class edge)
    // No need to check if part of class dependent triangle factor because this implied by being part of a triangle
    // or being a class edge
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.begin(), edge_counter.begin(), j.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.end(), edge_counter.end(), j.end()));
        thrust::for_each(first, last, decrease_edge_costs_func({n_nodes}));
    }

    print_vector(edge_costs, "edge_costs after msg to triplets");
    print_vector(class_costs, "class_costs after msg to triplets");
    print_vector(cdtf_costs, "cdtf_costs after msg to triplets");
    print_vector(t12_costs, "t12 after msg to triplets");
    print_vector(t13_costs, "t13 after msg to triplets");
    print_vector(t23_costs, "t23 after msg to triplets");
}

/**
 * Message passing functor from triplets to edges in the special case that the number of classes is 2
 * Uses min-marginals to distribute messages uniformly to the edges in the triangle
 */
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

/**
 * Message passing from triplets to the edges
 */
void multiwaycut_message_passing::send_messages_to_edges()
{
    if (options & MWCOptions::NO_MESSAGES_FROM_TRIANGLES) {
        throw std::logic_error("options indicate that no messages from the triangle subproblems should be send");
    }

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
    print_vector(cdtf_costs, "cdtf_costs after msg to edges");
    print_vector(t12_costs, "t12 after msg to edges ");
    print_vector(t13_costs, "t13 after msg to edges ");
    print_vector(t23_costs, "t23 after msg to edges ");
}

/**
 * Message passing functor from summation constraints back to the edges
 */
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

/**
 * Message passing from summation constraints to edges
 */
void multiwaycut_message_passing::send_messages_from_sum_to_edges()
{
    if (options & MWCOptions::NO_MESSAGES_FROM_TRIANGLES) {
        throw std::logic_error("options indicate that no messages from summation constraints should be send");
    }

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
    print_vector(cdtf_costs, "cdtf_costs after msg to edges 2");
    print_vector(t12_costs, "t12 after msg to edges 2");
    print_vector(t13_costs, "t13 after msg to edges 2");
    print_vector(t23_costs, "t23 after msg to edges 2");
}


void multiwaycut_message_passing::iteration()
{
    if (options ^ MWCOptions::NO_MESSAGES_FROM_EDGES) {
        send_messages_to_triplets();
    }

    if (options ^ MWCOptions::NO_MESSAGES_FROM_TRIANGLES) {
        send_messages_to_edges();
    }

    if (options ^ MWCOptions::NO_MESSAGES_FROM_SUM_CONSTRAINTS) {
        send_messages_from_sum_to_edges();
    }
}

/***************************** Helper functions ******************************/

thrust::device_vector<bool> multiwaycut_message_passing::create_class_edge_mask(
    thrust::device_vector<int> sources,
    thrust::device_vector<int> dests,
    std::function<bool(int,int)> is_class_edge_f) {
    assert(sources.size() == dests.size());
    thrust::device_vector<bool> mask(sources.size());

    for (int i = 0; i < sources.size(); ++i) {
        mask[i] = is_class_edge_f(sources[i], dests[i]);
    }

    return mask;
}

thrust::device_vector<int> multiwaycut_message_passing::get_nodes_in_triangle(int e1, int e2, int e3) {

    thrust::device_vector<int> nodes= std::vector<int>({i[e1], j[e1], i[e2], j[e2], i[e3], j[e3]});
    thrust::device_vector<int> unique_nodes(3);

    thrust::unique_copy(nodes.begin(), nodes.end(), unique_nodes.begin());
    assert(unique_nodes.size() == 3);  // No more than 3 unique nodes should be in the edges of an triangle
    thrust::sort(unique_nodes.begin(), unique_nodes.end());  // We want ascending order

    return unique_nodes;
}

/**
 * Partitions the edges by node. Assumes that i, j is sorted by the node values, this is the case after initializing
 * multicut
 * @return the number of edges per node and the offset at which the edges for a node u
 * start in the i or j array start.
 */
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>>
multiwaycut_message_passing::get_node_partition()
{
    assert(thrust::is_sorted(i.begin(), i.end()));  // This should be the case after initializing multicut

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

int multiwaycut_message_passing::get_class_edge_index(int node, int k) {

    if (node >= n_nodes) {
        throw std::invalid_argument("Node is not part of the base graph");
    }

    // TODO: at the moment quite inefficient as we have to iterate over all nodes to find the index
    //  in this case using a map would make more sense, however the current implementation of `get_node_partition`
    //  is easier to use with thrust
    thrust::device_vector<int> nodes; // Contains all nodes
    thrust::device_vector<int> node_offset;  // The start of the edges with node v as source in the i or j vector for each node v in V
    thrust::device_vector<int> n_edges;  // The number of edges with node v as source for each node v in V
    std::tie(nodes, node_offset, n_edges) = get_node_partition();

    for (int idx = 0; idx < nodes.size(); ++idx) {
        if (nodes[idx] == node) {
            int edges_end = node_offset[idx] + n_edges[idx];
            return edges_end - n_classes + k;
        }
    }

    throw std::runtime_error("Node is not part of the base graph");
}

/**
 * Adds the triangle consisting of the three edges and the two classes as a class-dependent triangle factor
 * subproblem
 * @param e1 Index of the first edge
 * @param e2 Index of the second edge
 * @param e3 Index of the third edge
 * @param c1 First class as integer in [0, K)
 * @param c2 Second class as integer in [0, K)
 */
void multiwaycut_message_passing::add_class_dependent_triangle_subproblems(int e1, int e2, int e3, int c1, int c2) {
    // The triangle has to be in the base graph
    assert(!is_class_edge[e1] && !is_class_edge[e2] && !is_class_edge[e3]);

    // Each subproblem consists of 9 costs, one for each edge
    for (int i = 0; i < 9; ++i) {
        cdtf_costs.push_back(0.0);
    }

    // e1, e2, e3 are already the indices of the edges as specified by i and j
    cdtf_counter[e1] += 1;
    cdtf_counter[e2] += 1;
    cdtf_counter[e3] += 1;

    // get_nodes_in_triangle returns an ascending vector of the node ids
    // Hence the loop will iterate in order 1c1 2c1 3c1 1c2 ...
    // as specified in equation (12)
    thrust::device_vector<int> nodes = get_nodes_in_triangle(e1, e2, e3);
    for (int cls: {c1, c2}) {
        for (int node: nodes) {
            // TODO: currently quite inefficient because of the implementation of get_class_edge_index being O(N), N being the
            //  number of nodes
            int edge_idx = get_class_edge_index(node, cls);
            cdtf_counter[edge_idx] += 1;

            cdtf_correspondence.push_back(edge_idx);
        }
    }

    // Insert the edge indices in the order specified by equation 12 in the paper
    cdtf_correspondence.push_back(e1);
    cdtf_correspondence.push_back(e2);
    cdtf_correspondence.push_back(e3);
}

/**
 * Adds the triangle consisting of the three edges and *all* class combinations as a class-dependent triangle factor
 * @param e1 Index of the first edge
 * @param e2 Index of the second edge
 * @param e3 Index of the third edge
 */
void multiwaycut_message_passing::add_class_dependent_triangle_subproblems(int e1, int e2, int e3) {

    for (int c1 = 0; c1 < n_classes; ++c1) {
        for (int c2 = c1 + 1; c2 < n_classes; ++c2) {
            add_class_dependent_triangle_subproblems(e1, e2, e3, c1, c2);
        }
    }
}

/**
 * Adds the class-dependent triangle factors consisting of all triangles and the two given classes
 * @param c1 First class as integer in [0, K)
 * @param c2 Second class as integer in [0, K)
 */
void multiwaycut_message_passing::add_class_dependent_triangle_subproblems(int c1, int c2) {
    for (int idx = 0; idx < triangle_correspondence_12.size(); ++idx) {
        int e1 = triangle_correspondence_12[idx];
        int e2 = triangle_correspondence_13[idx];
        int e3 = triangle_correspondence_23[idx];

        // Only base graph triangles
        if (is_class_edge[e1] || is_class_edge[e2] || is_class_edge[e3])
            continue;

        add_class_dependent_triangle_subproblems(e1, e2, e3, c1, c2);
    }
}


/**
 * Adds the class-dependent triangle factors consisting of all triangles and the combinations of the given classes
 * @param classes
 */
void multiwaycut_message_passing::add_class_dependent_triangle_subproblems(std::vector<int> classes) {

    std::vector<int> sorted_classes(classes.size());
    std::partial_sort_copy(classes.begin(), classes.end(), sorted_classes.begin(), sorted_classes.end());

    for (int c1: sorted_classes) {
        for (int c2: sorted_classes) {
            if (c2 < c1) continue;  // This case will be encountered when the order is switched
            add_class_dependent_triangle_subproblems(c1, c2);
        }
    }
}

/**
 * Adds all possible class dependent triangle factors, i.e. all base graph triangles * 2 class combinations
 */
void multiwaycut_message_passing::add_class_dependent_triangle_subproblems() {
    std::vector<int>classes(n_classes);
    std::iota(classes.begin(), classes.end(), 0);
    add_class_dependent_triangle_subproblems(classes);
}
