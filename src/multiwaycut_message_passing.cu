#include "multiwaycut_message_passing.h"
#include "rama_utils.h"
#include <unordered_set>
#include <algorithm>
#include <thrust/inner_product.h>

// An edge is a node class-edge iff the destination node value is larger than the largest base node
// which are assigned values from [0...n_nodes]
// TODO: Replace all occurences of the macro with the is_class_edge vector
#define IS_CLASS_EDGE(dest) (dest >= n_nodes)
#define CHOOSE2(N) (N*(N-1) / 2)


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
        class_costs(_n_nodes * _n_classes, 0),  // Zero initialize summation constraints
        is_class_edge(create_class_edge_mask(i, j, [_n_nodes](int source, int dest) {return dest >= _n_nodes;} )),
        base_edge_counter(edge_counter.size(), 0),  // In how many triangles in the base graph an edge is present
        node_counter(n_nodes, 0),  // In how many triangles in the base graph a node is present
        _k_choose_2(CHOOSE2(n_classes)),
        // We have cdtf costs for each 2 class combinations all the edges of all base graph triangles
        // triangle correspondence includes not only base graph triangles, but we will ignore
        // those when updating the costs
        cdtf_costs(CHOOSE2(n_classes) * triangle_correspondence_12.size() * 9, 0.0)
        {

    print_vector(i, "sources");
    print_vector(j, "dest   ");

    // Calculate how often each edge is part of a base graph triangle
    int base_triangles = 0;
    thrust::copy(edge_counter.begin(), edge_counter.end(), base_edge_counter.begin());
    // Should probably be parallelized
    for (int idx = 0; idx < triangle_correspondence_12.size(); ++idx) {
        int e12 = triangle_correspondence_12[idx];
        int e13 = triangle_correspondence_13[idx];
        int e23 = triangle_correspondence_23[idx];
        // If any edge is a class edge this is not a base graph triangle
        if (is_class_edge[e12]
        || is_class_edge[e13]
        || is_class_edge[e23]
        ) {
            base_edge_counter[e12] -= 1;
            base_edge_counter[e13] -= 1;
            base_edge_counter[e23] -= 1;
        } else {
            base_triangles += 1;
            // get the nodes of this triangle
            std::unordered_set<int> nodes = {i[e12], j[e12], i[e13], j[e13], i[e23], j[e23]};

            for (int node: nodes) {
                assert(node < n_nodes);  // There should be no class node in this triangle
                node_counter[node] += 1;
            }
        }
    }
    print_vector(base_edge_counter, "Edge counter base graph");
    print_vector(node_counter, "Node counter");
}


/**
 * Partitions the edges by node. Assumes that i, j is sorted by the node values, this is the case after initializing
 * multicut
 * @return the number of edges per node and the offset at which the edges for a node u
 * start in the i or j array start.
 */
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> multiwaycut_message_passing::get_node_partition()
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

    // Number of cdtf factors
    int size = CHOOSE2(n_classes) * triangle_correspondence_12.size();

    // We parallelize over the class-dependent triangle factors as (K, 2) * T >> 37 in most cases
    thrust::device_vector<float> results(size);
    auto idx = thrust::make_counting_iterator<int>(0);
    cdtf_lower_bound_func func({
        thrust::raw_pointer_cast(results.data()), thrust::raw_pointer_cast(cdtf_costs.data())
    });
    thrust::for_each(idx, idx + size, func);
    return thrust::reduce(results.begin(), results.end(), 0.0);
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
        double cdtflb = cdtf_lower_bound();
        double res = edge_lower_bound() + tlb + clb + cdtflb;
        printf("%f+%f+%f+%f = %f\n", edge_lower_bound(), tlb, clb, cdtflb, res);
        return res;
    } else {
        std::cout << "No classes provided. Defaulting to multicut\n";
        return multicut_message_passing::lower_bound();
    }
}



/**
 * Message passing functor from edges to triplets
 */
struct increase_triangle_costs_func_mwc {
    float* edge_costs;
    int* edge_counter;

    bool* is_class_edge;

    __device__ void operator()(const thrust::tuple<int,float&> t) const
    {
        const int edge_idx = thrust::get<0>(t);
        float& triangle_cost = thrust::get<1>(t);

        // TODO: Change weights based if part of new subproblem
        if (is_class_edge[edge_idx]) {
            float update = edge_costs[edge_idx]/(1.0f + float(edge_counter[edge_idx]));
            triangle_cost += update;
        } else {
            assert(edge_counter[edge_idx] > 0);
            triangle_cost += edge_costs[edge_idx] / float(edge_counter[edge_idx]);
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
        // TODO: Change weights based if part of new subproblem
        class_costs[class_idx] += edge_cost / (1.0f + float(edge_count));
    }
};

struct increase_cdtf_costs_func{
    // --- For offset calculation
    int n_classes;
    int n_triangles;
    int *start;
    int *sizes;
    // --- To check if the triangle include class edges
    bool *is_class_edge;
    // --- For the calculation of the messages/weights
    float *edge_costs;
    int *edge_counter;
    int *node_counter;
    // --- To get the starts and ends of the edges, which in turn index the counters above
    int *sources;
    int *dests;
    // ---
    float *cdtf_costs;

    // For each class dependent triangle factor we consider 9 edges - the base graph edges + the node class edges
    const int VALS_PER_TRIANGLE = 9;



    // The layout of cdtf_costs is as follows:
    // We have K classes, hence for each triangle we have (K, 2) possible combinations
    // We first index by the classes, i.e. given combination c_i c_j, i<j
    //  +----+----+----+----+-------+
    //  | K-1| K-2| K-3| ...|K-(K-1)|
    //  +----+----+----+----+-------+
    //  |    |    |         |
    //  |    |    |         +>c(K-1)cK
    //  |    |    +>c3c4  ... c3cK
    //  |    +>c2c3 c2c4  ... c2cK
    //  +>c1c2 c1c3 c1c4  ... c1cK
    //
    //  The start of the cic(i+1) combination can be calculated as (i choose 2) = i*(i-1)/2
    //  Now each cicj block looks as follows, where u<v<w and these are part of some triangle T
    // +--+--+--+---+--+
    // |T1|T2|T3|...|Tn|
    // +--+--+--+-+-+--+
    //            |
    //            | +---+---+---+---+---+---+---+---+---+
    //            +>|uv |uw |vw |uc1|uc2|vc1|vc2|wc1|wc2|
    //              +---+---+---+---+---+---+---+---+---+
    //
    //  Thus we need to take these elements into account when calculating the index:
    //  index(cicj) = 9n*[(1/2)*i*(i-1) + j]
    //
    //  The index for a triangle Tm is given as:
    //  index(Tm) = index(cicj) + m * 9n
    __host__ __device__ void operator()(const thrust::tuple<int, int, int, int> t) const {
        int t12 = thrust::get<0>(t);
        int t13 = thrust::get<1>(t);
        int t23 = thrust::get<2>(t);
        // The index of this triangle in the triangle_correspondence_xy vector
        int triangle_idx = thrust::get<3>(t);

        // We only consider base graph triangles
        if (is_class_edge[t12]
        || is_class_edge[t13]
        || is_class_edge[t23]
        ) return;

        // Update the costs of the edges in this triangle for all class combinations
        for (int c_i = 0; c_i < n_classes; ++c_i) {
            for (int c_j = c_i+1; c_j < n_classes; ++c_j) {
                const int offset = CHOOSE2(c_i) * n_triangles * VALS_PER_TRIANGLE;  // Start of the c_i block in the cost array
                const int c_j_start = offset + c_j * n_triangles * VALS_PER_TRIANGLE;  // Start of the c_j block in the c_i block
                const int triangle_start = c_j_start + triangle_idx * VALS_PER_TRIANGLE;

                // t12
                cdtf_costs[triangle_start] = edge_costs[t12]/(
                    1.0f + float(edge_counter[t12])
                    + float(node_counter[sources[t12]]) * float(n_classes - 1)
                );
                // t13
                cdtf_costs[triangle_start + 1] = edge_costs[t13]/(
                    1.0f + float(edge_counter[t13])
                    + float(node_counter[sources[t13]]) * float(n_classes - 1)
                );
                // t23
                cdtf_costs[triangle_start + 2] = edge_costs[t23]/(
                    1.0f + float(edge_counter[t23])
                    + float(node_counter[sources[t23]]) * float(n_classes - 1)
                );
                // We consider the nodes of the edges in ascending order
                const int size = 6;
                int nodes[size] = { sources[t12], dests[t12], sources[t13], dests[t13], sources[t23], dests[t23] };
                int unique_nodes[3] = {};
                // FIX: Thrust algorithms normally are intended to work on the host not the device
                // but using sequential execution policy works
                thrust::sort(thrust::seq, nodes, nodes + size);
                thrust::unique_copy(thrust::seq, nodes, nodes + size, unique_nodes);

                 // node-class costs start after the base graph edges
                const int class_edges_start = triangle_start + 3;
                for (int i = 0; i < 3; ++i) {
                    int source = unique_nodes[i];

                    // Start contains the index of the first edge with `source` as the beginning
                    // Size is the number of edges that contain this node (including class edges)
                    // The last `n_classes` elements in the range [start[source], start[source]+size[source])
                    // include the node-class edges in ascending order by, hence the following returns the index
                    // of the node-class edge.
                    int ic_i_idx = start[source] + sizes[source] - n_classes + c_i;
                    int ic_j_idx = start[source] + sizes[source] - n_classes + c_j;

                    // from unique_nodes[i] to c_i
                    cdtf_costs[class_edges_start + 2*i] += edge_costs[ic_i_idx] / (
                        1.0f + float(edge_counter[ic_i_idx])
                        + float(node_counter[source]) * float(n_classes - 1)
                    );
                    // from unique_nodes[i] to c_j
                    cdtf_costs[class_edges_start + 1 + 2*i] += edge_costs[ic_j_idx] / (
                        1.0f + float(edge_counter[ic_j_idx])
                        + float(node_counter[source]) * float(n_classes - 1)
                    );
                }
            }
        }
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

    // Messages to triangles
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

    // Messages to summation constraints
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

    // Send messaged to class dependent triangle factors
    // Cannot write cdtf costs if no triangles are present
    if (!cdtf_costs.empty()){
        thrust::counting_iterator<int> triangle_idx(0);
        auto first = thrust::make_zip_iterator(thrust::make_tuple(
            triangle_correspondence_12.begin(), triangle_correspondence_13.begin(), triangle_correspondence_23.begin(),
            triangle_idx
        ));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(
            triangle_correspondence_12.end(), triangle_correspondence_13.end(), triangle_correspondence_23.end(),
            triangle_idx + triangle_correspondence_12.size()
        ));
        thrust::device_vector<int> _node;
        thrust::device_vector<int> start;
        thrust::device_vector<int> size;
        std::tie(_node, start, size) = get_node_partition();
        increase_cdtf_costs_func func({
            n_classes, n_nodes,
            thrust::raw_pointer_cast(start.data()), thrust::raw_pointer_cast(size.data()),
            thrust::raw_pointer_cast(is_class_edge.data()),
            thrust::raw_pointer_cast(edge_costs.data()),
            thrust::raw_pointer_cast(edge_counter.data()), thrust::raw_pointer_cast(node_counter.data()),
            thrust::raw_pointer_cast(i.data()), thrust::raw_pointer_cast(j.data()),
            thrust::raw_pointer_cast(cdtf_costs.data())
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
    send_messages_to_triplets();
    send_messages_to_edges();
    send_messages_from_sum_to_edges();
}

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

