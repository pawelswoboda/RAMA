#include "multiwaycut_message_passing.h"
#include "multiwaycut_text_parser.h"
#include "test.h"
#include <random>
#include <stdexcept>

#define TEST_MAX_ITER 20
#define TEST_RAND_ITER 10   // How many test with random value
#define PRECISION 1e-5

/**
 * Allows easily setting and checking different options for the test cases
 *
 * Use `|` operator to combine multiple flags, note that some flags such as NO_TRIANGLES and ONLY_BASE_TRIANGLES
 * are exclusive and should not combined. Currently NO_TRIANGLES is checked before ONLY_BASE_TRIANGLES
 *
 * Use `&` operator to check if a option is turned on:
 * options & NO_TRIANGLES == true iff the second bit of options is turned on (options & NO_TRIANGLES) == NO_TRIANGLES
 * Note that the order is important, i.e. NO_TRIANGLES & options will lead to non expected behaviour, only being true
 * if the only flag in options is NO_TRIANGLES
 */
enum class TestOptions {
  ALL_ON = 0,                                     // Everything on i.e. normal triangles and summation messages
  NO_TRIANGLES = (1 << 0),                        // No triangles
  ONLY_BASE_TRIANGLES = (1 << 1),                 // Only triangles in the base graph
  NO_MESSAGES_FROM_TRIANGLES = (1 << 2),          // No message passing from the triangles is performed
  NO_MESSAGES_FROM_EDGES = (1 << 3),              // No messages from edges to triplets / summation constraints
  NO_MESSAGES_FROM_SUM_CONSTRAINTS = (1 << 4),    // No messages from the summation constraints back to the edges
  ALL_CDTF_PROBLEMS = (1 << 5)                    // Add all cdtf problems in the graph
};

constexpr enum TestOptions operator |(const enum TestOptions self, const enum TestOptions other) {
    return static_cast<TestOptions>(static_cast<int>(self) | static_cast<int>(other));
}
constexpr bool operator &(const enum TestOptions self, const enum TestOptions other) {
    return static_cast<TestOptions>(static_cast<int>(self) & static_cast<int>(other)) == other;
}

const TestOptions DEFAULT_OPTIONS = TestOptions::ALL_ON;  // TODO: Consider adding ALL_CDTF_PROBLEMS as default


/**
 * Calculates N-ary the cartesian power of the set {0, ..., K-1}, i.e. {0, ..., K-1}^N
 * @return Vector containing vectors of size N
 */
std::vector<std::vector<int>> product(const int K, const int N) {
    std::vector<std::vector<int>> result;
    // Start with the set {(0), ..., (K-1)}
    for (int i = 0; i < K; ++i) {
        result.push_back({ i });
    }

    // Iteratively add {0, ..., K-1} to each value in result
    for (int i = 1; i < N; ++i) {
        std::vector<std::vector<int>> temp;
        for (std::vector<int>& prod : result) {
            // Add all {0, ..., K-1} values to each element of {K-1}^(i-1)
            for (int j = 0; j < K; ++j) {
                auto set = prod;
                set.push_back(j);
                // Add the new product
                temp.push_back(set);
            }
        }
        // Result now is {K-1}^i
        result = temp;
    }
    return result;
}


/**
 * Calculates the expected lower bound for a triangle with K classes by iterating all possible node labelings
 * and adding the costs of the cut edges
 * @param K The number of classes
 * @param source Vector with sources of each edge
 * @param dest Vector with destination of each edge
 * @param cost Vector with the cost of each edge
 * @return The lower bound
 */
float triangle_expected_lb(
    const int K,
    const thrust::device_vector<int> &source,
    const thrust::device_vector<int> &dest,
    const thrust::device_vector<float> &cost
) {
    assert(source.size() == dest.size() && dest.size() == cost.size());

    float lb = std::numeric_limits<float>::infinity();
    // the labels are the cartesian product {0, ..., K-1}^3
    for (std::vector<int>&label: product(K, 3)) {
        // Iterate all the edges for this label and calculate the costs
        float labeling_cost = 0.0;
        for (int edge = 0; edge < source.size(); ++edge) {
            int left = source[edge];
            int right = dest[edge];
            assert(left < 3);  // Only base nodes on the left side of an edge
            assert(right < 3 + K); // Class nodes are encoded by n_nodes-1 (for triangle n_nodes=3) + n_classes

            if (right >= 3) {
                // We have a base-class edge -> we compare the label with the class index
                right = right - 3;  // Class index calculation, see above

                if (label[left] != right) {
                    labeling_cost += cost[edge];
                }

            } else {
                // We have a base edge ->  we compare the label of the two nodes
                if (label[left] != label[right]) {
                    labeling_cost += cost[edge];
                }
            }
        }
        lb = std::min(lb, labeling_cost);
    }
    return lb;
}

float expected_initial_lb(
    const thrust::device_vector<float> &cost
) {
    float result = 0.0;
    for (float c: cost) {
        result += std::min(c, 0.0f);
    }
    return result;
}


void test_multiway_cut_2_nodes_2_classes(
    const float edge_cost,
    const std::array<float, 2> c1,  // Costs for class 1, one entry for each node
    const std::array<float, 2> c2,  // Costs for class 2
    const bool add_triangles
) {
    int nodes = 2;
    int classes = 2;
    std::vector<int> src = {0};
    std::vector<int> dest = {1};
    std::vector<float> edge_costs = {edge_cost};
    std::vector<float> class_costs = {
        c1[0], c2[0], c1[1], c2[1]
    };
    thrust::device_vector<int> i;
    thrust::device_vector<int> j;
    thrust::device_vector<float> costs;
    std::tie(i, j, costs) = mwc_to_coo(nodes, classes, class_costs, src, dest, edge_costs);

    thrust::device_vector<int> t1, t2, t3;
    if (add_triangles) {
        t1 = std::vector<int>{0, 0};
        t2 = std::vector<int>{1, 1};
        t3 = std::vector<int>{2, 3};
    } else {
        t1 = std::vector<int>{};
        t2 = std::vector<int>{};
        t3 = std::vector<int>{};
    }


    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
    multiwaycut_message_passing mwcp(A, nodes, classes, std::move(t1), std::move(t2), std::move(t3));

    const double initial_lb = mwcp.lower_bound();
    std::cout << "initial lb = " << initial_lb << "\n";
    const double expected_initial_lb =
        // Initial lower bound only considers the edge costs
        std::min(edge_cost, 0.0f)
        // node-class edges are added to the edges, hence those need to be taken into account as well
        + std::min(c1[0], 0.0f)  + std::min(c1[1], 0.0f)  + std::min(c2[0], 0.0f)  + std::min(c2[1], 0.0f);
    test(std::abs(initial_lb - expected_initial_lb) <= PRECISION, "Initial lb before reparametrization must be " + std::to_string(expected_initial_lb));

    // Lower bound is edge costs + the two smallest class costs
    const double expected_final_lb =
        // Edge lower bound, for this test case the class edges should be zero in the end
        std::min(edge_cost, 0.0f)
        // Class lower bound
        + std::min(c1[0], c2[0])
        + std::min(c1[1], c2[1]);

    double last_lb = initial_lb;
    for (int k = 0; k < TEST_MAX_ITER; ++k) {
        std::cout << "---------------" << "iteration=" << k << "---------------\n";
        mwcp.send_messages_to_triplets();
        double new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < PRECISION, "Lower bound did not increase after message to triplets");
        last_lb = new_lb;

        mwcp.send_messages_to_edges();
        new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < PRECISION, "Lower bound did not increase after message to edges");
        last_lb = new_lb;

        mwcp.send_messages_from_sum_to_edges();
        new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < PRECISION, "Lower bound did not increase after messages from class constraints");
        last_lb = new_lb;

        // short circuit if we encounter the optimal lower bound earlier
        if (std::abs(last_lb - expected_final_lb) <= PRECISION)
            break;
    }

    const double final_lb = mwcp.lower_bound();
    std::cout << "final lb = " << final_lb << "\n";
    test(std::abs(final_lb - expected_final_lb) <= PRECISION, "Final lb after reparametrization must be " + std::to_string(expected_final_lb));
}


void test_multiway_cut_2_nodes_2_classes(
    const float edge_cost,
    const float class_cost,
    const bool add_triangles
) {
    test_multiway_cut_2_nodes_2_classes(
        edge_cost,
        std::array<float, 2>{{class_cost, class_cost}},
        std::array<float, 2>{{class_cost, class_cost}},
        add_triangles
    );
}


void two_nodes_2_classes_tests() {
    std::cout << "Testing 2 nodes 2 classes\n";
    test_multiway_cut_2_nodes_2_classes(1.0, 0.0, false);
    test_multiway_cut_2_nodes_2_classes(1.0, 0.0, true);
    test_multiway_cut_2_nodes_2_classes(1.0, -1.0, false);
    // Next tests fail
    test_multiway_cut_2_nodes_2_classes(1.0, -1.0, true);
    test_multiway_cut_2_nodes_2_classes(1.0, 1.0, false);
    test_multiway_cut_2_nodes_2_classes(1.0, 1.0, true);
}

/**
 * Tests a triangle with K classes
 * @param edge_costs Costs in the base graph, in order 01 02 12
 * @param cls_costs Class costs for each node, i.e. 3xK vector with each row being the class costs in order 0, ..., K-1
 * @param K Number of classes
 * @param options Options for this test
 * @param cdtf_classes For which classes the class dependent triangle factors should be added, with class encoded by
 *                      an integer in the range 0, ..., K-1
 */
void test_triangle(
    const std::vector<float> edge_costs,
    const std::vector<std::vector<float>> cls_costs,
    const int K,
    const TestOptions options = DEFAULT_OPTIONS,
    const std::vector<int> cdtf_classes = {}
){
    const int n_nodes = 3;

    if (K < 2) {
        throw std::invalid_argument("Multiway cut with 1 class makes no sense");
    }

    if (edge_costs.size() != n_nodes) {
        throw std::invalid_argument("Size of edge_costs has to be 3");
    }
    if (cls_costs.size() != n_nodes) {
        throw std::invalid_argument("cls_costs needs one vector with costs for each node");
    }
    if (!cdtf_classes.empty() && cdtf_classes.size() > K) {
        throw std::invalid_argument("Cannot have more classes in the cdtf problems than in the instance");
    }
    if (!cdtf_classes.empty() && cdtf_classes.size() < 2) {
        throw std::invalid_argument("Class-dependent triangle factors need at least two classes");
    }

    if (!cdtf_classes.empty() && options & TestOptions::NO_TRIANGLES) {
        throw std::invalid_argument("Class-dependent triangle factors are directly related to the triangles"
                                    "and cannot simply be turned off as the triangles passed to the class are"
                                    "basis for constructing the class-dependent triangle factors.");
    }


    // Set up the base triangle
    std::vector<int> src = {0, 0, 1};
    std::vector<int> dest = {1, 2, 2};
        // We store the class costs consecutively, first all class costs for the first node than the second ...
    std::vector<float> class_costs;
    for (int i = 0; i < n_nodes; ++i) {
        class_costs.insert(class_costs.end(), cls_costs[i].begin(), cls_costs[i].end());
    }

    // Convert to expected format
    thrust::device_vector<int> i;
    thrust::device_vector<int> j;
    thrust::device_vector<float> costs;
    std::tie(i, j, costs) = mwc_to_coo(n_nodes, K, class_costs, src, dest, edge_costs);

    // Create triangles
    thrust::device_vector<int> t1, t2, t3;
    if (options & TestOptions::ONLY_BASE_TRIANGLES) {
        t1 = std::vector<int>{0};
        t2 = std::vector<int>{1};
        t3 = std::vector<int>{2};
    }
    else {
        t1 = std::vector<int>{0};
        t2 = std::vector<int>{1};
        t3 = std::vector<int>{2};
        // Add class triangles
        // We choose 2 nodes from the base graph and then each class once
        for (int u = 0; u < n_nodes; ++u) {
            // No need to consider v < u+1 as this leads to symmetric triangles i.e. (u,v,ci) and (v, u, ci)
            for (int v = u+1; v < n_nodes; ++v) {
                for (int cls = n_nodes; cls < n_nodes + K; ++cls) {
                    t1.push_back(u);
                    t2.push_back(v);
                    t3.push_back(cls);
                }
            }
        }
    }

    MWCOptions mwc_options = MWCOptions::VERBOSE;
    if (options & TestOptions::NO_TRIANGLES) {
        // If we just set the triangles to be empty we will not be able to create
        // the class dependent triangle factors easily hence we just pretend that the
        // triangles don't exist.
        mwc_options = MWCOptions::VERBOSE | MWCOptions::IGNORE_TRIANGLES;
    }

    //
    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
    multiwaycut_message_passing mwcp(A, n_nodes, K, std::move(t1), std::move(t2), std::move(t3), mwc_options);

    if (options & TestOptions::ALL_CDTF_PROBLEMS) {
        mwcp.add_class_dependent_triangle_subproblems();
    }
    else if (!cdtf_classes.empty()) {
        mwcp.add_class_dependent_triangle_subproblems(cdtf_classes);
    }

    // Check initial lower bound
    const double initial_lb = mwcp.lower_bound();
    const double expected_initial = expected_initial_lb(costs);
    std::cout << "initial lb = " << initial_lb << "\n";
    test(std::abs(initial_lb - expected_initial) <= PRECISION,
         "Initial lb before reparametrization must be " + std::to_string(expected_initial)
    );

    // Perform message passing iterations
    const double expected_final_lb = triangle_expected_lb(K, i, j, costs);
    double last_lb = initial_lb;
    for (int k = 0; k < TEST_MAX_ITER; ++k)
    {
        std::cout << "---------------" << "iteration = " << k << "---------------\n";
        double new_lb = last_lb;
        if (options & TestOptions::NO_MESSAGES_FROM_EDGES) {
            std::cout << "Skipping messages from edges\n";
        } else {
            mwcp.send_messages_to_triplets();
            new_lb = mwcp.lower_bound();
            test(new_lb > last_lb || std::abs(new_lb - last_lb) < PRECISION,
                 "Lower bound did not increase after message to triplets");
            last_lb = new_lb;
        }

        if (options & TestOptions::NO_MESSAGES_FROM_TRIANGLES) {
            std::cout << "Skipping messages from triangles\n";
        } else {
            mwcp.send_messages_to_edges();
            new_lb = mwcp.lower_bound();
            test(new_lb > last_lb || std::abs(new_lb - last_lb) < PRECISION,
                 "Lower bound did not increase after message to edges");
            last_lb = new_lb;
        }

        if (options & TestOptions::NO_MESSAGES_FROM_SUM_CONSTRAINTS) {
            std::cout << "Skipping messages from summation constraints\n";
        } else {
            mwcp.send_messages_from_sum_to_edges();
            new_lb = mwcp.lower_bound();
            test(new_lb > last_lb || std::abs(new_lb - last_lb) < PRECISION,
                 "Lower bound did not increase after messages from class constraints");
            last_lb = new_lb;
        }

        // short circuit if we encounter the optimal lower bound earlier
        if (std::abs(last_lb - expected_final_lb) <= PRECISION)
            break;
    }

    const double final_lb = mwcp.lower_bound();
    std::cout << "final lb = " << final_lb << "\n\n";

    test(std::abs(final_lb - expected_final_lb) <= PRECISION,
         "Final lb after reparametrization must be " + std::to_string(expected_final_lb));
}

/**
 * Tests a triangle with the same costs in the base graph but possibly different costs for each class, i.e.
 * if class_costs[0] = -1 all edges to class 0 will have costs -1
 *
 * @param edge_cost Cost applied to each edge in the base graph
 * @param class_costs Vector with K costs, one for each class
 * @param options Options for test_triangle, see TestOptions declaration
 * @param cdtf_classes For which classes the class dependent triangle factors should be added, with class encoded by
 *                      an integer in the range 0, ..., K-1
 */
void test_triangle_N_classes_between_class_difference(
    const float edge_cost,
    const std::vector<float> class_costs,
    const TestOptions options = DEFAULT_OPTIONS,
    const std::vector<int> cdtf_classes = {}
) {

    // Create the per node class costs
    std::vector<std::vector<float>> costs;
    for (int i = 0; i < 3; ++i) {
        std::vector<float> temp;
        for (float cost: class_costs) {
            temp.push_back(cost);
        }
        costs.push_back(temp);
    }

    test_triangle(
        std::vector<float>({edge_cost, edge_cost, edge_cost}),
        costs,
        class_costs.size(),
        options,
        cdtf_classes
    );
}

/**
 * Tests a triangle with the same costs in the base graph and all class edges having equal cost
 * @param edge_cost Cost applied to each edge in the base graph
 * @param class_cost Cost applied to all base-class edges
 * @param options Options for test_triangle, see TestOptions declaration
 * @param cdtf_classes For which classes the class dependent triangle factors should be added, with class encoded by
 *                      an integer in the range 0, ..., K-1
 */
void test_triangle_2_classes_same_costs(
    const float edge_cost,
    const float class_cost,
    const TestOptions options = DEFAULT_OPTIONS,
    const std::vector<int> cdtf_classes = {}
) {
    test_triangle_N_classes_between_class_difference(
        edge_cost,
        std::vector<float>({class_cost, class_cost}),
        options,
        cdtf_classes
    );
}


void repulsive_triangle_tests(){

    std::cout << "Testing repulsive triangle\n";
    test_triangle_N_classes_between_class_difference(
        -1, std::vector<float>({-100.0, -10.0, -10.0}),
        TestOptions::NO_MESSAGES_FROM_TRIANGLES | TestOptions::NO_TRIANGLES | TestOptions::ALL_CDTF_PROBLEMS
    );

    test_triangle_2_classes_same_costs(-1.0, 0.0, TestOptions::ONLY_BASE_TRIANGLES);
    test_triangle_2_classes_same_costs(-1.0, -1.0, TestOptions::ONLY_BASE_TRIANGLES);
    test_triangle_2_classes_same_costs(-1.0, 1.0, TestOptions::ONLY_BASE_TRIANGLES);

    // fail both
    test_triangle_2_classes_same_costs(-1.0, 0.0);
    test_triangle_2_classes_same_costs(-1.0, 1.0);
    // converges
    test_triangle_2_classes_same_costs(-1.0, -1.0);
}


int main(int argc, char** argv)
{
    repulsive_triangle_tests();
    two_nodes_2_classes_tests();

    // Random number tests:
    std::mt19937 gen(17410);
    std::uniform_real_distribution<> dist(-5, 5);
    for (int i = 0; i < TEST_RAND_ITER; ++i) {
        float c = dist(gen);
        std::cout << "Testing with class_cost = " << c << "\n";
        test_triangle_2_classes_same_costs(-1.0, c, TestOptions::ONLY_BASE_TRIANGLES);
        test_triangle_2_classes_same_costs(-1.0, -c, TestOptions::ONLY_BASE_TRIANGLES);
        test_triangle_2_classes_same_costs(-1.0, c);
        test_triangle_2_classes_same_costs(-1.0, -c);

        test_multiway_cut_2_nodes_2_classes(1.0, c, false);
        test_multiway_cut_2_nodes_2_classes(1.0, c, true);
        test_multiway_cut_2_nodes_2_classes(1.0, -c, false);
        test_multiway_cut_2_nodes_2_classes(1.0, -c, true);
    }

    // Completely random class costs
    for (int i = 0; i < TEST_RAND_ITER; ++i) {
        std::array<float, 2> c1 = {{static_cast<float>(dist(gen)), static_cast<float>(dist(gen))}};
        std::array<float, 2> c2 = {{static_cast<float>(dist(gen)), static_cast<float>(dist(gen))}};
        std::cout << "Testing with class_cost = " << c1[0] << "," << c1[1] << " " << c2[0] << "," << c2[1] <<  "\n";

        test_multiway_cut_2_nodes_2_classes(1.0, c1, c2, false);
        test_multiway_cut_2_nodes_2_classes(1.0, c1, c2, true);
        test_multiway_cut_2_nodes_2_classes(1.0, {{-c1[0], -c1[1]}}, {{-c2[0], -c2[1]}}, false);
        test_multiway_cut_2_nodes_2_classes(1.0, {{-c1[0], -c1[1]}}, {{-c2[0], -c2[1]}}, true);
    }
    for (int i = 0; i < TEST_RAND_ITER; ++i) {
        std::vector<float> node1 = {static_cast<float>(dist(gen)), static_cast<float>(dist(gen))};
        std::vector<float> node2 = {static_cast<float>(dist(gen)), static_cast<float>(dist(gen))};
        std::vector<float> node3 = {static_cast<float>(dist(gen)), static_cast<float>(dist(gen))};
        std::cout << "Testing with class_cost = "
                  << node1[0] << "," << node1[1]
                  << " "
                  << node2[0] << "," << node2[1]
                  << " "
                  << node3[0] << "," << node3[1] << "\n";

        std::vector<std::vector<float>> class_costs({node1, node2, node3});
        test_triangle(
            std::vector<float>({-1.0, -1.0, -1.0}),
            class_costs,
            2,
            TestOptions::ONLY_BASE_TRIANGLES
        );
        test_triangle(
            std::vector<float>({-1.0, -1.0, -1.0}),
            class_costs,
            2
        );
    }
}
