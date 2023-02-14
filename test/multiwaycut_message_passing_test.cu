#include "multiwaycut_message_passing.h"
#include "multiwaycut_text_parser.h"
#include "test.h"
#include <random>

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
  ALL_ON = 0,                             // Everything on i.e. normal triangles and summation messages
  NO_TRIANGLES = 1,                       // No triangles
  ONLY_BASE_TRIANGLES = 2,                // Only triangles in the base graph
  NO_MESSAGES_FROM_TRIANGLES = 4          // No message passing from the triangles is performed
};

constexpr enum TestOptions operator |(const enum TestOptions self, const enum TestOptions other) {
    return static_cast<TestOptions>(static_cast<int>(self) | static_cast<int>(other));
}
constexpr bool operator &(const enum TestOptions self, const enum TestOptions other) {
    return static_cast<TestOptions>(static_cast<int>(self) & static_cast<int>(other)) == other;
}

const TestOptions DEFAULT_OPTIONS = TestOptions::ALL_ON;


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
void test_multiway_cut_repulsive_triangle(
    const float edge_cost,
    const std::array<float, 3> c1,  // Class costs for class 1 for all nodes
    const std::array<float, 3> c2,
    const bool add_triangles
) {
    int nodes = 3;
    int classes = 2;
    std::vector<int> src = {0, 0, 1};
    std::vector<int> dest = {1, 2, 2};
    std::vector<float> edge_costs = {edge_cost, edge_cost, edge_cost};
    std::vector<float> class_costs = {
        c1[0], c2[0], c1[1], c2[1], c1[2], c2[2]
    };
    thrust::device_vector<int> i;
    thrust::device_vector<int> j;
    thrust::device_vector<float> costs;
    std::tie(i, j, costs) = mwc_to_coo(nodes, classes, class_costs, src, dest, edge_costs);

    thrust::device_vector<int> t1, t2, t3;
    if (add_triangles) {
//        t1 = std::vector<int>{0};
//        t2 = std::vector<int>{1};
//        t3 = std::vector<int>{2};  // TODO: extra test case with only base graph triangle(s)
        t1 = std::vector<int>{0, 0, 0, 0, 0, 1, 1};
        t2 = std::vector<int>{1, 1, 2, 1, 2, 2, 2};
        t3 = std::vector<int>{2, 3, 3, 4, 4, 3, 4};
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
        3 * std::min(edge_cost, 0.0f)  // We have three base edges
        // node-class edges are added to the edges, hence those need to be taken into account as well
        + std::min(c1[0], 0.0f) + std::min(c1[1], 0.0f) + std::min(c1[2], 0.0f)
        + std::min(c2[0], 0.0f) + std::min(c2[1], 0.0f) + std::min(c2[2], 0.0f) ;
    test(std::abs(initial_lb - expected_initial_lb) <= PRECISION, "Initial lb before reparametrization must be " + std::to_string(expected_initial_lb));

    //const double expected_final_lb =
    //    // Edge lower bound, for this test case the class edges should be zero in the end
    //    3 * std::min(edge_cost, 0.0f)
    //    // Class lower bound
    //    + std::min(c1[0], c2[0])
    //    + std::min(c1[1], c2[1])
    //    + std::min(c1[2], c2[2]);

    const double expected_final_lb = [&]()
    {
        // there are 8 possible node labelings determining uniquely the edge labelings: 
        // (0,0,0), 
        // (0,0,1), (0,1,0), (1,0,0), 
        // (0,1,1), (1,0,1), (1,1,0), 
        // (1,1,1)

        if (add_triangles)
        {
            float lb = std::numeric_limits<float>::infinity();

            for (int x1 = 0; x1 < 2; ++x1)
            {
                for (int x2 = 0; x2 < 2; ++x2)
                {
                    for (int x3 = 0; x3 < 2; ++x3)
                    {
                        const float cost = (x1 == 0 ? c1[0] : c2[0])
                            + (x2 == 0 ? c1[1] : c2[1])
                            + (x2 == 0 ? c1[2] : c2[2])
                            + (x1 != x2 ? edge_cost : 0.0)
                            + (x1 != x3 ? edge_cost : 0.0)
                            + (x2 != x3 ? edge_cost : 0.0);

                        lb = std::min(lb, cost);
                    }
                }
            }
            return lb;
        }
        else
        {
        return 3 * std::min(edge_cost, float(0.0)) + std::min(c1[0], c2[0]) + std::min(c1[1], c2[1]) + std::min(c1[2], c2[2]);
        }

    }();

    double last_lb = initial_lb;
    for (int k = 0; k < TEST_MAX_ITER; ++k)
    {
        std::cout << "---------------"
                  << "iteration = " << k << "---------------\n";
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
    std::cout << "final lb = " << final_lb << "\n\n";

    test(std::abs(final_lb - expected_final_lb) <= PRECISION, "Final lb after reparametrization must be " + std::to_string(expected_final_lb));
}


void test_multiway_cut_repulsive_triangle(
    const float edge_cost,
    const float class_cost,
    const bool add_triangles
) {
    test_multiway_cut_repulsive_triangle(
        edge_cost,
        std::array<float, 3>{{class_cost, class_cost, class_cost}},
        std::array<float, 3>{{class_cost, class_cost, class_cost}},
        add_triangles
    );
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


int main(int argc, char** argv)
{
    std::cout << "Testing repulsive triangle\n";
//    test_multiway_cut_repulsive_triangle(-1.0, 0.0, false);
//    test_multiway_cut_repulsive_triangle(-1.0, -1.0, false);
//    test_multiway_cut_repulsive_triangle(-1.0, 1.0, false);
    test_multiway_cut_repulsive_triangle(-1.0, 0.0, true);
    // fails
    test_multiway_cut_repulsive_triangle(-1.0, 1.0, true);
    // converges
    test_multiway_cut_repulsive_triangle(-1.0, -1.0, true);
    std::cout << "Testing 2 nodes 2 classes\n";
    test_multiway_cut_2_nodes_2_classes(1.0, 0.0, false);
    test_multiway_cut_2_nodes_2_classes(1.0, 0.0, true);
    test_multiway_cut_2_nodes_2_classes(1.0, -1.0, false);
    // Next tests fails
    test_multiway_cut_2_nodes_2_classes(1.0, -1.0, true);
    test_multiway_cut_2_nodes_2_classes(1.0, 1.0, false);
    test_multiway_cut_2_nodes_2_classes(1.0, 1.0, true);


    // Random number tests:
    std::mt19937 gen(17410);
    std::uniform_real_distribution<> dist(-5, 5);
    for (int i = 0; i < TEST_RAND_ITER; ++i) {
        float c = dist(gen);
        std::cout << "Testing with class_cost = " << c << "\n";
        test_multiway_cut_repulsive_triangle(-1.0, c, false);
        test_multiway_cut_repulsive_triangle(-1.0, c, true);
        test_multiway_cut_repulsive_triangle(-1.0, -c, false);
        test_multiway_cut_repulsive_triangle(-1.0, -c, true);

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
        std::array<float, 3> c1 = {{static_cast<float>(dist(gen)), static_cast<float>(dist(gen)), static_cast<float>(dist(gen))}};
        std::array<float, 3> c2 = {{static_cast<float>(dist(gen)), static_cast<float>(dist(gen)), static_cast<float>(dist(gen))}};
        std::cout << "Testing with class_cost = "
                  << c1[0] << "," << c1[1] << "," << c1[2]
                  << " "
                  << c2[0] << "," << c2[1] << "," << c2[2] << "\n";

        test_multiway_cut_repulsive_triangle(-1.0, c1, c2, false);
        test_multiway_cut_repulsive_triangle(-1.0, c1, c2, true);
        test_multiway_cut_repulsive_triangle(-1.0, {{-c1[0], -c1[1], -c1[2]}}, {{-c2[0], -c2[1], -c2[2]}}, false);
        test_multiway_cut_repulsive_triangle(-1.0, {{-c1[0], -c1[1], -c1[2]}}, {{-c2[0], -c2[1], -c2[2]}}, true);
    }
}
