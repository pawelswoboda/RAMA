#include "multiwaycut_message_passing.h"
#include "multiwaycut_text_parser.h"
#include "test.h"

void test_multiway_cut_repulsive_triangle(const float edge_cost, const float class_cost, const bool add_triangles = true) {
    int nodes = 3;
    int classes = 2;
    std::vector<int> src = {0, 0, 1};
    std::vector<int> dest = {1, 2, 2};
    std::vector<float> edge_costs = {edge_cost, edge_cost, edge_cost};
    std::vector<float> class_costs = {
        class_cost,class_cost, class_cost,class_cost, class_cost,class_cost
    };
    thrust::device_vector<int> i;
    thrust::device_vector<int> j;
    thrust::device_vector<float> costs;
    std::tie(i, j, costs) = mwc_to_coo(nodes, classes, class_costs, src, dest, edge_costs);

    thrust::device_vector<int> t1, t2, t3;
    if (add_triangles) {
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
    test(std::abs(initial_lb - (3* std::min(edge_cost, float(0.0)) + 6*std::min(class_cost, float(0.0)))) <= 1e-6, "Initial lb before reparametrization must be -3");

    int iterations = 10;
    double last_lb = initial_lb;
    for (int k = 0; k < iterations; ++k) {
        std::cout << "---------------" << "iteration = " << k << "---------------\n";
        mwcp.send_messages_to_triplets();
        double new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < 1e-6, "Lower bound did not increase after message to triplets");
        last_lb = new_lb;

        mwcp.send_messages_to_edges();
        new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < 1e-6, "Lower bound did not increase after message to edges");
        last_lb = new_lb;

        mwcp.send_messages_from_sum_to_edges();
        new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < 1e-6, "Lower bound did not increase after messages from class constraints");
        last_lb = new_lb;
    }

    const double final_lb = mwcp.lower_bound();
    std::cout << "final lb = " << final_lb << "\n";
//    test(std::abs(final_lb - 1.0) <= 1e-6, "Final lb after reparametrization must be -2");
}


void test_multiway_cut_2_nodes_2_classes(const float edge_cost, const float class_cost, const bool add_triangles = true) {
    int nodes = 2;
    int classes = 2;
    std::vector<int> src = {0};
    std::vector<int> dest = {1};
    std::vector<float> edge_costs = {edge_cost};
    std::vector<float> class_costs = {
        class_cost,class_cost, class_cost,class_cost
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
    test(std::abs(initial_lb - (std::min(edge_cost, float(0.0)) + 4*std::min(class_cost, float(0.0)))) <= 1e-6, "Initial lb before reparametrization must be 0");

    int iterations = 22;  // Need 21 iterations to reach sufficiently close approximation
    double last_lb = initial_lb;
    for (int k = 0; k < iterations; ++k) {
        std::cout << "---------------" << "iteration=" << k << "---------------\n";
        mwcp.send_messages_to_triplets();
        double new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < 1e-6, "Lower bound did not increase after message to triplets");
        last_lb = new_lb;

        mwcp.send_messages_to_edges();
        new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < 1e-6, "Lower bound did not increase after message to edges");
        last_lb = new_lb;

        mwcp.send_messages_from_sum_to_edges();
        new_lb = mwcp.lower_bound();
        test(new_lb > last_lb || std::abs(new_lb - last_lb) < 1e-6, "Lower bound did not increase after messages from class constraints");
        last_lb = new_lb;
    }

    const double final_lb = mwcp.lower_bound();
    std::cout << "final lb = " << final_lb << "\n";
    test(std::abs(final_lb - (std::min(edge_cost, float(0.0)) + 2*std::min(class_cost, float(0.0)))) <= 1e-6, "Final lb after reparametrization must be 2");
}


int main(int argc, char** argv)
{
    std::cout << "Testing repulsive triangle\n";
//    test_multiway_cut_repulsive_triangle(-1.0, 0.0);
//    test_multiway_cut_repulsive_triangle(-1.0, -1.0);
//    test_multiway_cut_repulsive_triangle(-1.0, 1.0);
    std::cout << "Testing 2 nodes 2 classes\n";
    test_multiway_cut_2_nodes_2_classes(1.0, 0.0);
    test_multiway_cut_2_nodes_2_classes(1.0, -1.0, false);
    test_multiway_cut_2_nodes_2_classes(1.0, 1.0);
}
