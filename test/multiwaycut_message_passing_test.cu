#include "multiwaycut_message_passing.h"
#include "multiwaycut_text_parser.h"


void test_multicut() {
    int nodes = 4;
    int classes = 0;
    thrust::device_vector<int> i = std::vector<int>{0,0,0,1,2};
    thrust::device_vector<int> j = std::vector<int>{1,2,3,3,3};
    thrust::device_vector<float> edge_costs = std::vector<float>{1.0,1.0,-2.0,1.0,1.0};
    thrust::device_vector<int> t1 = std::vector<int>{0,0};
    thrust::device_vector<int> t2 = std::vector<int>{1,2};
    thrust::device_vector<int> t3 = std::vector<int>{3,3};
    dCOO A(i.begin(), i.end(), j.begin(), j.end(), edge_costs.begin(), edge_costs.end(), true);
    multiwaycut_message_passing mwcp(A, nodes, classes, std::move(t1), std::move(t2), std::move(t3));

    const double initial_lb = mwcp.lower_bound();
    std::cout << "initial lb = " << initial_lb << "\n";
    if(std::abs(initial_lb - (-2.0)) > 1e-6)
        throw std::runtime_error("initial lb before reparametrization must be -2");

    mwcp.send_messages_to_triplets();

    mwcp.iteration();
    mwcp.iteration();
    mwcp.iteration();

    const double final_lb = mwcp.lower_bound();
    std::cout << "final lb = " << final_lb << "\n";
    if(std::abs(final_lb) > 1e-6)
        throw std::runtime_error("final lb after reparametrization must be 0");
}


void test_multiway_cut() {
    int nodes = 2;
    int classes = 2;
    std::vector<int> src = {0};
    std::vector<int> dest = {1};
    std::vector<float> edge_costs = {-2.0};
    std::vector<float> class_costs = {
        1.0, 1.0,
        1.0, 1.0,
    };
    thrust::device_vector<int> i;
    thrust::device_vector<int> j;
    thrust::device_vector<float> costs;
    std::tie(i, j, costs) = mwc_to_coo(nodes, classes, class_costs, src, dest, edge_costs);

    // i={0, 0, 0, 1, 1}
    // j={1, 2, 3, 2, 3}  2 and 3 indicate the new class nodes

    thrust::device_vector<int> t1 = std::vector<int>{0, 0};
    thrust::device_vector<int> t2 = std::vector<int>{1, 1};
    thrust::device_vector<int> t3 = std::vector<int>{2, 3};
    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
    multiwaycut_message_passing mwcp(A, nodes, classes, std::move(t1), std::move(t2), std::move(t3));

    // As for multicut the initial lower bound should -2 in this case as
    // the dual costs are 0 initialized
    const double initial_lb = mwcp.lower_bound();
    std::cout << "initial lb = " << initial_lb << "\n";
    if(std::abs(initial_lb - (-2.0)) > 1e-6)
        throw std::runtime_error("initial lb before reparametrization must be -2");

    mwcp.send_messages_to_triplets();

    mwcp.iteration();
    std::cout << mwcp.lower_bound() << std::endl;
    mwcp.iteration();
    std::cout << mwcp.lower_bound() << std::endl;
    mwcp.iteration();
    std::cout << mwcp.lower_bound() << std::endl;
    mwcp.iteration();
    std::cout << mwcp.lower_bound() << std::endl;
    mwcp.iteration();
    std::cout << mwcp.lower_bound() << std::endl;
    mwcp.iteration();
    std::cout << mwcp.lower_bound() << std::endl;

}


int main(int argc, char** argv)
{
    std::cout << "Testing multicut\n";
    test_multicut();
    std::cout << "Testing multiwaycut\n";
    test_multiway_cut();
}
