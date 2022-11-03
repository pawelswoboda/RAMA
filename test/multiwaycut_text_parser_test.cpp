#include "multiwaycut_text_parser.h"
#include "test.h"
#include <tuple>
#include <numeric>


void test_parser() {
    size_t n, k;
    std::vector<float> node_class_cost_matrix;
    std::vector<int> source, dest;
    std::vector<float> costs;
    std::tie(n, k, node_class_cost_matrix, source, dest, costs) = read_file("./test/multiwaycut_test.txt");
    test(n == 3, "Number of nodes is incorrect");
    test(k == 3, "Number of classes is incorrect");
    test(node_class_cost_matrix.size() == 9, "Matrix has incorrect size");
    float class_costs = std::accumulate(node_class_cost_matrix.begin(), node_class_cost_matrix.end(), 0.0f);
    test(class_costs == 3.0, "The class costs are incorrect");
}


void test_mwc_to_coo() {
    size_t n = 5;
    size_t k = 2;
    std::vector<int> src = {0,0,0,1,2};
    std::vector<int> dest = {1,2,3,3,3};
    std::vector<float> edge_costs = {1.0,1.0,-2.0,1.0,1.0};
    std::vector<float> class_costs = {
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0
    };

    std::vector<int> res_src, res_dest;
    std::vector<float> res_costs;
    std::tie(res_src, res_dest, res_costs) = mwc_to_coo(n, k, class_costs, src, dest, edge_costs);

    // Compare if all vectors have the correct size
    std::vector<size_t> result_sizes = {res_src.size(), res_dest.size(), res_costs.size()};
    for (size_t const size: result_sizes) {
        test(size == (src.size() + n * k), "Invalid number of edges returned");
    }
}


int main(int argc, char const *argv[])
{
    test_parser();
    test_mwc_to_coo();
    return 0;
}
