#include "multiwaycut_text_parser.h"
#include "test.h"
#include <tuple>
#include <numeric>

int main(int argc, char const *argv[])
{
    size_t n, k;
    std::vector<float> node_class_cost_matrix;
    std::vector<int> source, dest;
    std::vector<float> costs;
    std::tie(n, k, node_class_cost_matrix, source, dest, costs) = read_file("./test/multiwaycut_test.txt");
    test(n == 3, "Number of nodes is incorrect");
    test(k == 3, "Number of classes is incorrect");
    test(node_class_cost_matrix.size() == 9, "Matrix has incorrect size");
    float class_costs = std::accumulate(node_class_cost_matrix.begin(), node_class_cost_matrix.end(), 0.0);
    test(class_costs == 3.0, "The class costs are incorrect");
    return 0;
}
