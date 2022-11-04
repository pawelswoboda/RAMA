#include "multiwaycut_message_passing.h"
#include "multiwaycut_text_parser.h"


int main(int argc, char** argv)
{
    int n = 4;
    int k = 2;
    std::vector<int> src = {0,0,0,1,2};
    std::vector<int> dest = {1,2,3,3,3};
    std::vector<float> edge_costs = {1.0,1.0,-2.0,1.0,1.0};
    std::vector<float> class_costs = {
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0
    };

    thrust::device_vector<int> i;
    thrust::device_vector<int> j;
    thrust::device_vector<float> costs;
    std::tie(i, j, costs) = mwc_to_coo(n, k, class_costs, src, dest, edge_costs);

    thrust::device_vector<int> t1 = std::vector<int>{0,0};
    thrust::device_vector<int> t2 = std::vector<int>{1,2};
    thrust::device_vector<int> t3 = std::vector<int>{3,3};
    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
    multiwaycut_message_passing mwcp(A, n, k, class_costs, std::move(t1), std::move(t2), std::move(t3));

    // With this configuration the initial lower bound should still be -2
    // as the term sum evaluates to zero
    const double initial_lb = mwcp.lower_bound();
    std::cout << initial_lb << "\n";

    mwcp.iteration();
}
