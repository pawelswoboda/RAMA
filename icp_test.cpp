#include "icp.h"
#include <thrust/host_vector.h>

int main(int argc, char** argv)
{
    const std::vector<int> i = {0, 1, 0, 2, 3, 0, 2, 0, 3, 4, 5, 4};
    const std::vector<int> j = {1, 2, 2, 3, 4, 3, 4, 4, 5, 5, 6, 6};
    const std::vector<float> costs = {2., 3., -1., 4., 1.5, 5., 2., -2., -3., 2., -1.5, 0.5};
    const std::vector<float> expected_cost = {2., 3., 0., 1., 0., 2., 0., 0., -1.5, 0., -1., 0.};

    thrust::device_vector<int> i_d = i;
    thrust::device_vector<int> j_d = j;
    thrust::device_vector<float> costs_d = costs;

    std::tie(i_d, j_d, costs_d) = parallel_cycle_packing_cuda(i_d, j_d, costs_d, 7, 10);

    thrust::host_vector<float> costs_result_h = costs_d;
    std::cout<<"Actual Result: \n";
    thrust::copy(costs_result_h.begin(), costs_result_h.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout<<"\n";

    std::cout<<"Expected Result: \n";
    std::copy(expected_cost.begin(), expected_cost.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout<<"\n";

}
