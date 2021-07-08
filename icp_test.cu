#include "icp_small_cycles.h"
#include <thrust/device_vector.h>
#include <cusparse.h>
#include "utils.h"

int main(int argc, char** argv)
{
    const std::vector<int> i = {0, 1, 0, 2, 3, 0, 2, 0, 3, 4, 5, 4};
    const std::vector<int> j = {1, 2, 2, 3, 4, 3, 4, 4, 5, 5, 6, 6};
    const std::vector<float> costs = {2., 3., -1., 4., 1.5, 5., 2., -2., -3., 2., -1.5, 0.5};

    assert(compute_lower_bound(i, j, costs, 5) == -2.5);

    thrust::device_vector<int> i_d = i;
    thrust::device_vector<int> j_d = j;
    thrust::device_vector<float> costs_d = costs;

    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");

    // First compute without any packing (re-arranges the edges):
    thrust::device_vector<int> i_new, j_new;
    thrust::device_vector<float> costs_original_d;
    std::tie(i_new, j_new, costs_original_d) = parallel_small_cycle_packing_costs(handle, i_d, j_d, costs_d, 0);

    // Now, pack cycles:
    thrust::device_vector<float> costs_packed_d;
    std::tie(i_d, j_d, costs_packed_d) = parallel_small_cycle_packing_costs(handle, i_d, j_d, costs_d, 5);

    for (int e = 0; e < costs.size(); e++)
    {
        if (costs_original_d[e] * costs_packed_d[e] < 0)
        {
            std::cout<<"Test failed at edge: "<<i_new[e]<<", "<<j_new[e]<<", original cost: "<<costs_original_d[e]<<", packed cost: "<<costs_packed_d[e]<<". Signs should match! \n";
        }
    }
}
