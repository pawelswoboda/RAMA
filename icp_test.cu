#include "icp_small_cycles.h"
#include <thrust/device_vector.h>
#include <cusparse.h>
#include "utils.h"
#include "dCOO.h"

int main(int argc, char** argv)
{
    const std::vector<int> i = {0, 1, 0, 2, 3, 0, 2, 0, 3, 4, 5, 4};
    const std::vector<int> j = {1, 2, 2, 3, 4, 3, 4, 4, 5, 5, 6, 6};
    const std::vector<float> costs = {2., 3., -1., 4., 1.5, 5., 2., -2., -3., 2., -1.5, 0.5};

    double lb;
    dCOO A;
    std::tie(lb, A) = parallel_small_cycle_packing_cuda(i, j, costs, 5);
    assert(lb == -2.5);

    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");

    // First compute without any packing (re-arranges the edges):
    std::tie(lb, A) = parallel_small_cycle_packing_cuda(i, j, costs, 0);

    // Now, pack cycles:
    dCOO A_packed;
    std::tie(lb, A_packed) = parallel_small_cycle_packing_cuda(i, j, costs, 5);

    thrust::device_vector<float> costs_original_d = A.get_data();
    thrust::device_vector<float> costs_packed_d = A_packed.get_data();

    for (int e = 0; e < A.edges(); e++)
        if (costs_original_d[e] * costs_packed_d[e] < 0)
            std::cout<<"Test failed. Original cost: "<<costs_original_d[e]<<", packed cost: "<<costs_packed_d[e]<<". Signs should match! \n";
}
