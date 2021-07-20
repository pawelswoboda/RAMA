#include "icp_small_cycles.h"
#include <thrust/device_vector.h>
#include <cusparse.h>
#include "parallel_gaec_utils.h"
#include "dCOO.h"

int main(int argc, char** argv)
{
    const std::vector<int> i = {0, 1, 0, 2, 3, 0, 2, 0, 3, 4, 5, 4};
    const std::vector<int> j = {1, 2, 2, 3, 4, 3, 4, 4, 5, 5, 6, 6};
    const std::vector<float> costs = {2., 3., -1., 4., 1.5, 5., 2., -2., -3., 2., -1.5, 0.5};

    double lb;
    dCOO A;
    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    std::tie(lb, A, triangles_v1, triangles_v2, triangles_v3) = parallel_small_cycle_packing_cuda(i, j, costs, 5, 5);
    assert(lb == -2.5);

    // First compute without any packing (re-arranges the edges):
    std::tie(lb, A, triangles_v1, triangles_v2, triangles_v3) = parallel_small_cycle_packing_cuda(i, j, costs, 0, 0);

    // Now, pack cycles:
    dCOO A_packed;
    std::tie(lb, A_packed, triangles_v1, triangles_v2, triangles_v3) = parallel_small_cycle_packing_cuda(i, j, costs, 5, 5);

    thrust::device_vector<float> costs_original_d = A.get_data();
    thrust::device_vector<float> costs_packed_d = A_packed.get_data();

    for (int e = 0; e < A.nnz(); e++)
        if (costs_original_d[e] * costs_packed_d[e] < 0)
            std::cout<<"Test failed. Original cost: "<<costs_original_d[e]<<", packed cost: "<<costs_packed_d[e]<<". Signs should match! \n";

    std::cout<<"Found triangles: \n";

    for (int t = 0; t < triangles_v1.size(); t++)
    {
        std::cout<<triangles_v1[t]<<" "<<triangles_v2[t]<<" "<<triangles_v3[t]<<"\n";
    }
}
