#include "conflicted_cycles_cuda.h"
#include <thrust/device_vector.h>
#include "rama_utils.h"
#include "dCOO.h"
#include "test.h"

int main(int argc, char** argv)
{
    const float tri_budget = 10.0;
    {
        const std::vector<int> i = {0, 1, 0, 2, 3, 0, 2, 0, 3, 4, 5, 4};
        const std::vector<int> j = {1, 2, 2, 3, 4, 3, 4, 4, 5, 5, 6, 6};
        const std::vector<float> costs = {2., 3., -1., 4., 1.5, 5., 2., -2., -3., 2., -1.5, 0.5};

        dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
        thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;

        std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 5, tri_budget);

        std::cout<<"Found triangles: \n";

        for (int t = 0; t < triangles_v1.size(); t++)
        {
            std::cout<<triangles_v1[t]<<" "<<triangles_v2[t]<<" "<<triangles_v3[t]<<"\n";
        }
    }

    { // 3-cycle:
        const std::vector<int> i = {0, 1, 2};
        const std::vector<int> j = {1, 2, 0};
        const std::vector<float> costs = {-1, 2, 1};

        dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
        thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 3, tri_budget);
            test(triangles_v1.size() == 1);
        }
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 4, tri_budget);
            test(triangles_v1.size() == 1);
        }
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 5, tri_budget);
            test(triangles_v1.size() == 1);
        }
    }
    
    { // 4-cycle:
        const std::vector<int> i = {0, 1, 2, 3};
        const std::vector<int> j = {1, 2, 3, 0};
        const std::vector<float> costs = {-1, 2, 1, 3};

        dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
        thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 3, tri_budget);
            test(triangles_v1.size() == 0);
        }
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 4, tri_budget);
            test(triangles_v1.size() == 2);
        }
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 5, tri_budget);
            test(triangles_v1.size() == 2);
        }
    }

        { // 4-cycle (non-conflicting):
            const std::vector<int> i = {0, 1, 2, 3};
            const std::vector<int> j = {1, 2, 3, 0};
            const std::vector<float> costs = {-1, -2, 1, 3};
    
            dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
            thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
            {
                std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 3, tri_budget);
                test(triangles_v1.size() == 0);
            }
            {
                std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 4, tri_budget);
                test(triangles_v1.size() == 0);
            }
            {
                std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 5, tri_budget);
                test(triangles_v1.size() == 0);
            }
        }

    { // 5-cycle:
        const std::vector<int> i = {0, 1, 2, 3, 4};
        const std::vector<int> j = {1, 2, 3, 4, 0};
        const std::vector<float> costs = {-1, 2, 1, 3, 2};

        dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
        thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;

        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 3, tri_budget);
            test(triangles_v1.size() == 0);
        }
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 4, tri_budget);
            test(triangles_v1.size() == 0);
        }
        {
            std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, 5, tri_budget);
            test(triangles_v1.size() == 3);
        }
    }
}