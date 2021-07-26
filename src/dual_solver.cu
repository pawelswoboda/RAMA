#include "dual_solver.h"
#include "conflicted_cycles_cuda.h"
#include "multicut_message_passing.h"
#include "time_measure_util.h"
#include "parallel_gaec_utils.h"

std::tuple<dCOO, double> dual_update_cycle_length(const dCOO& A, const int cycle_length, const int num_dual_steps_per_cycle, const float tri_memory_factor)
{
    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, cycle_length, tri_memory_factor);

    multicut_message_passing mp(A, std::move(triangles_v1), std::move(triangles_v2), std::move(triangles_v3));
    for(int iter = 0; iter < num_dual_steps_per_cycle; ++iter)
    {
        const double lb = mp.lower_bound();
        std::cout << "dual updates cycle length: " << cycle_length << ", iteration: " << iter << ", lower bound: " << lb << "\n";
        mp.iteration();
    }
    const double final_lb = mp.lower_bound();
    std::cout << "dual updates cycle length: " << cycle_length << ", final lower bound: " << final_lb << "\n";
    thrust::device_vector<int> i_repam, j_repam;
    thrust::device_vector<float> costs_repam;
    std::tie(i_repam, j_repam, costs_repam) = mp.reparametrized_edge_costs();
    dCOO A_repam(std::move(j_repam), std::move(i_repam), std::move(costs_repam), true, true);
    return {A_repam, final_lb};
}

double dual_solver(dCOO& A, const int max_cycle_length, const int num_iter, const float tri_memory_factor)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    double final_lb;
    if (max_cycle_length < 3 || num_iter == 0)
        return get_lb(A.get_data());

    for (int c = 3; c <= max_cycle_length; ++c)
    {
        std::tie(A, final_lb) = dual_update_cycle_length(A, c, num_iter, tri_memory_factor);
    }
    return final_lb;
}