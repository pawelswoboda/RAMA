#include "dual_solver.h"
#include "conflicted_cycles_cuda.h"
#include "multicut_message_passing.h"
#include "time_measure_util.h"
#include "rama_utils.h"

std::tuple<dCOO, double, int> dual_update_cycle_length(const dCOO& A, const int cycle_length, const int num_dual_steps_per_cycle, const float tri_memory_factor, const float tol_ratio, const bool verbose)
{

    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    std::tie(triangles_v1, triangles_v2, triangles_v3) = conflicted_cycles_cuda(A, cycle_length, tri_memory_factor, tol_ratio, verbose);
    const int num_tri = triangles_v1.size();
    if (num_tri == 0)
        return {A, get_lb(A.get_data()), 0};

    multicut_message_passing mp(A, std::move(triangles_v1), std::move(triangles_v2), std::move(triangles_v3), verbose);
    double prev_lb = 0;
    for(int iter = 0; iter < num_dual_steps_per_cycle; ++iter)
    {
        const double lb = mp.lower_bound();
        if (verbose)
            std::cout << "dual updates cycle length: " << cycle_length << ", iteration: " << iter << ", lower bound: " << lb << "\n";
        if (iter > 0 && (lb - prev_lb) < 1e-3)
            break;
        mp.iteration(true); //!!!!!!!!!!! 
        prev_lb = lb;
    }
    const double final_lb = mp.lower_bound();
    if (verbose)
        std::cout << "dual updates cycle length: " << cycle_length << ", final lower bound: " << final_lb << "\n";
    thrust::device_vector<int> i_repam, j_repam;
    thrust::device_vector<float> costs_repam;
    std::tie(i_repam, j_repam, costs_repam) = mp.reparametrized_edge_costs();
    dCOO A_repam(std::move(j_repam), std::move(i_repam), std::move(costs_repam), true, true);
    return {A_repam, final_lb, num_tri};
}

double dual_solver(dCOO& A, const int max_cycle_length, const int num_iter, const float tri_memory_factor, const int num_outer_itr, const float tol_ratio, const bool verbose)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    double final_lb;
    if (max_cycle_length < 3 || num_iter == 0 || num_outer_itr == 0)
        return get_lb(A.get_data());

    double prev_lb = 0;
    int num_triangles;
    try {
        for (int outer_itr = 0; outer_itr < num_outer_itr; outer_itr++)
        {
            for (int c = 3; c <= max_cycle_length; ++c)
            {
                std::tie(A, final_lb, num_triangles) = dual_update_cycle_length(A, c, num_iter, tri_memory_factor, tol_ratio, verbose);
            }
            if (num_triangles == 0 || (outer_itr > 0 && (final_lb - prev_lb) < 1e-3))
                break;
            prev_lb = final_lb;
        }
    }
    catch(const std::bad_alloc& ex) { 
        std::cerr<<"Dual solver out of memory, returning trivial lower bound\n";
        final_lb = get_lb(A.get_data());
    }

    return final_lb;
}
