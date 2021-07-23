#include "parallel_gaec_cuda.h"
#include "multicut_text_parser.h"
#include "icp_small_cycles.h"
#include "conflicted_cycles.h"
#include "multicut_message_passing.h"
#include<stdexcept>

int main(int argc, char** argv)
{
    if(argc != 2)
        throw std::runtime_error("no filename given");
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> costs;
    std::tie(i,j,costs) = read_file(argv[1]);

    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    // double lb;
    // dCOO A;
    // std::tie(lb, A, triangles_v1, triangles_v2, triangles_v3) = parallel_small_cycle_packing_cuda(i, j, costs, 1, 0);
    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
    {
        conflicted_cycles conf_cyc(A, 5);
        std::tie(triangles_v1, triangles_v2, triangles_v3) = conf_cyc.enumerate_conflicted_cycles();
    }

    multicut_message_passing mp(A, triangles_v1, triangles_v2, triangles_v3);
    const double initial_lb = mp.lower_bound();
    std::cout << "initial lower bound: " << initial_lb << "\n";
    for(int iter=0; iter<50; ++iter)
    {
        mp.iteration();
        const double lb = mp.lower_bound();
        std::cout << "iteration " << iter << " lower bound: " << lb << "\n";
    }
    const double final_lb = mp.lower_bound();
    std::cout << "final lower bound: " << final_lb << "\n";
    thrust::device_vector<int> i_repam, j_repam, costs_repam;
    std::tie(i_repam, j_repam, costs_repam) = mp.reparametrized_edge_costs();
    const std::vector<int> h_node_mapping = parallel_gaec_cuda(std::move(i_repam), std::move(j_repam), std::move(costs_repam));

    print_obj_original(h_node_mapping, i, j, costs);
}

