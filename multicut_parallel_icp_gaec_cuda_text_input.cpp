#include "parallel_gaec_cuda.h"
#include "multicut_text_parser.h"
#include "icp_small_cycles.h"
#include<stdexcept>

int main(int argc, char** argv)
{
    if(argc != 2)
        throw std::runtime_error("no filename given");
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> costs;
    std::tie(i,j,costs) = read_file(argv[1]);

    double lb;
    dCOO A;
    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    std::tie(lb, A, triangles_v1, triangles_v2, triangles_v3) = parallel_small_cycle_packing_cuda(i, j, costs, 1, 1);

    dCOO A_undir = A.export_undirected();
    const std::vector<int> h_node_mapping = parallel_gaec_cuda(A_undir); 

    print_obj_original(h_node_mapping, i, j, costs);
}

