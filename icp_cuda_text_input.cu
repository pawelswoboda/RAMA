#include "icp_small_cycles.h"
#include "multicut_text_parser.h"
#include<stdexcept>

int main(int argc, char** argv)
{
    if(argc != 2)
        throw std::runtime_error("no filename given");
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> costs;
    std::tie(i,j,costs) = read_file(argv[1]);

    double lb = parallel_small_cycle_packing_cuda_lower_bound(i, j, costs, 5);

    std::cout<<"Final lower bound: "<<lb<<"\n";
}
