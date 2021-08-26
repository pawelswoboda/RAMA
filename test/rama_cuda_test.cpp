#include "rama_cuda.h"
#include "multicut_solver_options.h"

int main(int argc, char** argv)
{
    multicut_solver_options opts;
    opts.matching_thresh_crossover_ratio = 1.1;
    std::vector<int> i = {0,0,1};
    std::vector<int> j = {1,2,2};
    std::vector<float> costs = {1.5,-3.0,1.0};

    rama_cuda(i, j, costs, opts); 
}