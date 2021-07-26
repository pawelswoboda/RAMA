#include "parallel_gaec_cuda.h"
#include "multicut_text_parser.h"
#include "dual_solver.h"
#include "dCOO.h"
#include "multicut_solver_options.h"
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> costs;
    
    multicut_solver_options opts;
    int e = opts.from_cl(argc, argv);
    if (e != -1)
        return e;
        
    std::tie(i,j,costs) = read_file(opts.input_file);

    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end(), true);
    double final_lb = dual_solver(A, opts.max_cycle_length_lb, opts.num_dual_itr_lb);
    const std::vector<int> h_node_mapping = parallel_gaec_cuda(A, opts);

    std::cout<<"\tfinal lower bound: "<<final_lb<<"\n";
    print_obj_original(h_node_mapping, i, j, costs);
}