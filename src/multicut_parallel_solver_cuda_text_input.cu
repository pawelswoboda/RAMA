#include "parallel_gaec_cuda.h"
#include "multicut_text_parser.h"
#include "multicut_solver_options.h"
#include "parallel_gaec_utils.h"
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
        
    std::tie(i, j, costs) = read_file(opts.input_file);

    std::vector<int> h_node_mapping;
    double lb;
    std::tie(h_node_mapping, lb) = parallel_gaec_cuda(i, j, costs, opts);
    
    double obj = get_obj(h_node_mapping, i, j, costs); 
    std::cout<<"\tcost w.r.t original objective: "<<obj<<"\n";
    std::cout<<"\tfinal lower bound: "<<lb<<"\n";
    if (opts.out_info_file != "")
    {
        std::ofstream outfile;
        outfile.open(opts.out_info_file, std::ios_base::app);
        outfile << std::fixed << opts.input_file << "," << obj << "," << lb << "\n";
        outfile.close();
    }
}