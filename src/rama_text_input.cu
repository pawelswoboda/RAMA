#include "rama_cuda.h"
#include "multicut_text_parser.h"
#include "multicut_solver_options.h"
#include "rama_utils.h"
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
    int dur;
    std::vector<std::vector<int>> timeline;
    std::tie(h_node_mapping, lb, dur, timeline) = rama_cuda(i, j, costs, opts);
    double obj = 0;
    if (!opts.only_compute_lb)
    {
        obj = get_obj(h_node_mapping, i, j, costs); 
        std::cout<<"\tcost w.r.t original objective: "<<obj<<"\n";
    }
    std::cout<<"\tfinal lower bound: "<<lb<<"\n";
    std::cout<<"\tGPU compute time: "<<dur<<"ms\n";

    if (opts.output_sol_file != "")
    {
        std::ofstream sol_file;
        sol_file.open(opts.output_sol_file);
        std::cout<<"Writing solution to file: "<<opts.output_sol_file<<"\n";
        const int num_clusters = *std::max_element(h_node_mapping.begin(), h_node_mapping.end()) + 1;
        std::vector<int> node_ids(h_node_mapping.size());
        std::iota(node_ids.begin(), node_ids.end(), 0);
        for (int c = 0; c != num_clusters; ++c)
        {
            std::vector<int> nodes_in_c;
            std::copy_if(node_ids.begin(), node_ids.end(), std::back_inserter(nodes_in_c), [&](int i) { return h_node_mapping[i] == c;});
            std::copy(nodes_in_c.begin(), nodes_in_c.end(), std::ostream_iterator<int>(sol_file, "\t"));
            sol_file<<"\n";
        }
        sol_file.close();
    }
}