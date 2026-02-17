#include "rama_solver.h"
#include "multicut_text_parser.h"
#include "multicut_solver_options.h"
#include "rama_utils.h"
#include <chrono>

int main(int argc, char** argv)
{
    multicut_solver_options opts;
    int e = opts.from_cl(argc, argv);
    if (e != -1)
        return e;

    auto [i, j, costs] = read_file(opts.input_file);

    initialize_gpu(opts.verbose);
    thrust::device_vector<int> i_gpu(i.begin(), i.end());
    thrust::device_vector<int> j_gpu(j.begin(), j.end());
    thrust::device_vector<float> costs_gpu(costs.begin(), costs.end());

    sort_edge_nodes(i_gpu, j_gpu);
    DeviceGraph G(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu));

    auto start = std::chrono::steady_clock::now();
    auto [node_mapping, lb, timeline] = rama_solver<thrust::device_vector>(G, opts);
    auto end = std::chrono::steady_clock::now();
    int dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<int> h_node_mapping(node_mapping.size());
    thrust::copy(node_mapping.begin(), node_mapping.end(), h_node_mapping.begin());

    if (!opts.only_compute_lb)
    {
        double obj = get_obj(h_node_mapping, i, j, costs);
        std::cout << "\tcost w.r.t original objective: " << obj << "\n";
    }
    std::cout << "\tfinal lower bound: " << lb << "\n";
    std::cout << "\tGPU compute time: " << dur << "ms\n";
}