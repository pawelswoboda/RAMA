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

    auto inst = read_file(opts.input_file);

    initialize_gpu(opts.verbose);
    thrust::device_vector<int> i_gpu(inst.i.begin(), inst.i.end());
    thrust::device_vector<int> j_gpu(inst.j.begin(), inst.j.end());
    thrust::device_vector<float> costs_gpu(inst.costs.begin(), inst.costs.end());
    thrust::device_vector<int> li_gpu(inst.lifted_i.begin(), inst.lifted_i.end());
    thrust::device_vector<int> lj_gpu(inst.lifted_j.begin(), inst.lifted_j.end());
    thrust::device_vector<float> lc_gpu(inst.lifted_costs.begin(), inst.lifted_costs.end());

    sanitize_input_edges<thrust::device_vector>(i_gpu, j_gpu, costs_gpu, li_gpu, lj_gpu, lc_gpu);

    DeviceGraph base_G(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu));
    DeviceGraph lifted_G;
    if (!li_gpu.empty())
        lifted_G = DeviceGraph(std::move(li_gpu), std::move(lj_gpu), std::move(lc_gpu));

    auto start = std::chrono::steady_clock::now();
    auto [node_mapping, lb, timeline] = rama_solver<thrust::device_vector>(base_G, lifted_G, opts);
    auto end = std::chrono::steady_clock::now();
    int dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<int> h_node_mapping(node_mapping.size());
    thrust::copy(node_mapping.begin(), node_mapping.end(), h_node_mapping.begin());

    if (!opts.only_compute_lb)
    {
        double obj = get_obj(h_node_mapping, inst.i, inst.j, inst.costs);
        obj += get_obj(h_node_mapping, inst.lifted_i, inst.lifted_j, inst.lifted_costs);
        std::cout << "\tcost w.r.t original objective: " << obj << "\n";
    }
    std::cout << "\tfinal lower bound: " << lb << "\n";
    std::cout << "\tGPU compute time: " << dur << "ms\n";
}