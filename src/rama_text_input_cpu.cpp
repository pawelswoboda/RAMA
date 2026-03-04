#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "rama_solver.h"
#include "multicut_text_parser.h"
#include "multicut_solver_options.h"
#include "rama_utils.h"
#include <chrono>
#include <algorithm>

int main(int argc, char** argv)
{
    multicut_solver_options opts;
    int e = opts.from_cl(argc, argv);
    if (e != -1)
        return e;

    auto inst = read_file(opts.input_file);

    thrust::host_vector<int> base_i(inst.i.begin(), inst.i.end());
    thrust::host_vector<int> base_j(inst.j.begin(), inst.j.end());
    thrust::host_vector<float> base_costs(inst.costs.begin(), inst.costs.end());
    thrust::host_vector<int> lifted_i(inst.lifted_i.begin(), inst.lifted_i.end());
    thrust::host_vector<int> lifted_j(inst.lifted_j.begin(), inst.lifted_j.end());
    thrust::host_vector<float> lifted_costs(inst.lifted_costs.begin(), inst.lifted_costs.end());

    sanitize_input_edges<thrust::host_vector>(base_i, base_j, base_costs, lifted_i, lifted_j, lifted_costs);

    HostGraph base_G(std::move(base_i), std::move(base_j), std::move(base_costs));
    HostGraph lifted_G;
    if (!lifted_i.empty())
        lifted_G = HostGraph(std::move(lifted_i), std::move(lifted_j), std::move(lifted_costs));

    auto start = std::chrono::steady_clock::now();
    auto [node_mapping, lb, timeline] = rama_solver<thrust::host_vector>(base_G, lifted_G, opts);
    auto end = std::chrono::steady_clock::now();
    int dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<int> h_node_mapping(node_mapping.begin(), node_mapping.end());

    if (!opts.only_compute_lb)
    {
        double obj = get_obj(h_node_mapping, inst.i, inst.j, inst.costs);
        obj += get_obj(h_node_mapping, inst.lifted_i, inst.lifted_j, inst.lifted_costs);
        std::cout << "\tcost w.r.t original objective: " << obj << "\n";
    }
    std::cout << "\tfinal lower bound: " << lb << "\n";
    std::cout << "\tCPU compute time: " << dur << "ms\n";
}