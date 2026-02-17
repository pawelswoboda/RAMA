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

    auto [i, j, costs] = read_file(opts.input_file);

    // Normalize edges to (min, max) orientation
    for (size_t e = 0; e < i.size(); ++e)
    {
        if (i[e] > j[e])
            std::swap(i[e], j[e]);
    }

    HostGraph G(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());

    auto start = std::chrono::steady_clock::now();
    auto [node_mapping, lb, timeline] = rama_solver<thrust::host_vector>(G, opts);
    auto end = std::chrono::steady_clock::now();
    int dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<int> h_node_mapping(node_mapping.begin(), node_mapping.end());

    if (!opts.only_compute_lb)
    {
        double obj = get_obj(h_node_mapping, i, j, costs);
        std::cout << "\tcost w.r.t original objective: " << obj << "\n";
    }
    std::cout << "\tfinal lower bound: " << lb << "\n";
    std::cout << "\tCPU compute time: " << dur << "ms\n";
}