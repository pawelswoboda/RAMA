#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "parallel-gaec-eigen.h"

namespace py=pybind11;

PYBIND11_MODULE(parallel_gaec_py, m) {
    m.doc() = "Python binding parallel gaec for multicut";

    m.def("parallel_gaec_eigen", [](const std::vector<std::tuple<size_t,size_t,double>>& edge_costs) {
            std::vector<weighted_edge> e;
            e.reserve(edge_costs.size());
            for(auto [i,j,c] : edge_costs)
            e.push_back({i,j,c});
            return parallel_gaec(e); 
            });
}

