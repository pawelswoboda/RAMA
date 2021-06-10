#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "parallel-gaec-eigen.h"
#include "multicut_text_parser.h"

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

    m.def("read_multicut_file", [](const std::string& filename) {
            const auto e = read_file(filename);
            std::vector<std::tuple<int,int,float>> e_return;
            e_return.reserve(e.size());
            for(const auto [i,j,c] : e)
            e_return.push_back({i,j,c});
            return e_return; 
            });
}

