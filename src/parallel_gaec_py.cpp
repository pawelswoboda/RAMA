#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "parallel_gaec_eigen.h"
#include "parallel_gaec_cuda.h"
#include "multicut_solver_options.h"
#include "multicut_text_parser.h"

namespace py=pybind11;

PYBIND11_MODULE(parallel_gaec_py, m) {
    m.doc() = "Python binding parallel gaec for multicut";
    py::class_<multicut_solver_options>(m, "multicut_solver_options")
        .def(py::init<>())
        .def(py::init<const int&, const int&, const int&, const int&, const int&, 
                const float&, const float&, const float&, 
                const bool&, const int&, const bool&>());

    m.def("parallel_gaec_eigen", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs) {
            return parallel_gaec_eigen(i,j,edge_costs); 
            });

    m.def("parallel_gaec_cuda", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs, const multicut_solver_options& opts) {
            return parallel_gaec_cuda(i, j, edge_costs, opts);
            });

    m.def("read_multicut_file", [](const std::string& filename) {
            return read_file(filename);
            });
}

