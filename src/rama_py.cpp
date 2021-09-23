#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rama_cuda.h"
#include "multicut_solver_options.h"
#include "multicut_text_parser.h"

namespace py=pybind11;

PYBIND11_MODULE(rama_py, m) {
    m.doc() = "Bindings for RAMA: Rapid algorithm for multicut. "
                "For running purely primal algorithm initialize multicut_solver_options with \"P\". "
                "For algorithm with best quality call with \"PD+\" where \"PD\" is default algorithm. "
                "For only computing the lower bound call with \"D\". ";
    py::class_<multicut_solver_options>(m, "multicut_solver_options")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def(py::init<const int&, const int&, const int&, const int&, const int&, 
                const float&, const float&, const float&, 
                const bool&, const int&, const bool&>());

    m.def("rama_cuda", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs, const multicut_solver_options& opts) {
            return rama_cuda(i, j, edge_costs, opts);
            });

    m.def("read_multicut_file", [](const std::string& filename) {
            return read_file(filename);
            });
}

