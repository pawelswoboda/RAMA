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
                const bool&, const int&, const bool&>())
        .def_readwrite("max_cycle_length_lb", &multicut_solver_options::max_cycle_length_lb)
        .def_readwrite("num_dual_itr_lb", &multicut_solver_options::num_dual_itr_lb)
        .def_readwrite("max_cycle_length_primal", &multicut_solver_options::max_cycle_length_primal)
        .def_readwrite("num_dual_itr_primal", &multicut_solver_options::num_dual_itr_primal)
        .def_readwrite("num_outer_itr_dual", &multicut_solver_options::num_outer_itr_dual)
        .def_readwrite("mean_multiplier_mm", &multicut_solver_options::mean_multiplier_mm)
        .def_readwrite("matching_thresh_crossover_ratio", &multicut_solver_options::matching_thresh_crossover_ratio)
        .def_readwrite("tri_memory_factor", &multicut_solver_options::tri_memory_factor)
        .def_readwrite("only_compute_lb", &multicut_solver_options::only_compute_lb)
        .def_readwrite("max_time_sec", &multicut_solver_options::max_time_sec)
        .def("__repr__", [](const multicut_solver_options &a) {
            return std::string("<multicut_solver_options>:") +
                        "max_cycle_length_lb: " + std::to_string(a.max_cycle_length_lb) +
                        ", num_dual_itr_lb: " + std::to_string(a.num_dual_itr_lb) +
                        ", max_cycle_length_primal: " + std::to_string(a.max_cycle_length_primal) +
                        ", num_outer_itr_dual: " + std::to_string(a.num_outer_itr_dual) +
                        ", mean_multiplier_mm: " + std::to_string(a.mean_multiplier_mm) +
                        ", matching_thresh_crossover_ratio: " + std::to_string(a.matching_thresh_crossover_ratio) +
                        ", tri_memory_factor: " + std::to_string(a.tri_memory_factor) +
                        ", only_compute_lb: " + std::to_string(a.only_compute_lb) +
                        ", max_time_sec: " + std::to_string(a.max_time_sec);
        });

    m.def("rama_cuda", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs, const multicut_solver_options& opts) {
            return rama_cuda(i, j, edge_costs, opts);
            });

    m.def("read_multicut_file", [](const std::string& filename) {
            return read_file(filename);
            });
}

