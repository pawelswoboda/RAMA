#pragma once

// Python bindings shared by the GPU (rama_py) and CPU (rama_py_cpu) modules.

#include <string>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "multicut_solver_options.h"
#include "multicut_text_parser.h"

namespace py = pybind11;

// Both modules bind multicut_solver_options. module_local() keeps the two
// registrations apart so that rama_py and rama_py_cpu can be imported into the
// same Python process. pybind11 still accepts the other module's options
// object, since both wrap the same C++ type.
inline void bind_multicut_solver_options(py::module_& m)
{
    py::class_<multicut_solver_options>(m, "multicut_solver_options", py::module_local())
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
        .def_readwrite("verbose", &multicut_solver_options::verbose)
        .def_readwrite("dump_timeline", &multicut_solver_options::dump_timeline)
        .def_readwrite("sanitize_graph", &multicut_solver_options::sanitize_graph)
        .def_readwrite("use_lifted_cut_constraints", &multicut_solver_options::use_lifted_cut_constraints)
        .def("__repr__", [](const multicut_solver_options &a) {
            return a.get_string();
        });
}

inline void bind_read_multicut_file(py::module_& m)
{
    m.def("read_multicut_file", [](const std::string& filename) {
            auto inst = read_file(filename);
            return std::make_tuple(inst.i, inst.j, inst.costs,
                                   inst.lifted_i, inst.lifted_j, inst.lifted_costs);
            });
}
