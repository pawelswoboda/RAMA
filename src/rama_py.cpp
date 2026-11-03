#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "multicut_solver_options.h"
#include "multicut_text_parser.h"
#include "rama_py_backend.h"

namespace py=pybind11;

#ifdef RAMA_WITH_CUDA
static constexpr bool with_cuda = true;
#else
static constexpr bool with_cuda = false;
#endif

using rama_py_tuple = std::tuple<std::vector<int>, double, int, std::vector<std::vector<int>>>;

static rama_py_tuple to_tuple(rama_py_result&& r)
{
    return {std::move(r.node_labels), r.lower_bound, r.duration_ms, std::move(r.timeline)};
}

static void check_input(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs)
{
    if (i.size() != j.size() || i.size() != costs.size())
        throw std::invalid_argument("i, j and edge_costs must have the same length");
    if (i.empty())
        throw std::invalid_argument("graph must contain at least one edge");
}

static rama_py_tuple rama_cpu(const std::vector<int>& i, const std::vector<int>& j,
                              const std::vector<float>& costs, const multicut_solver_options& opts)
{
    check_input(i, j, costs);
    py::gil_scoped_release release;
    return to_tuple(rama_solve_cpu(i, j, costs, opts));
}

static rama_py_tuple rama_cuda(const std::vector<int>& i, const std::vector<int>& j,
                               const std::vector<float>& costs, const multicut_solver_options& opts)
{
#ifdef RAMA_WITH_CUDA
    check_input(i, j, costs);
    py::gil_scoped_release release;
    return to_tuple(rama_solve_gpu(i, j, costs, opts));
#else
    throw std::runtime_error("RAMA was built without CUDA support, use device=\"cpu\" or rama_cpu()");
#endif
}

static rama_py_tuple rama(const std::vector<int>& i, const std::vector<int>& j,
                          const std::vector<float>& costs, const multicut_solver_options& opts,
                          const std::string& device)
{
    if (device == "cpu")
        return rama_cpu(i, j, costs, opts);
    if (device == "cuda" || device == "gpu")
        return rama_cuda(i, j, costs, opts);
    if (device == "auto")
    {
#ifdef RAMA_WITH_CUDA
        if (rama_gpu_available())
            return rama_cuda(i, j, costs, opts);
#endif
        return rama_cpu(i, j, costs, opts);
    }
    throw std::invalid_argument("device must be one of \"auto\", \"cpu\", \"cuda\", got \"" + device + "\"");
}

PYBIND11_MODULE(rama_py, m) {
    m.doc() = "Bindings for RAMA: Rapid algorithm for multicut. "
                "Use rama() to solve on the GPU if available and on the CPU otherwise, "
                "or rama_cpu() / rama_cuda() to pick the backend explicitly. "
                "For running purely primal algorithm initialize multicut_solver_options with \"P\". "
                "For algorithm with best quality call with \"PD+\" where \"PD\" is default algorithm. "
                "For only computing the lower bound call with \"D\". ";
    m.attr("with_cuda") = with_cuda;

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
        .def_readwrite("verbose", &multicut_solver_options::verbose)
        .def_readwrite("dump_timeline", &multicut_solver_options::dump_timeline)
        .def_readwrite("sanitize_graph", &multicut_solver_options::sanitize_graph)
        .def_readwrite("use_lifted_cut_constraints", &multicut_solver_options::use_lifted_cut_constraints)
        .def_readwrite("cut_factor_budget_ratio", &multicut_solver_options::cut_factor_budget_ratio)
        .def_readwrite("max_raw_triangles", &multicut_solver_options::max_raw_triangles)
        .def("__repr__", [](const multicut_solver_options &a) {
            return a.get_string();
        });

    m.def("rama", &rama,
          "Solve a multicut instance. device is \"auto\" (GPU if available, else CPU), \"cpu\" or \"cuda\". "
          "Returns (node_labels, lower_bound, duration_ms, timeline).",
          py::arg("i"), py::arg("j"), py::arg("edge_costs"),
          py::arg("opts") = multicut_solver_options(), py::arg("device") = "auto");

    m.def("rama_cpu", &rama_cpu,
          "Solve a multicut instance on the CPU. Returns (node_labels, lower_bound, duration_ms, timeline).",
          py::arg("i"), py::arg("j"), py::arg("edge_costs"), py::arg("opts") = multicut_solver_options());

    m.def("rama_cuda", &rama_cuda,
          "Solve a multicut instance on the GPU. Returns (node_labels, lower_bound, duration_ms, timeline).",
          py::arg("i"), py::arg("j"), py::arg("edge_costs"), py::arg("opts") = multicut_solver_options());

    m.def("read_multicut_file", [](const std::string& filename) {
            auto inst = read_file(filename);
            return std::make_tuple(inst.i, inst.j, inst.costs,
                                   inst.lifted_i, inst.lifted_j, inst.lifted_costs);
            });

#ifdef RAMA_WITH_CUDA
    register_gpu_bindings(m);
#endif
}
