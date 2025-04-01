#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rama_cuda.h"
#include "multicut_solver_options.h"
#include "multicut_text_parser.h"

#include "multicut_message_passing.h"
#include "dCOO.h"
#include "conflicted_cycles_cuda.h"


#ifdef WITH_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#include <thrust/device_vector.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

namespace py=pybind11;
using namespace pybind11::literals;

#ifdef WITH_TORCH
std::vector<torch::Tensor> rama_torch(
    const torch::Tensor& _i,
    const torch::Tensor& _j,
    const torch::Tensor& _costs,
	const multicut_solver_options& opts) 
{
	CHECK_INPUT(_i);
	CHECK_INPUT(_j);
	CHECK_INPUT(_costs);
	if (_i.size(0) != _j.size(0) || _i.size(0) != _costs.size(0))
		throw std::runtime_error("Input shapes must match");
    if (_i.scalar_type() != _j.scalar_type())
		throw std::runtime_error("Node indices i, j should be of same type");

    TORCH_CHECK(_i.dim() == 1, "i should be one-dimensional");
    TORCH_CHECK(_j.dim() == 1, "j should be one-dimensional");
    TORCH_CHECK(_costs.dim() == 1, "costs should be one-dimensional");

	// at::checkScalarType(_i, ScalarType::int);
	thrust::device_vector<int> i(_i.data_ptr<int32_t>(), _i.data_ptr<int32_t>() + _i.size(0));
	thrust::device_vector<int> j(_j.data_ptr<int32_t>(), _j.data_ptr<int32_t>() + _j.size(0));
	thrust::device_vector<float> costs(_costs.data_ptr<float>(), _costs.data_ptr<float>() + _costs.size(0));
	thrust::device_vector<int> node_mapping;
    std::vector<std::vector<int>> timeline;
	double lb;
	const int device = _costs.device().index();
	if (device < 0)
		throw std::runtime_error("Invalid device ID");
    std::tie(node_mapping, lb, timeline) = rama_cuda(std::move(i), std::move(j), std::move(costs), opts, device);
    
	torch::Tensor node_mapping_torch = at::empty({long(node_mapping.size())}, _i.options());
    thrust::copy(node_mapping.begin(), node_mapping.end(), node_mapping_torch.data_ptr<int32_t>());

	torch::Tensor lb_torch = at::empty({1}, _i.options());
	lb_torch.toType(torch::kFloat64);
	lb_torch.fill_(lb);
	return {node_mapping_torch, lb_torch};
}
#endif

std::vector<std::vector<int>> rama_cuda_gpu_pointers(const int* const i, const int* const j, const float* const edge_costs, 
                        int* const node_labels, const int num_nodes, const int num_edges, const int gpuDeviceID, const multicut_solver_options& opts)
{
    thrust::device_vector<int> i_thrust(i, i + num_edges);
    thrust::device_vector<int> j_thrust(j, j + num_edges);
    thrust::device_vector<float> costs_thrust(edge_costs, edge_costs + num_edges);
    thrust::device_vector<int> node_mapping;
    std::vector<std::vector<int>> timeline;
    double lb;

    std::tie(node_mapping, lb, timeline) = rama_cuda(std::move(i_thrust), std::move(j_thrust), std::move(costs_thrust), opts, gpuDeviceID);
    thrust::copy(node_mapping.begin(), node_mapping.end(), node_labels);
    return timeline;
}

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
        .def_readwrite("verbose", &multicut_solver_options::verbose)
        .def_readwrite("dump_timeline", &multicut_solver_options::dump_timeline)
        .def_readwrite("sanitize_graph", &multicut_solver_options::sanitize_graph)
	.def_readwrite("dump_timeline", &multicut_solver_options::dump_timeline)
        .def("__repr__", [](const multicut_solver_options &a) {
            return a.get_string();
        });

    m.def("rama_cuda", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs, const multicut_solver_options& opts) {    
            return rama_cuda(i, j, edge_costs, opts);
            });

    m.def("rama_cuda_gpu_pointers", [](const long i_ptr, const long j_ptr, const long edge_costs_ptr, const long node_labels_out_ptr, 
                                    const int num_nodes, const int num_edges, const int gpuDeviceID, const multicut_solver_options& opts) {
            const int* const i = reinterpret_cast<const int* const>(i_ptr);
            const int* const j = reinterpret_cast<const int* const>(j_ptr);
            const float* const edge_costs = reinterpret_cast<const float* const>(edge_costs_ptr);
            int* const node_labels = reinterpret_cast<int* const>(node_labels_out_ptr);
            return rama_cuda_gpu_pointers(i, j, edge_costs, node_labels, num_nodes, num_edges, gpuDeviceID, opts);
            });

    m.def("read_multicut_file", [](const std::string& filename) {
            return read_file(filename);
            });

	#ifdef WITH_TORCH
		m.def("rama_torch", &rama_torch, "RAMA CUDA solver with torch interface.");
	#endif


    py::class_<dCOO>(m, "dCOO")
        .def(py::init<>())
        .def("rows", &dCOO::rows)
        .def("cols", &dCOO::cols)
        .def("nnz", &dCOO::nnz)
        .def("sum", &dCOO::sum)
        .def("min", &dCOO::min)
        .def("max", &dCOO::max)
        .def("get_row_ids", &dCOO::get_row_ids)
        .def("get_col_ids", &dCOO::get_col_ids)
        .def("get_data", &dCOO::get_data);

    py::class_<multicut_message_passing>(m, "MulticutMessagePassing")
        .def(py::init<const dCOO&, std::vector<int>, std::vector<int>, std::vector<int>, bool>())
        .def("get_triangles", &multicut_message_passing::get_triangles)
        .def("get_lagrange_multipliers", &multicut_message_passing::get_lagrange_multipliers)
        .def("get_edges", &multicut_message_passing::get_edges)
        .def("get_i", &multicut_message_passing::get_i)
        .def("get_j", &multicut_message_passing::get_j)
        .def("get_edge_costs", &multicut_message_passing::get_edge_costs)
        .def("get_t12_costs", &multicut_message_passing::get_t12_costs)
        .def("get_t13_costs", &multicut_message_passing::get_t13_costs)
        .def("get_t23_costs", &multicut_message_passing::get_t23_costs)
        .def("get_tri_corr_12", &multicut_message_passing::get_tri_corr_12)
        .def("get_tri_corr_13", &multicut_message_passing::get_tri_corr_13)
        .def("get_tri_corr_23", &multicut_message_passing::get_tri_corr_23)
        .def("get_edge_counter", &multicut_message_passing::get_edge_counter);

        m.def("get_message_passing_data", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs, int cycle_length) {
            dCOO A(i, j, edge_costs, true);
            auto [t1, t2, t3] = conflicted_cycles_cuda(A, cycle_length, 1.0, 1e-4, false);
            multicut_message_passing mp(A, std::move(t1), std::move(t2), std::move(t3), false);

            return py::dict(
                "i"_a = mp.get_i(),
                "j"_a = mp.get_j(),
                "edge_costs"_a = mp.get_edge_costs(),
                "t12_costs"_a = mp.get_t12_costs(),
                "t13_costs"_a = mp.get_t13_costs(),
                "t23_costs"_a = mp.get_t23_costs(),
                "tri_corr_12"_a = mp.get_tri_corr_12(),
                "tri_corr_13"_a = mp.get_tri_corr_13(),
                "tri_corr_23"_a = mp.get_tri_corr_23(),
                "edge_counter"_a = mp.get_edge_counter()
            );
        });

}

