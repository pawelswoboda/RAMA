#pragma once

// Interface between the Python module (rama_py.cpp) and the solver backends
// (rama_py_cpu.cpp, rama_py_gpu.cu). Only standard types may cross this
// boundary: nvcc wraps Thrust in an ABI namespace (thrust::_V_<version>_SM_<arch>),
// so thrust::host_vector in a .cu file is a different type than the one rama_cpu
// is instantiated for, and the two cannot be mixed in one translation unit.

#include <vector>
#include "multicut_solver_options.h"

#ifdef RAMA_WITH_CUDA
#include <pybind11/pybind11.h>
#endif

struct rama_py_result {
    std::vector<int> node_labels;
    double lower_bound;
    int duration_ms;
    std::vector<std::vector<int>> timeline;
};

rama_py_result rama_solve_cpu(const std::vector<int>& i, const std::vector<int>& j,
                              const std::vector<float>& costs, const multicut_solver_options& opts);

#ifdef RAMA_WITH_CUDA
rama_py_result rama_solve_gpu(const std::vector<int>& i, const std::vector<int>& j,
                              const std::vector<float>& costs, const multicut_solver_options& opts);

bool rama_gpu_available();

// Registers the GPU-only entry points (rama_cuda_gpu_pointers, rama_torch).
void register_gpu_bindings(pybind11::module_& m);
#endif
