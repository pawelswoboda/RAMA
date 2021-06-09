#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "parallel-gaec-eigen.h"

namespace py=pybind11;

PYBIND11_MODULE(parallel_gaec_py, m) {
    m.doc() = "Python binding parallel gaec for multicut";

    m.def("parallel_gaec_eigen", [](const Eigen::Array<size_t, Eigen::Dynamic, 2>& edge_indices, const Eigen::Array<double, Eigen::Dynamic, 1>& edge_costs) {
        std::vector<weighted_edge> e;
        e.reserve(edge_indices.rows());
        for(size_t i=0; i<edge_indices.rows(); ++i)
            e.push_back({edge_indices(i, 0), edge_indices(i, 1), edge_costs(i, 0)});

        return parallel_gaec(e); 
    });
}

