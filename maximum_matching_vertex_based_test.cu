#include "maximum_matching_vertex_based.h"
#include "dCOO.h"
#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusparse.h>

int main(int argc, char** argv)
{
    const std::vector<int> i = {0, 1, 0, 3, 3};
    const std::vector<int> j = {1, 2, 2, 0, 1};
    const std::vector<float> costs = {2., 3., -1., 1, 2};

    thrust::device_vector<int> i_d = i;
    thrust::device_vector<int> j_d = j;
    thrust::device_vector<float> costs_d = costs;

    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");
    dCOO A(i_d.begin(), i_d.end(),
        j_d.begin(), j_d.end(),
        costs_d.begin(), costs_d.end());


    thrust::device_vector<int> node_mapping;
    int nr_matched_edges;
    std::tie(node_mapping, nr_matched_edges) = filter_edges_by_matching_vertex_based(handle, A.export_undirected());

    std::cout<<"node_mapping: \n";
    thrust::copy(node_mapping.begin(), node_mapping.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout<<"\n";

    std::cout<<"nr_matched_edges: "<<nr_matched_edges<<"\n";
    assert(nr_matched_edges == 4); // relative to undirected graph.

    cusparseDestroy(handle);
}
