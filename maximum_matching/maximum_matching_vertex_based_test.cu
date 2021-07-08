#include "maximum_matching_vertex_based.h"
#include "../dCSR.h"
#include "../utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusparse.h>

int main(int argc, char** argv)
{
    const std::vector<int> i = {0, 1, 0, 1, 2, 2};
    const std::vector<int> j = {1, 2, 2, 0, 1, 0};
    const std::vector<float> costs = {2., 3., -1., 2., 3., -1.};

    thrust::device_vector<int> i_d = i;
    thrust::device_vector<int> j_d = j;
    thrust::device_vector<float> costs_d = costs;

    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle), "cusparse init failed");
    dCSR A(handle, 
        i_d.begin(), i_d.end(),
        j_d.begin(), j_d.end(),
        costs_d.begin(), costs_d.end());

    thrust::device_vector<int> i_matched, j_matched; 
    std::tie(i_matched, j_matched) = filter_edges_by_matching_vertex_based(A);

    std::cout<<"i_matched: \n";
    thrust::copy(i_matched.begin(), i_matched.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout<<"\n";

    std::cout<<"j_matched: \n";
    thrust::copy(j_matched.begin(), j_matched.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout<<"\n";

    cusparseDestroy(handle);
}
