#include "lifted_edge_contraction_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_lifted_edge_contraction_tests<thrust::device_vector>();
    return 0;
}
