#include "graph_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_graph_tests<thrust::device_vector>();
    return 0;
}
