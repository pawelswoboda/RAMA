#include "graph_test.h"
#include <thrust/host_vector.h>

int main()
{
    run_all_graph_tests<thrust::host_vector>();
    return 0;
}
