#include "edge_contractions_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_edge_contractions_tests<thrust::device_vector>();
    return 0;
}
