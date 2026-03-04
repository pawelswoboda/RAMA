#include "find_cycles_mst_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_find_cycles_mst_tests<thrust::device_vector>();
    return 0;
}
