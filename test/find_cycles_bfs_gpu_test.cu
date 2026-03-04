#include "find_cycles_bfs_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_find_cycles_bfs_tests<thrust::device_vector>();
    return 0;
}
