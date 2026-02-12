#include "find_triangles_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_find_triangles_tests<thrust::device_vector>();
    return 0;
}
