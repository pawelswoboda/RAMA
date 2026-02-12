#include "find_quadrangles_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_find_quadrangles_tests<thrust::device_vector>();
    return 0;
}