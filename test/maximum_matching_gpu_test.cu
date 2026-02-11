#include "maximum_matching_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_maximum_matching_tests<thrust::device_vector>();
    return 0;
}