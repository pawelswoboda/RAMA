#include "find_pentagons_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_find_pentagons_tests<thrust::device_vector>();
    return 0;
}
