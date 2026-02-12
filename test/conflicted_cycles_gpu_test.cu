#include "conflicted_cycles_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_conflicted_cycles_tests<thrust::device_vector>();
    return 0;
}
