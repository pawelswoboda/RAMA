#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "conflicted_cycles_test.h"
#include <thrust/host_vector.h>

int main()
{
    run_all_conflicted_cycles_tests<thrust::host_vector>();
    return 0;
}
