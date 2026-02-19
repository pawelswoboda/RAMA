#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "lifted_cut_constraints_test.h"
#include <thrust/host_vector.h>

int main()
{
    run_all_lifted_cut_constraints_tests<thrust::host_vector>();
    return 0;
} 