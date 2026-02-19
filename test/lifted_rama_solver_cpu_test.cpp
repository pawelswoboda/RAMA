#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "lifted_rama_solver_test.h"

int main()
{
    run_all_lifted_rama_solver_tests<thrust::host_vector>();
    return 0;
}
