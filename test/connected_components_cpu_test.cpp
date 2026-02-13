#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "connected_components_test.h"
#include <thrust/host_vector.h>

int main()
{
    run_all_connected_components_tests<thrust::host_vector>();
    return 0;
}
