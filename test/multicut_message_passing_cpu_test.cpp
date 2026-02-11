#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "multicut_message_passing_test.h"

int main(int argc, char** argv)
{
    run_all_multicut_message_passing_tests<thrust::host_vector>();
    return 0;
}
