#include "lifted_message_passing_test.h"
#include <thrust/device_vector.h>

int main()
{
    run_all_lifted_message_passing_tests<thrust::device_vector>();
    return 0;
}