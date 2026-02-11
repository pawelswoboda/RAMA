#include "multicut_message_passing_test.h"

int main(int argc, char** argv)
{
    run_all_multicut_message_passing_tests<thrust::device_vector>();
    return 0;
}
