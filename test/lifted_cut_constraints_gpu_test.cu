#include "lifted_cut_constraints_test.h"

int main()
{
    run_all_lifted_cut_constraints_tests<thrust::device_vector>();
    return 0;
}