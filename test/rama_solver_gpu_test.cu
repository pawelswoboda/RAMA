#include "rama_solver_test.h"

int main()
{
    run_all_rama_solver_tests<thrust::device_vector>();
    return 0;
}