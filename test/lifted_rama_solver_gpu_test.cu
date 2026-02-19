#include "lifted_rama_solver_test.h"

int main()
{
    run_all_lifted_rama_solver_tests<thrust::device_vector>();
    return 0;
}
