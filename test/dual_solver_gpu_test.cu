#include "dual_solver_test.h"

int main(int argc, char** argv)
{
    run_all_dual_solver_tests<thrust::device_vector>();
    return 0;
}
