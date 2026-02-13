#include "mst_boruvka_test.h"

int main()
{
    run_all_mst_boruvka_tests<thrust::device_vector>();
    return 0;
}
