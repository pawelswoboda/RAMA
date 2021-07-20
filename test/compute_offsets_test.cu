#include "parallel_gaec_utils.h"
#include <vector>
#include <thrust/device_vector.h>
#include "test.h"

int main(int argc, char** argv)
{
    { 
        std::vector<int> i_init = {0, 1, 2, 2, 3, 4, 4};
        const thrust::device_vector<int> i(i_init.begin(), i_init.end());
        const thrust::device_vector<int> offsets = compute_offsets(i);
        test(offsets[0] == 0);
        test(offsets[1] == 1);
        test(offsets[2] == 2);
        test(offsets[3] == 4);
        test(offsets[4] == 5);
        test(offsets[5] == 7); 
    }
}
