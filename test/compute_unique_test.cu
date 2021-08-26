#include "rama_utils.h"
#include <vector>
#include <thrust/device_vector.h>
#include "test.h"

int main(int argc, char** argv)
{
    std::vector<int> i_init = {0, 1, 2, 2, 3, 4, 4};
    const thrust::device_vector<int> i(i_init.begin(), i_init.end());
    thrust::device_vector<int> unique_values, counts;
    std::tie(unique_values, counts) = get_unique_with_counts(i);
    test(unique_values.size() == 5);
    test(counts.size() == 5);
    test(unique_values[0] == 0);
    test(unique_values[1] == 1);
    test(unique_values[2] == 2);
    test(unique_values[3] == 3);
    test(unique_values[4] == 4);

    const thrust::device_vector<int> inverted_unique = invert_unique(unique_values, counts);
    test(inverted_unique.size() == i.size());
    for (int idx = 0; idx < inverted_unique.size(); idx++)
        test(inverted_unique[idx] == i[idx]);
}