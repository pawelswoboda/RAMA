#include "multiwaycut_message_passing.h"


multiwaycut_message_passing::multiwaycut_message_passing(
        const dCOO &A,
        const size_t _n_nodes,
        const size_t _n_classes,
        thrust::device_vector<float> _class_costs,
        thrust::device_vector<int> &&_t1,
        thrust::device_vector<int> &&_t2,
        thrust::device_vector<int> &&_t3,
        const bool verbose
        )
        : multicut_message_passing(A, std::move(_t1), std::move(_t2), std::move(_t3), verbose),
        n_classes(_n_classes),
        n_nodes(_n_nodes),
        class_costs(std::move(_class_costs))
        {
}


bool multiwaycut_message_passing::is_class_edge(int source, int dest)
{
    if (dest >= n_nodes)
        return true;
    else
        return false;
}
bool multiwaycut_message_passing::is_class_edge(int idx)
{
    return is_class_edge(i[idx], j[idx]);
}

struct class_lower_bound_parallel {
    size_t k;  // number of classes
    size_t N;  // number of elements in class_costs
    float* class_costs;
    float* result;
    __host__ __device__ void operator() (int node) {
        size_t offset_start = node * k;
        size_t offset_end = offset_start + k - 1;
        assert(offset_end < N);

        // k should be quite small compared to N/k so summing using the for loop
        // should not be an issue
        float largest = class_costs[0];
        for (size_t i = offset_start; i < offset_end; ++i) {
            *result += class_costs[i];
            if (class_costs[i] > largest)
                largest = class_costs[i];
        }
        *result -= largest;
    }
};
double multiwaycut_message_passing::class_lower_bound()
{
    // The minimal class-cost-configuration for a node will always be given as
    // the sum of the elements in the cost vector with its largest element removed.
    // Hence instead of calculating the dot product for each of the K possible
    // configuration, we can sum up all elements and subtract the largest cost.

    // As typically N >> K it makes more sense to parallelize over N, we can
    // then simply use a for loop (with only K iterations) to calculate the dot
    // product
    thrust::counting_iterator<int> iter(0);  // Iterates over the node indices
    thrust::device_vector<float> res(1);
    class_lower_bound_parallel f({
        n_classes, n_nodes * n_classes,
        thrust::raw_pointer_cast(class_costs.data()),
        thrust::raw_pointer_cast(res.data())
    });
    thrust::for_each(iter, iter + n_nodes, f);

    return res[0];
}

double multiwaycut_message_passing::lower_bound()
{
    return multicut_message_passing::lower_bound() + class_lower_bound();
}
