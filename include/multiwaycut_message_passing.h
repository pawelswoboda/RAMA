#ifndef RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
#define RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
#include "multicut_message_passing.h"

class multiwaycut_message_passing: public multicut_message_passing {
public:
    multiwaycut_message_passing(
                const dCOO& A,
                const size_t _n_nodes,
                const size_t _n_classes,
                thrust::device_vector<float> _class_costs,
                thrust::device_vector<int>&& _t1,
                thrust::device_vector<int>&& _t2,
                thrust::device_vector<int>&& _t3,
                const bool _verbose = true
                );

    double lower_bound() override;

private:
    size_t n_nodes;
    size_t n_classes;
    thrust::device_vector<float> class_costs;  // Lagrange multipliers of the sum constraint
protected:
    double class_lower_bound();

    /**
     * Return true if the edge is a edge from a node to a class node.
     * Internally we compare if the dest > n_nodes, i.e. we assume that class nodes have an index
     * larger than any node index.
     *
     * @param source Source node of the edge
     * @param dest Destination of the edge
     * @return Boolean indicating if the edge connects a node and a class
     */
    bool is_class_edge(int source, int dest);
    /**
     * \overload is_class_edge
     * @param idx Index of the edge
     */
    bool is_class_edge(int idx);
};

#endif //RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
