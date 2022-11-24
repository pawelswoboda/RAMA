#ifndef RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
#define RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
#include "multicut_message_passing.h"

class multiwaycut_message_passing: public multicut_message_passing {
public:
    multiwaycut_message_passing(
                const dCOO& A,
                const int _n_nodes,
                const int _n_classes,
                thrust::device_vector<int>&& _t1,
                thrust::device_vector<int>&& _t2,
                thrust::device_vector<int>&& _t3,
                const bool _verbose = true
                );

    double lower_bound() override;
    void send_messages_to_triplets() override;
    void send_messages_from_sum_to_edges();
    void send_messages_to_edges() override;
    void iteration() override;
    /**
     * Calculates the start and the size of all node "chunks".
     * The edges and costs are stored in one consecutively, ascending array, i.e. if we have the edges
     * 01 02 12 we have the arrays i=[0, 0, 1] and j=[1, 2, 2]
     * This method returns the number of edges for each source node and the first index of the source node in
     * the i array.
     *
     * @return Array with the node index, array with the the first index of the node in the i array and an array with
     * the number of edges for each node.
     */
    std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> get_node_partition();
private:
    int n_nodes;
    int n_classes;
    thrust::device_vector<float> class_costs;
protected:
    double class_lower_bound();

};

#endif //RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
