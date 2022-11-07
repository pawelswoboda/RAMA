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
    void iteration() override;
private:
    int n_nodes;
    int n_classes;
    thrust::device_vector<float> class_costs;  // Lagrange multipliers of the sum constraint
protected:
    double class_lower_bound();

};

#endif //RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
