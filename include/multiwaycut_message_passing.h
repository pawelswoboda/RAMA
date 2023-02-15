#ifndef RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
#define RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
#include "multicut_message_passing.h"
#include "functional"

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


    void add_class_dependent_triangle_subproblems(int e1, int e2, int e3, int c1, int c2);
    void add_class_dependent_triangle_subproblems(int e1, int e2, int e3);
    void add_class_dependent_triangle_subproblems(std::vector<int> classes);
    void add_class_dependent_triangle_subproblems(int c1, int c2);
    void add_class_dependent_triangle_subproblems();

private:
    int n_nodes;
    int n_classes;
    thrust::device_vector<float> class_costs;
    thrust::device_vector<float> cdtf_costs;  // class-dependent triangle factor costs
    // Stores the edge index for all class-dependent triangle factors
    // Even tough the order should not be important we add the edges as follows
    // 1c1 2c1 3c1 1c2 2c2 3c3 e12 e13 e23
    // where 1,2,3 indicate the nodes sorted in ascending order
    // This is based on the order of the cost vector in equation (12)

    // Based on this it is clear that chunks of 9 represent a single cdtf subproblem
    thrust::device_vector<int> cdtf_correspondence;

    /**
     * Returns a vector with as many items as edges, describing if the edge is a class edge or not
     * @param sources Start of an edge
     * @param dests End of an edge
     * @param is_class_edge_f Function taking the source and destination of an edge and returning whether it is a class edge
     */
    static thrust::device_vector<bool> create_class_edge_mask(
        thrust::device_vector<int> sources,
        thrust::device_vector<int> dests,
        std::function<bool(int, int)> is_class_edge_f
    );

    /**
     * Returns the set of nodes in this triangle
     */
    thrust::device_vector<int> get_nodes_in_triangle(int e1, int e2, int e3);

    /**
     * Returns the index of the edge node - class k in the i / j vector
     * @param node Node u specified by its index
     * @param k Class id in the range [0, K)
     * @return Index of u - k edge in i / j vector
     */
    int get_class_edge_index(int node, int k);

protected:
    double class_lower_bound();
    double cdtf_lower_bound();
    double triangle_lower_bound_2_classes();
    thrust::device_vector<bool> is_class_edge;
    thrust::device_vector<int> cdtf_counter;  // In how many class-dependent triangle factors a edge is present
};

#endif //RAMA_MULTIWAYCUT_MESSAGE_PASSING_H
