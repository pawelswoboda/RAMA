#include "edge_contractions_woc.h"
#include "dCOO.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "test.h"

int main(int argc, char** argv)
{
    {
        const std::vector<int> i = {0, 1, 1, 0, 0};
        const std::vector<int> j = {1, 2, 3, 3, 2};
        const std::vector<float> costs = {2., 3., 1., -3, -2};

        thrust::device_vector<int> i_d = i;
        thrust::device_vector<int> j_d = j;
        thrust::device_vector<float> costs_d = costs;

        dCOO A(i_d.begin(), i_d.end(),
            j_d.begin(), j_d.end(),
            costs_d.begin(), costs_d.end(), true); 

        thrust::device_vector<int> node_mapping;
        int nr_edges_to_contract;
        edge_contractions_woc c_mapper(A);
        std::tie(node_mapping, nr_edges_to_contract) = c_mapper.find_contraction_mapping();

        test(node_mapping[0] != node_mapping[3]);
        test(node_mapping[0] != node_mapping[2]);
    }

    {
        const std::vector<int> i = {0, 1, 1, 0, 0, 0, 3, 3, 3};
        const std::vector<int> j = {1, 2, 3, 3, 2, 4, 4, 5, 2};
        const std::vector<float> costs = {2., 3., 1., -3, -2, 0.5, 0.75, 0.9, 0.8};

        thrust::device_vector<int> i_d = i;
        thrust::device_vector<int> j_d = j;
        thrust::device_vector<float> costs_d = costs;

        dCOO A(i_d.begin(), i_d.end(),
            j_d.begin(), j_d.end(),
            costs_d.begin(), costs_d.end(), true); 

        thrust::device_vector<int> node_mapping;
        int nr_edges_to_contract;
        edge_contractions_woc c_mapper(A);
        std::tie(node_mapping, nr_edges_to_contract) = c_mapper.find_contraction_mapping();

        test(node_mapping[0] != node_mapping[3]);
        test(node_mapping[0] != node_mapping[2]);
    }

    {
        const std::vector<int> i = {0, 1, 1, 0, 3, 3, 3, 1, 0, 0};
        const std::vector<int> j = {1, 2, 3, 4, 4, 5, 2, 5, 2, 3};
        const std::vector<float> costs = {2., 3., 1., 0.5, 0.75, 0.9, 0.8, -2, -3, -1};

        thrust::device_vector<int> i_d = i;
        thrust::device_vector<int> j_d = j;
        thrust::device_vector<float> costs_d = costs;

        dCOO A(i_d.begin(), i_d.end(),
            j_d.begin(), j_d.end(),
            costs_d.begin(), costs_d.end(), true); 

        thrust::device_vector<int> node_mapping;
        int nr_edges_to_contract;
        edge_contractions_woc c_mapper(A);
        std::tie(node_mapping, nr_edges_to_contract) = c_mapper.find_contraction_mapping();

        test(node_mapping[1] != node_mapping[5]);
        test(node_mapping[0] != node_mapping[2]);
        test(node_mapping[0] != node_mapping[3]);

        std::cout<<"node_mapping: \n";
        thrust::copy(node_mapping.begin(), node_mapping.end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout<<"\n";

        std::cout<<"nr_edges_to_contract: "<<nr_edges_to_contract<<"\n";
    }
}