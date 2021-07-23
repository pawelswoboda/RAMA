#include "multicut_message_passing.h"
#include <iostream>

int main(int argc, char** argv)
{
    thrust::device_vector<int> i = std::vector<int>{0,0,0,1,2};
    thrust::device_vector<int> j = std::vector<int>{1,2,3,3,3};
    thrust::device_vector<float> edge_costs = std::vector<float>{1.0,1.0,-2.0,1.0,1.0};
    thrust::device_vector<int> t1 = std::vector<int>{0,0};
    thrust::device_vector<int> t2 = std::vector<int>{1,2};
    thrust::device_vector<int> t3 = std::vector<int>{3,3};
    dCOO A(i.begin(), i.end(), j.begin(), j.end(), edge_costs.begin(), edge_costs.end(), true);
    multicut_message_passing mcp(A, t1, t2, t3);

    const double initial_lb = mcp.lower_bound();
    std::cout << "initial lb = " << initial_lb << "\n";
    if(std::abs(initial_lb - (-2.0)) > 1e-6)
        throw std::runtime_error("initial lb before reparametrization must be -2");

    mcp.send_messages_to_triplets();

    mcp.iteration();
    mcp.iteration();
    mcp.iteration();

    const double final_lb = mcp.lower_bound();
    std::cout << "final lb = " << final_lb << "\n";
    if(std::abs(final_lb) > 1e-6)
        throw std::runtime_error("final lb after reparametrization must be 0");
}
