#include "parallel-gaec-eigen.h"

int main(int argc, char** argv)
{
    std::vector<weighted_edge> edges = {
        {0,1,1.0},
        {0,2,-3.0},
        {1,2,1.0} 
    };

    parallel_gaec(edges); 
}
