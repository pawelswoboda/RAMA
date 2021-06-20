#include "parallel_gaec_cuda.h"

int main(int argc, char** argv)
{
    std::vector<std::tuple<int,int,float>> edges = {
        {0,1,1.5},
        {0,2,-3.0},
        {1,2,1.0} 
    };

    parallel_gaec_cuda(edges); 
}

