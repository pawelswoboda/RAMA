#include "parallel_gaec_eigen.h"

int main(int argc, char** argv)
{
    std::vector<int> i = {0,0,1};
    std::vector<int> j = {1,2,2};
    std::vector<float> costs = {1.5,-3.0,1.0};

    parallel_gaec_eigen(i,j,costs); 
}
