#include "parallel-gaec-eigen.h"
#include "multicut_text_parser.h"

int main(int argc, char** argv)
{
    if(argc != 2)
        throw std::runtime_error("no filename given");
    const auto e = read_file(argv[1]);
    std::vector<weighted_edge> e2;
    e2.reserve(e.size());
    for(const auto [i,j,c] : e)
        e2.push_back({i,j,c});
    parallel_gaec(e2); 
}

