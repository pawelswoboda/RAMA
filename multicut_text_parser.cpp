#include <string>
#include <fstream>
#include "parallel-gaec-eigen.h"

std::vector<weighted_edge> read_file(const std::string& filename)
{
    std::ifstream f;
    f.open(filename);
    if(!f.is_open())
        throw std::runtime_error("Could not open multicut input file " + filename);

    std::string init_line;
    std::getline(f, init_line);
    if(init_line != "MULTICUT")
        throw std::runtime_error("first line must be 'MULTICUT'");
    int i, j;
    float c;
    std::vector<weighted_edge> edges;
    while(f >> i >> j >> c)
        edges.push_back({i,j,c});

    return edges;
}
