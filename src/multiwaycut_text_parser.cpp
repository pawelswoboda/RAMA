#include "multiwaycut_text_parser.h"
#include <string>
#include <fstream>
#include <vector>
#include <tuple>
#include <sstream>
#include <cassert>


std::tuple<size_t, size_t, std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>> read_file(const std::string& filename)
{
    std::ifstream f;
    f.open(filename);
    if(!f.is_open())
        throw std::runtime_error("Could not open multiwaycut input file " + filename);

    std::string init_line;
    std::getline(f, init_line);
    if(init_line != "MULTIWAYCUT")
        throw std::runtime_error("first line must be 'MULTIWAYCUT'");

    // Read the class costs for each line until we encounter an empty line
    size_t n = 0;
    size_t k = 0;
    std::vector<float> node_class_cost_vec;
    std::string line;
    float class_cost;
    // TODO: check if every line contains the same number of classes
    while (std::getline(f, line, '\n') && !line.empty())
    {
        n += 1;
        k = 0;
        std::istringstream stream(line);
        while(stream >> class_cost || !stream.eof()) 
        {
            k += 1;
            node_class_cost_vec.push_back(class_cost);
        }
    }

    // Read the edge costs
    std::vector<int> i_vec;
    std::vector<int> j_vec;
    std::vector<float> cost_vec;
    int i, j;
    float cost;
    while(f >> i >> j >> cost)
    {
        i_vec.push_back(i);
        j_vec.push_back(j);
        cost_vec.push_back(cost);
    }

    return {n, k, node_class_cost_vec, i_vec, j_vec, cost_vec};
}
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>> mwc_to_coo(size_t n,
                size_t k,
                std::vector<float> class_costs,
                std::vector<int> src,
                std::vector<int> dest,
                std::vector<float> edge_costs)
{
    assert(class_costs.size() == n*k);
    for (int node = 0; node < n; ++node) {
        for (int cls = 0; cls < k; ++cls) {
            src.push_back(node);
            dest.push_back(cls);
            edge_costs.push_back(class_costs[node + cls]);
        }
    }
    return {src, dest, edge_costs};
}
