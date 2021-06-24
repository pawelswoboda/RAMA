#include <string>
#include <fstream>
#include <vector>
#include <tuple>

std::tuple<std::vector<int>, std::vector<int>, std::vector<float>> read_file(const std::string& filename)
{
    std::ifstream f;
    f.open(filename);
    if(!f.is_open())
        throw std::runtime_error("Could not open multicut input file " + filename);

    std::string init_line;
    std::getline(f, init_line);
    if(init_line != "MULTICUT")
        throw std::runtime_error("first line must be 'MULTICUT'");

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

    return {i_vec, j_vec, cost_vec};
}
