#include "multicut_text_parser.h"
#include <fstream>
#include <stdexcept>

MulticutInstance read_file(const std::string& filename)
{
    std::ifstream f;
    f.open(filename);
    if(!f.is_open())
        throw std::runtime_error("Could not open multicut input file " + filename);

    std::string init_line;
    std::getline(f, init_line);

    bool is_lifted = false;
    if (init_line == "LIFTED MULTICUT")
        is_lifted = true;
    else if (init_line != "MULTICUT")
        throw std::runtime_error("first line must be 'MULTICUT' or 'LIFTED MULTICUT'");

    MulticutInstance inst;
    std::string line;
    while (std::getline(f, line))
    {
        if (is_lifted && line == "LIFTED")
            break;

        int i, j;
        float cost;
        if (std::sscanf(line.c_str(), "%d %d %f", &i, &j, &cost) == 3)
        {
            inst.i.push_back(i);
            inst.j.push_back(j);
            inst.costs.push_back(cost);
        }
    }

    if (is_lifted)
    {
        int i, j;
        float cost;
        while (f >> i >> j >> cost)
        {
            inst.lifted_i.push_back(i);
            inst.lifted_j.push_back(j);
            inst.lifted_costs.push_back(cost);
        }
    }

    return inst;
}