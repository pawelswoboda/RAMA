#include "multicut_text_parser.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

MulticutInstance read_file(const std::string& filename)
{
    std::ifstream f;
    f.open(filename);
    if(!f.is_open())
        throw std::runtime_error("Could not open multicut input file " + filename);

    std::string init_line;
    std::getline(f, init_line);
    if (init_line != "MULTICUT" && init_line != "LIFTED MULTICUT")
        throw std::runtime_error("first line must be 'MULTICUT' or 'LIFTED MULTICUT'");

    MulticutInstance inst;
    std::string line;
    bool found_lifted_section = false;
    while (std::getline(f, line))
    {
        if (line == "LIFTED")
        {
            found_lifted_section = true;
            break;
        }

        if (line.empty())
            continue;

        int i, j;
        float cost;
        if (std::sscanf(line.c_str(), "%d %d %f", &i, &j, &cost) != 3)
            throw std::runtime_error("Parse error in " + filename + ": '" + line + "'");

        inst.i.push_back(i);
        inst.j.push_back(j);
        inst.costs.push_back(cost);
    }

    if (found_lifted_section)
    {
        while (std::getline(f, line))
        {
            if (line.empty())
                continue;

            int i, j;
            float cost;
            if (std::sscanf(line.c_str(), "%d %d %f", &i, &j, &cost) != 3)
                throw std::runtime_error("Parse error in lifted section of " + filename + ": '" + line + "'");

            inst.lifted_i.push_back(i);
            inst.lifted_j.push_back(j);
            inst.lifted_costs.push_back(cost);
        }
    }

    int max_node = 0;
    if (!inst.i.empty()) max_node = std::max(max_node, *std::max_element(inst.i.begin(), inst.i.end()));
    if (!inst.j.empty()) max_node = std::max(max_node, *std::max_element(inst.j.begin(), inst.j.end()));
    if (!inst.lifted_i.empty()) max_node = std::max(max_node, *std::max_element(inst.lifted_i.begin(), inst.lifted_i.end()));
    if (!inst.lifted_j.empty()) max_node = std::max(max_node, *std::max_element(inst.lifted_j.begin(), inst.lifted_j.end()));
    int num_nodes = inst.i.empty() ? 0 : max_node + 1;

    std::cout << "Graph: " << num_nodes << " nodes, "
              << inst.i.size() << " base edges";
    if (!inst.lifted_i.empty())
        std::cout << ", " << inst.lifted_i.size() << " lifted edges";
    std::cout << "\n";

    return inst;
}