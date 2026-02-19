#pragma once

#include <string>
#include <vector>

struct MulticutInstance {
    std::vector<int> i, j;
    std::vector<float> costs;
    std::vector<int> lifted_i, lifted_j;
    std::vector<float> lifted_costs;
};

MulticutInstance read_file(const std::string& filename);