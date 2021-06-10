#pragma once

#include <vector>
#include <tuple>

struct weighted_edge { int i; int j; float val; };

std::vector<int> parallel_gaec(const std::vector<weighted_edge>& edges, const double contract_ratio, const size_t num_itr);

