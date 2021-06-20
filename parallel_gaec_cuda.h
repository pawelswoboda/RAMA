#pragma once

#include <vector>
#include <tuple>

std::vector<int> parallel_gaec_cuda(const std::vector<std::tuple<int,int,float>>& edges); 
