#pragma once

#include <vector>
#include <tuple>
#include "dCOO.h"

std::vector<int> parallel_gaec_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs); 

std::vector<int> parallel_gaec_cuda(dCOO& A);
