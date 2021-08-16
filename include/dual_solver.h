#pragma once

#include "dCOO.h"

double dual_solver(dCOO& A, const int max_cycle_length, const int num_iter, const float tri_memory_factor, const int num_outer_dual_itr = 1, const float tol_ratio = 1e-4);