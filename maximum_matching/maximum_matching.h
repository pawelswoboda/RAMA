#pragma once

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> maximum_matching(const int numVertices, int numEdges, int* i, int* j, int* w);
