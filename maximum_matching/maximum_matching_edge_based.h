#pragma once

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> maximum_matching_edge_based(const int numVertices, int numEdges, int* i, int* j, int* w);
