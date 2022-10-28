#ifndef MULTIWAYCUT_TEXT_PARSER_H
#define MULTIWAYCUT_TEXT_PARSER_H

#include <string>
#include <vector>

/**
 * @brief Reads a multiway cut file of the following format:
 * 
 * ```
 * MULTIWAYCUT
 * c_11 c_12 ... c_1k
 * ...
 * c_n1 c_n2 ... c_nk
 * 
 * i_1 j_1 cost_1
 * i_2 j_2 cost_2
 * ...
 * i_n j_n cost_n
 * ```
 * 
 * The first part of the header describes the cost of node i being in class k.
 * The second part is the same as for the multicut text file, i.e. the edge costs.
 * 
 * @param filename 
 * @return std::tuple<
 * std::vector<float>,
 * std::vector<int>,
 * std::vector<int>,
 * std::vector<float>
 * > Returns a tuple with the number of nodes n, the number of classes k, a n*k vector representing the node-class cost matrix, a vector for the sources, destinations and costs of each edge.
 */
std::tuple<size_t, size_t, std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>> read_file(const std::string& filename);

#endif
