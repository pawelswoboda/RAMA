#pragma once

#include <vector>
#include <random>

struct RandomGraph {
    int num_nodes;
    std::vector<int> tails;
    std::vector<int> heads;
    std::vector<float> costs;
};

// Generate a random graph with num_nodes nodes.
// Each pair (i,j) with i<j is included as an edge with probability edge_prob.
// Costs are drawn uniformly from [-1, 1].
inline RandomGraph generate_random_graph(
    const int num_nodes, const double edge_prob, const unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> edge_dist(0.0, 1.0);
    std::uniform_real_distribution<float> cost_dist(-1.0f, 1.0f);

    RandomGraph g;
    g.num_nodes = num_nodes;

    for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = i + 1; j < num_nodes; ++j)
        {
            if (edge_dist(rng) < edge_prob)
            {
                g.tails.push_back(i);
                g.heads.push_back(j);
                g.costs.push_back(cost_dist(rng));
            }
        }
    }

    return g;
}
