#pragma once

#include <vector>
#include <random>

struct RandomGraph {
    int num_nodes;
    std::vector<int> tails;
    std::vector<int> heads;
    std::vector<float> costs;
    std::vector<int> lifted_tails;
    std::vector<int> lifted_heads;
    std::vector<float> lifted_costs;
};

// Generate a random graph with base and lifted edges.
// Each pair (i,j) with i<j is independently included as a base edge with
// probability base_edge_prob and as a lifted edge with probability lifted_edge_prob.
// Base costs are drawn uniformly from [-1, 1].
// Lifted costs are drawn uniformly from [-1, 1].
inline RandomGraph generate_random_graph(
    const int num_nodes, const double base_edge_prob,
    const double lifted_edge_prob, const unsigned seed)
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
            if (edge_dist(rng) < base_edge_prob)
            {
                g.tails.push_back(i);
                g.heads.push_back(j);
                g.costs.push_back(cost_dist(rng));
            }
            if (edge_dist(rng) < lifted_edge_prob)
            {
                g.lifted_tails.push_back(i);
                g.lifted_heads.push_back(j);
                g.lifted_costs.push_back(cost_dist(rng));
            }
        }
    }

    return g;
}

// Generate a random base-only graph (no lifted edges).
// Preserves the original RNG sequence for backward compatibility.
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