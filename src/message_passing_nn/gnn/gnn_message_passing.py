import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNMessagePassing(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=3):
        super(GNNMessagePassing, self).__init__()

        self.gnn1 = GCNConv(input_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def send_messages_to_triplets(self, edge_costs, t12, t13, t23,
                                     corr_12, corr_13, corr_23, edge_counter):

        edge_vals_12 = edge_costs[corr_12]
        counts_12 = edge_counter[corr_12]
        t12 = t12 + edge_vals_12 / counts_12

        edge_vals_13 = edge_costs[corr_13]
        counts_13 = edge_counter[corr_13]
        t13 = t13 + edge_vals_13 / counts_13

        edge_vals_23 = edge_costs[corr_23]
        counts_23 = edge_counter[corr_23]
        t23 = t23 + edge_vals_23 / counts_23

        mask = edge_counter > 0
        edge_costs[mask] = 0.0

        return edge_costs, t12, t13, t23

    def send_messages_to_edges(self, edge_costs, t12, t13, t23,
                                corr_12, corr_13, corr_23, edge_index):
        
        tri_features = torch.stack([t12, t13, t23], dim=1)

        x = self.gnn1(tri_features, edge_index)
        x = torch.relu(x)
        x = self.gnn2(x, edge_index)
        x = torch.relu(x)
        
        delta = self.out_layer(x)

        t12 = t12 - delta[:, 0]
        t13 = t13 - delta[:, 1]
        t23 = t23 - delta[:, 2]

        edge_updates = torch.zeros_like(edge_costs)
        edge_updates.scatter_add_(0, corr_12, delta[:, 0])
        edge_updates.scatter_add_(0, corr_13, delta[:, 1])
        edge_updates.scatter_add_(0, corr_23, delta[:, 2])
        edge_costs = edge_costs + edge_updates

        return edge_costs, t12, t13, t23

    def forward(self, edge_costs, t12, t13, t23,
                      corr_12, corr_13, corr_23,
                      edge_counter, edge_index):
        edge_costs, t12, t13, t23 = self.send_messages_to_triplets(
            edge_costs, t12, t13, t23, corr_12, corr_13, corr_23, edge_counter
        )
        edge_costs, t12, t13, t23 = self.send_messages_to_edges(
            edge_costs, t12, t13, t23, corr_12, corr_13, corr_23, edge_index
        )
        return edge_costs, t12, t13, t23
