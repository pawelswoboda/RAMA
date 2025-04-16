# gnn_message_passing.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNNMessagePassing(nn.Module):
    def __init__(self, use_edge_features=True, node_dim=1, hidden_dim=32, triangle_dim=3):
        super(GNNMessagePassing, self).__init__()
        self.use_edge_features = use_edge_features

        # GNN zur Berechnung von Kontext-Information für Knoten
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Input-Dimension MLP: 3 triangle costs + node context [+ optional edge costs]
        input_dim = triangle_dim + hidden_dim
        if use_edge_features:
            input_dim += triangle_dim  # edge_costs for 3 edges

        self.triangle_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, triangle_dim)
        )

    def forward(self, edge_costs, t12, t13, t23,
                corr_12, corr_13, corr_23, edge_counter, edge_index, num_nodes):

        device = edge_costs.device

        # Dummy node features (1 for each node)
        
        x = torch.ones((num_nodes, 1), device=device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if corr_12.max() >= edge_costs.size(0):
            print("[ERROR] corr_12 index out of bounds:", corr_12.max().item(), ">=", edge_costs.size(0))
        if corr_13.max() >= edge_costs.size(0):
            print("[ERROR] corr_13 index out of bounds:", corr_13.max().item(), ">=", edge_costs.size(0))
        if corr_23.max() >= edge_costs.size(0):
            print("[ERROR] corr_23 index out of bounds:", corr_23.max().item(), ">=", edge_costs.size(0))

        if corr_12.max() >= x.size(0):
            print("[ERROR] corr_12 node out of bounds:", corr_12.max().item(), ">=", x.size(0))
        # Node context für jedes Dreieck (Durchschnitt der 3 involvierten Knoten)
        node_context = (x[corr_12] + x[corr_13] + x[corr_23]) / 3.0

        # Triangle costs
        triangle_costs = torch.stack([t12, t13, t23], dim=1)

        features = [triangle_costs, node_context]

        if self.use_edge_features:
            # Hole auch die aktuellen Edge-Kosten der Kanten in jedem Triangle
            edge_feats = torch.stack([
                edge_costs[corr_12],
                edge_costs[corr_13],
                edge_costs[corr_23]
            ], dim=1)
            features.append(edge_feats)

        # Kombiniere alle Features
        mlp_input = torch.cat(features, dim=1)

        # Berechne Updates pro Dreieck
        delta = self.triangle_mlp(mlp_input)

        # Update Triangle-Kosten
        t12 = t12 - delta[:, 0]
        t13 = t13 - delta[:, 1]
        t23 = t23 - delta[:, 2]

        # Aggregiere Updates zurück zu Kanten
        edge_updates = torch.zeros_like(edge_costs)
        edge_updates.scatter_add_(0, corr_12, delta[:, 0])
        edge_updates.scatter_add_(0, corr_13, delta[:, 1])
        edge_updates.scatter_add_(0, corr_23, delta[:, 2])

        edge_costs = edge_costs + edge_updates

        return edge_costs, t12, t13, t23
