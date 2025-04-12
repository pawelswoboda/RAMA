import torch
import torch.nn as nn

class MLPMessagePassing(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(MLPMessagePassing, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, output_dim)
        )

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

    def send_messages_to_edges_mlp(self, edge_costs, t12, t13, t23,
                                          corr_12, corr_13, corr_23):

        tri_features = torch.stack([t12, t13, t23], dim=1)

        delta = self.mlp(tri_features)

        t12 = t12 - delta[:, 0] 
        t13 = t13 - delta[:, 1] 
        t23 = t23 - delta[:, 2] 

        edge_updates = torch.zeros_like(edge_costs)
        edge_updates.scatter_add_(0, corr_12, delta[:, 0]) 
        edge_updates.scatter_add_(0, corr_13, delta[:, 1]) 
        edge_updates.scatter_add_(0, corr_23, delta[:, 2]) 
        #edge_updates[corr_12] = edge_updates[corr_12] + delta[:, 0]
        #edge_updates[corr_13] = edge_updates[corr_13] + delta[:, 1]
        #edge_updates[corr_23] = edge_updates[corr_23] + delta[:, 2]

        edge_costs = edge_costs + edge_updates
        
        return edge_costs, t12, t13, t23
    
    def forward(self, edge_costs, t12_costs, t13_costs, t23_costs,
                        tri_corr_12, tri_corr_13, tri_corr_23, edge_counter):

        #print("[INFO] USING PYTHON ")
        
        edge_costs, t12_costs, t13_costs, t23_costs = self.send_messages_to_triplets(
            edge_costs, t12_costs, t13_costs, t23_costs,
            tri_corr_12, tri_corr_13, tri_corr_23, edge_counter
        )

        edge_costs, t12_costs, t13_costs, t23_costs = self.send_messages_to_edges_mlp(
            edge_costs, t12_costs, t13_costs, t23_costs,
            tri_corr_12, tri_corr_13, tri_corr_23
        )

        return edge_costs, t12_costs, t13_costs, t23_costs