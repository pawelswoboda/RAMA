import torch
import torch.nn as nn

class MLPMessagePassing(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=3):
        super(MLPMessagePassing, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def send_messages_to_triplets(self, data):
        for key, corr_key in zip(
            ["t12_costs", "t13_costs", "t23_costs"],
            ["tri_corr_12", "tri_corr_13", "tri_corr_23"]
        ):
            edge_vals = data["edge_costs"].gather(0, data[corr_key])
            counts = data["edge_counter"].gather(0, data[corr_key]).float().clamp(min=1)
            data[key] = data[key] + edge_vals / counts

        mask = data["edge_counter"] > 0
        data["edge_costs"] = torch.where(mask, torch.zeros_like(data["edge_costs"]), data["edge_costs"])
        
        return data

    def send_messages_to_edges_mlp(self, data):
        tri_features = torch.stack([data["t12_costs"], data["t13_costs"], data["t23_costs"]], dim=1)
        delta = self.mlp(tri_features)

        t12_old = data["t12_costs"].clone()
        t13_old = data["t13_costs"].clone()
        t23_old = data["t23_costs"].clone()

        data["t12_costs"] -= delta[:, 0]
        data["t13_costs"] -= delta[:, 1]
        data["t23_costs"] -= delta[:, 2]

        edge_updates = torch.zeros_like(data["edge_costs"])
        counts_12 = data["edge_counter"][data["tri_corr_12"]].float().clamp(min=1)
        counts_13 = data["edge_counter"][data["tri_corr_13"]].float().clamp(min=1)
        counts_23 = data["edge_counter"][data["tri_corr_23"]].float().clamp(min=1)

        edge_updates = edge_updates.scatter_add(0, data["tri_corr_12"], t12_old / counts_12)
        edge_updates = edge_updates.scatter_add(0, data["tri_corr_13"], t13_old / counts_13)
        edge_updates = edge_updates.scatter_add(0, data["tri_corr_23"], t23_old / counts_23)

        data["edge_costs"] = data["edge_costs"] + edge_updates
        
        data["t12_costs"] = torch.zeros_like(t12_old)
        data["t13_costs"] = torch.zeros_like(t13_old)
        data["t23_costs"] = torch.zeros_like(t23_old)

        return data

    def forward(self, data):
        data = self.send_messages_to_triplets(data)
        data = self.send_messages_to_edges_mlp(data)
        return data
