import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
from config import ModelConfig

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential( 
            nn.RMSNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU()
            )
    def forward(self, x):
        return x + self.block(x)

class EdgeToTriMLP(nn.Module):
    def __init__(self, config: ModelConfig, input_dim=3, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, config.hidden_dim)]  # input = [edge_cost, edge_counter, lagrange_multiplier]
        for _ in range(config.num_res_blocks):
            layers.append(ResBlock(config.hidden_dim))
        layers.append(nn.Linear(config.hidden_dim, output_dim))  # output = weights
        self.mlp = nn.Sequential(*layers)
        self._init_weights(config)

    def _init_weights(self, config: ModelConfig):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=config.weight_init_mean, std=config.weight_init_std)
                nn.init.zeros_(m.bias)

    def forward(self, edge_features, edge_ids_all):
        logits = self.mlp(edge_features).squeeze(-1) 
        weights = scatter_softmax(logits, edge_ids_all, dim=0) 

        return weights
    
class EdgeToTriAttention(nn.Module):
    def __init__(self, config: ModelConfig, input_dim=3, output_dim=1):
        super().__init__()
        self.d_k = config.d_k
        self.query = nn.Linear(input_dim, config.d_k)  # input = [edge_cost, edge_counter, lagrange_multiplier]
        self.key = nn.Linear(input_dim, config.d_k)
        self.value = nn.Linear(input_dim, config.d_k)
        
        layers = [nn.Linear(config.d_k, config.hidden_dim)]
        for _ in range(config.num_res_blocks):
            layers.append(ResBlock(config.hidden_dim))
        layers.append(nn.Linear(config.hidden_dim, output_dim))  # output = weights
        self.attention_mlp = nn.Sequential(*layers)
        self._init_weights(config)

    def _init_weights(self, config: ModelConfig):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=config.weight_init_mean, std=config.weight_init_std)
                nn.init.zeros_(m.bias)

    def forward(self, edge_features, edge_ids_all, triangle_ids_all):
        queries = self.query(edge_features)  # [num_edges, 2] -> [num_edges, d_k = 16]
        keys = self.key(edge_features)
        values = self.value(edge_features)

        attention_scores = torch.matmul(queries, keys.transpose(0, 1)) / (self.d_k ** 0.5) 
        attention_mask = (triangle_ids_all.unsqueeze(1) == triangle_ids_all.unsqueeze(0)) # nur kanten in gleichen triangles berücksichtigen
        attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf')) # kanten aus verschiedenen triangles kriegen -inf score
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended_values = torch.matmul(attention_weights, values)  # [num_edges, d_k]
        
        logits = self.attention_mlp(attended_values).squeeze(-1)  # [num_edges]
        weights = scatter_softmax(logits, edge_ids_all, dim=0)
        
        return weights
    
class TriToEdgeMLP(nn.Module):
    def __init__(self, config: ModelConfig, input_dim=3, output_dim=3):
        super().__init__()
        layers = [nn.Linear(input_dim, config.hidden_dim)]  # input = [t12, t13, t23]
        for _ in range(config.num_res_blocks):
            layers.append(ResBlock(config.hidden_dim))
        layers.append(nn.Linear(config.hidden_dim, output_dim))  # output = [t12_updated, t13_updated, t23_updated]
        self.mlp = nn.Sequential(*layers)
        self._init_weights(config)

    def _init_weights(self, config: ModelConfig):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=config.weight_init_mean, std=config.weight_init_std)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

class MLPMessagePassing(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super(MLPMessagePassing, self).__init__()
        self.edge_to_tri_mlp = EdgeToTriMLP(config, input_dim=3, output_dim=1)  # input = [edge_cost, edge_counter, lagrange_multiplier], output = weights
        self.edge_to_tri_attention = EdgeToTriAttention(config, input_dim=3, output_dim=1)  # input = [edge_cost, edge_counter, lagrange_multiplier], output = weights
        self.tri_to_edge_mlp = TriToEdgeMLP(config, input_dim=3, output_dim=3)  # input = [t12, t13, t23], output = [t12_updated, t13_updated, t23_updated]
    
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

    def send_messages_to_triplets_mlp(self, edge_costs, t12, t13, t23,
                                   corr_12, corr_13, corr_23, edge_counter, use_attention=False):

        num_triangles = t12.shape[0] 
        triangle_ids = torch.arange(num_triangles, device=edge_costs.device) 
        edge_ids_all = torch.cat([corr_12, corr_13, corr_23], dim=0) 
        triangle_ids_all = torch.cat([triangle_ids, triangle_ids, triangle_ids], dim=0)
        positions = torch.cat([
            torch.full_like(corr_12, 0),  # t12
            torch.full_like(corr_13, 1),  # t13
            torch.full_like(corr_23, 2)   # t23
        ])
        
        lagrange_multipliers_all = torch.cat([
            t12[triangle_ids], 
            t13[triangle_ids], 
            t23[triangle_ids]
        ])
        
        edge_features = torch.stack([
            edge_costs[edge_ids_all],  
            edge_counter[edge_ids_all].float(),
            lagrange_multipliers_all   
        ], dim=1)  

        if use_attention:
            weights = self.edge_to_tri_attention(edge_features, edge_ids_all, triangle_ids_all)
        else:
            weights = self.edge_to_tri_mlp(edge_features, edge_ids_all).squeeze(-1) 
        contrib = edge_costs[edge_ids_all] * weights  
 
        t12_updated = t12.clone()
        t13_updated = t13.clone()
        t23_updated = t23.clone()

        mask_12 = positions == 0
        mask_13 = positions == 1
        mask_23 = positions == 2

        t12_updated.scatter_add_(0, triangle_ids_all[mask_12], contrib[mask_12])
        t13_updated.scatter_add_(0, triangle_ids_all[mask_13], contrib[mask_13])
        t23_updated.scatter_add_(0, triangle_ids_all[mask_23], contrib[mask_23])

        mask = edge_counter > 0
        edge_costs[mask] = 0.0

        return edge_costs, t12_updated, t13_updated, t23_updated
    


    def send_messages_to_edges_mlp(self, edge_costs, t12, t13, t23,
                                          corr_12, corr_13, corr_23):

        tri_features = torch.stack([t12, t13, t23], dim=1)
        delta = self.tri_to_edge_mlp(tri_features)

        t12 = t12 - delta[:, 0] 
        t13 = t13 - delta[:, 1] 
        t23 = t23 - delta[:, 2] 

        edge_updates = torch.zeros_like(edge_costs)
        edge_updates.scatter_add_(0, corr_12, delta[:, 0]) 
        edge_updates.scatter_add_(0, corr_13, delta[:, 1]) 
        edge_updates.scatter_add_(0, corr_23, delta[:, 2]) 

        edge_costs = edge_costs + edge_updates
        
        return edge_costs, t12, t13, t23

    def forward(self, edge_costs, t12_costs, t13_costs, t23_costs,
                        tri_corr_12, tri_corr_13, tri_corr_23, edge_counter, dist="mlp"):
        if dist == "uniform":
            edge_costs, t12_costs, t13_costs, t23_costs = self.send_messages_to_triplets(
                edge_costs, t12_costs, t13_costs, t23_costs,
                tri_corr_12, tri_corr_13, tri_corr_23, edge_counter
            )
        elif dist in ["mlp", "attention"]:
            edge_costs, t12_costs, t13_costs, t23_costs = self.send_messages_to_triplets_mlp(
                edge_costs, t12_costs, t13_costs, t23_costs,
                tri_corr_12, tri_corr_13, tri_corr_23, edge_counter, use_attention= True if dist == "attention" else False
            )

        edge_costs, t12_costs, t13_costs, t23_costs = self.send_messages_to_edges_mlp(
            edge_costs, t12_costs, t13_costs, t23_costs,
            tri_corr_12, tri_corr_13, tri_corr_23
        )

        return edge_costs, t12_costs, t13_costs, t23_costs 