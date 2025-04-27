import torch
import torch.nn as nn
from torch_scatter import scatter_softmax

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        return x / (rms + self.eps) * self.weight

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential( 
            RMSNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.block(x)

class EdgeToTriMLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(8, 16),
            nn.GELU(),
            ResBlock(16),
            nn.LayerNorm(16),
            ResBlock(16),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.LayerNorm(8),
            nn.Linear(8, output_dim)
        )

    def forward(self, edge_costs, edge_ids_all):
        logits = self.mlp(edge_costs).squeeze(-1) 
        # [10,-4,19,14,-9,-2]
        weights = scatter_softmax(logits, edge_ids_all, dim=0) 
        # Kante 4: softmax([10,-4]) = [0.8, 0.2]
        # Kante 5: softmax([19]) = [1.0]
        # Kante 9: softmax([14]) = [1.0]
        # Kante 6: softmax([-9]) = [1.0]
        # Kante 10: softmax([-2]) = [1.0]
        # weights = [0.8, 0.2, 1.0, 1.0, 1.0, 1.0]
        return weights
    
class EdgeToTriAttention(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, d_k=8):
        super().__init__()
        self.d_k = d_k
        self.query = nn.Linear(input_dim, d_k)  # input: [edge_cost, edge_counter]
        self.key = nn.Linear(input_dim, d_k)
        self.value = nn.Linear(input_dim, d_k)
        self.attention_mlp = nn.Sequential(
            nn.Linear(d_k, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            ResBlock(32),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)   # output = weights
        )

    def forward(self, edge_features, edge_ids_all, triangle_ids_all):
        queries = self.query(edge_features)  # [num_edges, 2] -> [num_edges, d_k = 8]
        keys = self.key(edge_features)       
        values = self.value(edge_features)   

        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        attention_scores = torch.matmul(queries, keys.transpose(0, 1)) / (self.d_k ** 0.5)  # [num_edges, num_edges], aehnlichkeitsmatrix   
        attention_mask = (triangle_ids_all.unsqueeze(1) == triangle_ids_all.unsqueeze(0)) # nur kanten in gleichen triangles berücksichtigen
        attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf')) # kanten aus verschiedenen triangles kriegen -inf score
        attention_weights = torch.softmax(attention_scores, dim=1)  # [num_edges, num_edges]
        attended_values = torch.matmul(attention_weights, values)  # [num_edges, 8]
        
        logits = self.attention_mlp(attended_values).squeeze(-1)  # [num_edges]
        weights = scatter_softmax(logits, edge_ids_all, dim=0)
        
        return weights
    
class TriToEdgeMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            nn.Linear(32, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

class MLPMessagePassing(nn.Module):
    def __init__(self):
        super(MLPMessagePassing, self).__init__()
        self.edge_to_tri_mlp = EdgeToTriMLP(input_dim=1, output_dim=1) # input = edge_cost, output = weights
        self.edge_to_tri_attention = EdgeToTriAttention(input_dim=2, output_dim=1, d_k=8) # input = [edge_cost, edge_counter], output = weights
        self.tri_to_edge_mlp = TriToEdgeMLP(input_dim=3, output_dim=3) # input = [t12, t13, t23], output = [t12_updated, t13_updated, t23_updated]
    
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
                                   corr_12, corr_13, corr_23, edge_counter):

        # bspw: 2 triangles: [4,5,6], [4,9,10], corr_12 = [4,4] etc.
        num_triangles = t12.shape[0] 
        # 2
        triangle_ids = torch.arange(num_triangles, device=edge_costs.device) 
        # [0,1]
        edge_ids_all = torch.cat([corr_12, corr_13, corr_23], dim=0) 
        # [4,4,5,9,6,10]
        triangle_ids_all = torch.cat([triangle_ids, triangle_ids, triangle_ids], dim=0)
        # [0,1,0,1,0,1]
        positions = torch.cat([
            torch.full_like(corr_12, 0),  # t12
            torch.full_like(corr_13, 1),  # t13
            torch.full_like(corr_23, 2)   # t23
        ])
        # [0,0,1,1,2,2] 0 for t12, 1 for t13, 2 for t23
        # mapping (lagrange_id in triangle_id) -> edge_id 

        edge_costs_input = edge_costs[edge_ids_all].unsqueeze(-1)      

        weights = self.edge_to_tri_mlp(edge_costs_input, edge_ids_all).squeeze(-1) 
        contrib = edge_costs[edge_ids_all] * weights  

        # [edge_costs[4] * 0.8, edge_costs[4] * 0.2, edge_costs[5] * 1, ... ] 
 
        t12_updated = t12.clone()
        t13_updated = t13.clone()
        t23_updated = t23.clone()

        mask_12 = positions == 0 # [True, True, False, False, False, False]
        mask_13 = positions == 1 # [False, False, True, True, False, False]
        mask_23 = positions == 2 # [False, False, False, False, True, True]

        t12_updated.scatter_add_(0, triangle_ids_all[mask_12], contrib[mask_12])
        # contrib[mask_12] = [edge_costs[4] * 0.8, edge_costs[4] * 0.2]
        # triangle_ids_all[mask_12] = [0,1]  
        # t12[0] += edge_costs[4] * 0.8  
        # t12[1] += edge_costs[4] * 0.2 
        t13_updated.scatter_add_(0, triangle_ids_all[mask_13], contrib[mask_13])
        # contrib[mask_13] = [edge_costs[5] * 1, edge_costs[9] * 1]
        # triangle_ids_all[mask_13] = [0,1]  
        # t13[0] += edge_costs[5] * 1  
        # t13[1] += edge_costs[9] * 1 
        t23_updated.scatter_add_(0, triangle_ids_all[mask_23], contrib[mask_23])
        # contrib[mask_23] = [edge_costs[6] * 1, edge_costs[10] * 1]
        # triangle_ids_all[mask_23] = [0,1]  
        # t23[0] += edge_costs[6] * 1  
        # t23[1] += edge_costs[10] * 1 

        mask = edge_counter > 0
        edge_costs[mask] = 0.0

        return edge_costs, t12_updated, t13_updated, t23_updated
    
    def send_messages_to_triplets_attention(self, edge_costs, t12, t13, t23,
                                         corr_12, corr_13, corr_23, edge_counter):

        num_triangles = t12.shape[0] 
        triangle_ids = torch.arange(num_triangles, device=edge_costs.device) 
        edge_ids_all = torch.cat([corr_12, corr_13, corr_23], dim=0) 
        triangle_ids_all = torch.cat([triangle_ids, triangle_ids, triangle_ids], dim=0)
        positions = torch.cat([
            torch.full_like(corr_12, 0),  
            torch.full_like(corr_13, 1),  
            torch.full_like(corr_23, 2)   
        ])

        edge_features = torch.stack([
            edge_costs[edge_ids_all],
            edge_counter[edge_ids_all].float()
        ], dim=1)  

        weights = self.edge_to_tri_attention(edge_features, edge_ids_all, triangle_ids_all)
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
                        tri_corr_12, tri_corr_13, tri_corr_23, edge_counter):
        
        edge_costs, t12_costs, t13_costs, t23_costs = self.send_messages_to_triplets(
            edge_costs, t12_costs, t13_costs, t23_costs,
            tri_corr_12, tri_corr_13, tri_corr_23, edge_counter
        )

        edge_costs, t12_costs, t13_costs, t23_costs = self.send_messages_to_edges_mlp(
            edge_costs, t12_costs, t13_costs, t23_costs,
            tri_corr_12, tri_corr_13, tri_corr_23
        )

        return edge_costs, t12_costs, t13_costs, t23_costs