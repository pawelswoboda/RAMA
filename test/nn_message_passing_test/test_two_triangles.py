import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import rama_py
import torch
from message_passing_nn.mlp_message_passing import MLPMessagePassing
from message_passing_nn.nn_utils import extract_data, lower_bound
from message_passing_nn.config.config import ModelConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# Erstelle eine ModelConfig mit d_k
config = ModelConfig()
config.d_k = 8  # Setze einen Wert für d_k

model = MLPMessagePassing(config).to(device)
model.eval()
MODEL_PATH = "./mlp_model.pt"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

i = [0, 0, 1, 2, 1]  
j = [1, 2, 2, 3, 3]  
costs = [1.0, -2.0, 1.5, 0.8, -1.2]  

mp_data = rama_py.get_message_passing_data(i, j, costs, 3)
edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = extract_data(mp_data, device)

with torch.no_grad():
    for _ in range(2):
        updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                edge_costs, t12_costs, t13_costs, t23_costs,
                corr_12, corr_13, corr_23, edge_counter, dist="mlp"
        )
        edge_costs, t12_costs, t13_costs, t23_costs = updated_edge_costs, updated_t12, updated_t13, updated_t23

lb = lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)
assert (-1.71 < lb.item() < -1.70)