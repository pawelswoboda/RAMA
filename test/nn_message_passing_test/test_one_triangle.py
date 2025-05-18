import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import rama_py
import torch
from message_passing_nn.mlp_message_passing import MLPMessagePassing
from message_passing_nn.nn_utils import extract_data, lower_bound

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MLPMessagePassing().to(device)
model.eval()
MODEL_PATH = "./mlp_model.pt"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

i = [0, 0, 1]
j = [1, 2, 2]
costs = [1.5, -2.3, 0.8]

mp_data = rama_py.get_message_passing_data(i, j, costs, 3)
edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = extract_data(mp_data, device)

with torch.no_grad():
    updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
        edge_costs, t12_costs, t13_costs, t23_costs,
        corr_12, corr_13, corr_23, edge_counter, dist="mlp"
    )

lb = lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)
assert(-1.51 <= lb <= -1.49)




