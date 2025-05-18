import torch
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_data(mp_data, device):
    edge_costs = torch.tensor(mp_data["edge_costs"], dtype=torch.float32).to(device)
    t12_costs = torch.tensor(mp_data["t12_costs"], dtype=torch.float32).to(device)
    t13_costs = torch.tensor(mp_data["t13_costs"], dtype=torch.float32).to(device)
    t23_costs = torch.tensor(mp_data["t23_costs"], dtype=torch.float32).to(device)
    corr_12 = torch.tensor(mp_data["tri_corr_12"], dtype=torch.long).to(device)
    corr_13 = torch.tensor(mp_data["tri_corr_13"], dtype=torch.long).to(device)
    corr_23 = torch.tensor(mp_data["tri_corr_23"], dtype=torch.long).to(device)
    edge_counter = torch.tensor(mp_data["edge_counter"], dtype=torch.int32).to(device)

    return edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter

def lower_bound(edge_costs, t12, t13, t23):
    edge_lb = torch.sum(torch.where(edge_costs < 0, edge_costs, torch.zeros_like(edge_costs)))
        
    a, b, c = t12, t13, t23
    zero = torch.zeros_like(a)

    lb = torch.stack([
        zero,
        a + b,
        a + c,
        b + c,
        a + b + c
    ])
    tri_lb = torch.min(lb, dim=0).values.sum()
    return edge_lb + tri_lb
   
def normalise_costs(costs, eps=1e-6, device=None):
    costs = torch.tensor(costs, dtype=torch.float32, device=device)
    scale = costs.abs().max()
    factor = scale + eps
    return costs / factor, factor
