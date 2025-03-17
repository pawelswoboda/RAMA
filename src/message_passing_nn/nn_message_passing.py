import torch
import cupy as cp
import numpy as np
from mp.dbca_message_passing import ClassicalMessagePassing
from torch.utils.data import DataLoader, Dataset
from mp.mlp_message_passing import MLPMessagePassing
import os


TORCH_DTYPE_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64
}

def ptr_to_tensor(ptr, num_elements, dtype, device='cuda'):
   
    if dtype not in TORCH_DTYPE_TO_NUMPY:
        raise TypeError(f"Unsupported dtype: {dtype}")
    np_dtype = TORCH_DTYPE_TO_NUMPY[dtype]

    cp_array = cp.ndarray(
        (num_elements,),
        dtype=np_dtype,
        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, num_elements * np_dtype().itemsize, None), 0)
    )
    torch_tensor = torch.as_tensor(cp_array).to(device).clone()
    return torch_tensor


def nn_update(edge_costs_ptr, t1_ptr, t2_ptr, t3_ptr, i_ptr, j_ptr, 
                       t12_costs_ptr, t13_costs_ptr, t23_costs_ptr,
                       triangle_corr_12_ptr, triangle_corr_13_ptr, triangle_corr_23_ptr,
                       edge_counter_ptr, num_edges, num_triangles, num_nodes):
    
    print("=== Python ===")
    
    edge_costs = ptr_to_tensor(edge_costs_ptr, num_edges, torch.float32)

    t1 = ptr_to_tensor(t1_ptr, num_triangles, torch.int32)
    t2 = ptr_to_tensor(t2_ptr, num_triangles, torch.int32)
    t3 = ptr_to_tensor(t3_ptr, num_triangles, torch.int32)

    i = ptr_to_tensor(i_ptr, num_nodes, torch.int32)
    j = ptr_to_tensor(j_ptr, num_nodes, torch.int32)

    t12_costs = ptr_to_tensor(t12_costs_ptr, num_triangles, torch.float32)
    t13_costs = ptr_to_tensor(t13_costs_ptr, num_triangles, torch.float32)
    t23_costs = ptr_to_tensor(t23_costs_ptr, num_triangles, torch.float32)

    tri_corr_12 = ptr_to_tensor(triangle_corr_12_ptr, num_triangles, torch.int32).long()
    tri_corr_13 = ptr_to_tensor(triangle_corr_13_ptr, num_triangles, torch.int32).long()
    tri_corr_23 = ptr_to_tensor(triangle_corr_23_ptr, num_triangles, torch.int32).long()
    edge_counter   = ptr_to_tensor(edge_counter_ptr, num_edges, torch.int32)

    updated_edge_costs, updated_t12_costs, updated_t13_costs, updated_t23_costs = via_mlp(
        edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
        t12_costs, t13_costs, t23_costs,edge_counter
    )

    return (
        updated_edge_costs.cpu().numpy(),
        updated_t12_costs.cpu().numpy(),
        updated_t13_costs.cpu().numpy(),
        updated_t23_costs.cpu().numpy()
    )




def via_dbca(edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
             t12_costs, t13_costs, t23_costs, edge_counter):

    mp = ClassicalMessagePassing(edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
                                   t12_costs, t13_costs, t23_costs, edge_counter)
    mp.iteration()  
    print("[PYTHON] Lower bound:", mp.compute_lower_bound())
    return mp.edge_costs, mp.t12_costs, mp.t13_costs, mp.t23_costs



def via_mlp(edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
            t12_costs, t13_costs, t23_costs, edge_counter,
            num_epochs=1, lr=1e-3):

    def compute_lower_bound(data):
        edge_lb = torch.sum(torch.where(data["edge_costs"] < 0, data["edge_costs"], torch.zeros_like(data["edge_costs"])))
        a, b, c = data["t12_costs"], data["t13_costs"], data["t23_costs"]
        tri_lb = torch.min(torch.stack([torch.zeros_like(a), a+b, a+c, b+c, a+b+c], dim=0), dim=0).values.sum()
        return edge_lb + tri_lb

    def loss_fn(data):
        return -compute_lower_bound(data)

    class SingleMulticutDataset(Dataset):
        def __init__(self, data):
            self.data = data  

        def __len__(self):
            return 1  

        def __getitem__(self, idx):
            return self.data
    
    device = edge_costs.device  
    MODEL_PATH = "./mlp_model.pt"
    
    data_dict = {
        "edge_costs": edge_costs.clone(),
        "tri_corr_12": tri_corr_12.clone(),
        "tri_corr_13": tri_corr_13.clone(),
        "tri_corr_23": tri_corr_23.clone(),
        "t12_costs": t12_costs.clone(),
        "t13_costs": t13_costs.clone(),
        "t23_costs": t23_costs.clone(),
        "edge_counter": edge_counter.clone()
    }

    dataset = SingleMulticutDataset(data_dict)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = MLPMessagePassing().to(device)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(num_epochs):
        for dicts in data_loader:
            dicts = {k: v.squeeze(0) for k, v in dicts.items()}
            optimizer.zero_grad()
            updated = model(dicts)
            loss = loss_fn(updated)
            print("[PYTHON] Loss:", loss)
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), MODEL_PATH)

    return updated["edge_costs"].detach(), updated["t12_costs"].detach(), updated["t13_costs"].detach(), updated["t23_costs"].detach()