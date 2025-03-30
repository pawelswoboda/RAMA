import torch
import cupy as cp
import numpy as np
from mp.dbca_message_passing import ClassicalMessagePassing
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
    torch_tensor = torch.as_tensor(cp_array).to(device)
    return torch_tensor


def nn_update(edge_costs_ptr, t1_ptr, t2_ptr, t3_ptr, i_ptr, j_ptr, 
                       t12_costs_ptr, t13_costs_ptr, t23_costs_ptr,
                       triangle_corr_12_ptr, triangle_corr_13_ptr, triangle_corr_23_ptr,
                       edge_counter_ptr, num_edges, num_triangles, num_nodes):
        

    t1 = ptr_to_tensor(t1_ptr, num_triangles, torch.int32)
    t2 = ptr_to_tensor(t2_ptr, num_triangles, torch.int32)
    t3 = ptr_to_tensor(t3_ptr, num_triangles, torch.int32)

    i = ptr_to_tensor(i_ptr, num_nodes, torch.int32)
    j = ptr_to_tensor(j_ptr, num_nodes, torch.int32)

    tri_corr_12 = ptr_to_tensor(triangle_corr_12_ptr, num_triangles, torch.int32).long()
    tri_corr_13 = ptr_to_tensor(triangle_corr_13_ptr, num_triangles, torch.int32).long()
    tri_corr_23 = ptr_to_tensor(triangle_corr_23_ptr, num_triangles, torch.int32).long()

    edge_counter = ptr_to_tensor(edge_counter_ptr, num_edges, torch.int32)

    edge_costs = ptr_to_tensor(edge_costs_ptr, num_edges, torch.float32)
    t12_costs = ptr_to_tensor(t12_costs_ptr, num_triangles, torch.float32)
    t13_costs = ptr_to_tensor(t13_costs_ptr, num_triangles, torch.float32)
    t23_costs = ptr_to_tensor(t23_costs_ptr, num_triangles, torch.float32)

    updated_edge_costs, updated_t12_costs, updated_t13_costs, updated_t23_costs = via_mlp(
        edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
        t12_costs, t13_costs, t23_costs,edge_counter
    )
      
    return (
        updated_edge_costs,
        updated_t12_costs,
        updated_t13_costs,
        updated_t23_costs
    )

def via_dbca(edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
             t12_costs, t13_costs, t23_costs, edge_counter):

    mp = ClassicalMessagePassing(edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
                                   t12_costs, t13_costs, t23_costs, edge_counter)
    mp.iteration()  

  #  print("[PYTHON] Lower bound:", mp.compute_lower_bound())

    return mp.edge_costs.cpu().numpy(), mp.t12_costs.cpu().numpy(), mp.t13_costs.cpu().numpy(), mp.t23_costs.cpu().numpy()


def via_mlp(edge_costs, tri_corr_12, tri_corr_13, tri_corr_23,
            t12_costs, t13_costs, t23_costs, edge_counter, train=True):
    
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

    def loss_fn(edge_costs, t12, t13, t23):
        return -lower_bound(edge_costs, t12, t13, t23) / edge_costs.numel()

    device = edge_costs.device
    model = MLPMessagePassing().to(device)

    MODEL_PATH = "./mlp_model.pt"
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
            edge_costs, t12_costs, t13_costs, t23_costs,
            tri_corr_12, tri_corr_13, tri_corr_23, edge_counter
        )

        loss = loss_fn(updated_edge_costs, updated_t12, updated_t13, updated_t23)
        loss.backward()
        print(f"Loss: {loss.item():.6f}")
        #print(f"lb: {lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)}")

        for name, param in model.named_parameters():
            if param.grad is not None:
               # print(name, "grad norm", param.grad.norm().item())
               x = 0
            else:
                print(name, "grad is None")

        optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)

    else:
        model.eval()
        with torch.no_grad():
            updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                edge_costs, t12_costs, t13_costs, t23_costs,
                tri_corr_12, tri_corr_13, tri_corr_23, edge_counter
            )

    return (
        updated_edge_costs.detach().cpu().numpy(),
        updated_t12.detach().cpu().numpy(),
        updated_t13.detach().cpu().numpy(),
        updated_t23.detach().cpu().numpy()
    )
