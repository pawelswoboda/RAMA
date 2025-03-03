import torch
import cupy as cp
import numpy as np

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

    cp_array = cp.ndarray((num_elements,), dtype=np_dtype, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, num_elements * np_dtype().itemsize, None), 0))
    torch_tensor = torch.as_tensor(cp_array).to(device)
  
    return torch_tensor

def nn_update_lagrange(edge_costs_ptr, t1_ptr, t2_ptr, t3_ptr, i_ptr, j_ptr, 
                       t12_costs_ptr, t13_costs_ptr, t23_costs_ptr, 
                       num_edges, num_triangles, num_nodes):
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

    print("[PYTHON] Edge Costs:", edge_costs)
    print("[PYTHON] Triangles:", t1, t2, t3)
    print("[PYTHON] Edges:", i, j)

    updated_t12_costs = t12_costs + 0.1
    updated_t13_costs = t13_costs + 0.1
    updated_t23_costs = t23_costs + 0.1

    updated_edge_costs = edge_costs + 0.1

    print("[PYTHON] Updated Edge Costs:", updated_edge_costs)
    print("[PYTHON] Updated Lagrange Multipliers:")
    print("[PYTHON] t12:", updated_t12_costs)
    print("[PYTHON] t13:", updated_t13_costs)
    print("[PYTHON] t23:", updated_t23_costs)

    return (updated_edge_costs.cpu().numpy(),
            updated_t12_costs.cpu().numpy(),
            updated_t13_costs.cpu().numpy(),
            updated_t23_costs.cpu().numpy())
