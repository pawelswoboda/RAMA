import os
from torch.utils.data import DataLoader
from multicut_dataset import MulticutGraphDataset
import rama_py
import torch
from mp.mlp_message_passing import MLPMessagePassing

# DISABLE_MLP=1 /bin/python3 /home/houraghene/RAMA/src/message_passing_nn/mlp_tester.py  

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

def save_results(name, lb, eval_dir):
    out_path = os.path.join(eval_dir, name.replace(".txt", ".out"))
    with open(out_path, "w") as f:
        f.write(f"{lb:.6f}\n")


def test(mlp=False):
    
    data_dir = "src/message_passing_nn/data"
    test_dir = os.path.join(data_dir, "test")
    cpp_dir = os.path.join(data_dir, "eval/cpp")
    mlp_dir = os.path.join(data_dir, "eval/mlp")
    eval_dir = mlp_dir if mlp else cpp_dir

    dataset = MulticutGraphDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False

    device = "cuda" if torch.cuda.is_available else "cpu"
    model = MLPMessagePassing().to(device)
    model.eval()

    MODEL_PATH = "./mlp_model.pt"
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    print(f"[INFO] Found {len(dataset)} Multicut instances.")
    
    fails = set()
    for sample in loader:
        name = sample["name"][0]    
        try:
            i = sample["i"]
            j = sample["j"]
            costs = sample["costs"]   
            with torch.no_grad():
                if mlp:
                        mp_data = rama_py.get_message_passing_data(i, j, costs, 3)
                       
                        edge_costs = torch.tensor(mp_data["edge_costs"], dtype=torch.float32).to(device)
                        t12_costs = torch.tensor(mp_data["t12_costs"], dtype=torch.float32).to(device)
                        t13_costs = torch.tensor(mp_data["t13_costs"], dtype=torch.float32).to(device)
                        t23_costs = torch.tensor(mp_data["t23_costs"], dtype=torch.float32).to(device)
                        corr_12 = torch.tensor(mp_data["tri_corr_12"], dtype=torch.long).to(device)
                        corr_13 = torch.tensor(mp_data["tri_corr_13"], dtype=torch.long).to(device)
                        corr_23 = torch.tensor(mp_data["tri_corr_23"], dtype=torch.long).to(device)
                        edge_counter = torch.tensor(mp_data["edge_counter"], dtype=torch.int32).to(device)
                        
         # SPÄTER das if else weg machen und einfach dual solver aufgerufen mit/ohne DISABLE_MLP = 1               
                        for k in range(10):
                            updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                                edge_costs, t12_costs, t13_costs, t23_costs,
                                corr_12, corr_13, corr_23, edge_counter
                            )
                            edge_costs, t12_costs, t13_costs, t23_costs = updated_edge_costs, updated_t12, updated_t13, updated_t23

                        lb = lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)
                else:
                    _, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)

            save_results(name, lb, eval_dir)
            print(f"[SUCCESS] {name}") #: Clusters: {mapping}, LB: {lb}")
        except Exception as e:
            print(f"[ERROR] Failed on {name}: {e}")
            fails.add(name)

    if fails:
        n = len(fails)
        print(f"[SUMMARY] {n} Failed instances:")
        for f in sorted(fails):
            print(f" - {f}")
    
    print("Testing finished.")

if __name__ == "__main__":
    test()
