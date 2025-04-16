import os
from torch.utils.data import DataLoader
from multicut_dataset import MulticutGraphDataset
import rama_py
import torch
from mlp.mlp_message_passing import MLPMessagePassing
from gnn.gnn_message_passing import GNNMessagePassing
import nn_utils as utils
  
def save_results(name, lb, eval_dir):
    out_path = os.path.join(eval_dir, name.replace(".txt", ".out"))
    with open(out_path, "w") as f:
        f.write(f"{lb:.6f}\n")

def test(model_type="mlp"): # use "mlp" or "gnn"
    #... or "cpp" with DISABLE_MLP=1 /bin/python3 /home/houraghene/RAMA/src/message_passing_nn/nn_tester.py
    utils.set_seed(42)

    data_dir = "src/message_passing_nn/data"
    test_dir = os.path.join(data_dir, "test")
    cpp_dir = os.path.join(data_dir, "eval/cpp")
    mlp_dir = os.path.join(data_dir, "eval/mlp")
    gnn_dir = os.path.join(data_dir, "eval/gnn")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MulticutGraphDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False

    if model_type == "mlp":
        eval_dir = mlp_dir
        model = MLPMessagePassing().to(device)
        model.eval()
    elif model_type == "gnn":
        eval_dir = gnn_dir
        model = GNNMessagePassing().to(device)
        model.eval()
    elif model_type == "cpp":
        eval_dir = cpp_dir
    else:
        print("[ERROR] CANT FIND MODEL")
        return    

    MODEL_PATH = f"./{model_type}_model.pt"
    if model_type != "cpp" and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  

    print(f"[INFO] MODEL: {model_type}")
    print(f"[INFO] DEVICE: {device}")
    print(f"[INFO] Found {len(dataset)} Multicut instances.")
    
    fails = set()
    for sample in loader:
        name = sample["name"][0]    
        try:
            i = sample["i"]
            j = sample["j"]
            costs = sample["costs"]   
            with torch.no_grad():
                if model_type == "cpp":
                    _, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
                elif model_type == "mlp":
                    mp_data = rama_py.get_message_passing_data(i, j, costs, 3)
                       
                    edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = utils.extract_data(mp_data, device)

                    for _ in range(15):
                        updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                            edge_costs, t12_costs, t13_costs, t23_costs,
                            corr_12, corr_13, corr_23, edge_counter
                        )
                        edge_costs, t12_costs, t13_costs, t23_costs = updated_edge_costs, updated_t12, updated_t13, updated_t23

                    lb = utils.lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)
                elif model_type == "gnn":
                    print("TODO")
                else:
                    print("[ERROR] CANT FIND MODEL")
                    return
                
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


# SPÄTER das if else weg machen und einfach dual solver aufgerufen mit/ohne DISABLE_MLP = 1               
