import sys
import os
from torch.utils.data import DataLoader
from multicut_dataset import MulticutGraphDataset
import rama_py
import torch
from mlp.mlp_message_passing import MLPMessagePassing
from gnn.gnn_message_passing import GNNMessagePassing
from dbca.dbca_message_passing import ClassicalMessagePassing
import nn_utils as utils
import hydra
from configuration.config import Config

def save_results(name, lb, eval_dir):
    out_path = os.path.join(eval_dir, name.replace(".txt", ".out"))
    with open(out_path, "w") as f:
        f.write(f"{lb:.6f}\n")

@hydra.main(version_base=None, config_name="config")
def test(cfg: Config):
    utils.set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = MulticutGraphDataset(cfg.data.test_dir)
    loader = DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False)

    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False

    if cfg.test.model_type == "mlp":
        eval_dir = cfg.data.mlp_dir
        model = MLPMessagePassing(cfg.model).to(device)
        model.eval()
        
    elif cfg.test.model_type == "gnn":
        eval_dir = cfg.data.gnn_dir
        model = GNNMessagePassing().to(device)
        model.eval()
        
    elif cfg.test.model_type == "cpp":
        eval_dir = cfg.data.cpp_dir
        
    else:
        print("[ERROR] CANT FIND MODEL, USE mlp, gnn OR cpp")
        return    

    MODEL_PATH = f"./{cfg.test.model_type}_model.pt"
    if cfg.test.model_type != "cpp" and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  

    print(f"[INFO] MODEL: {cfg.test.model_type}")
    print(f"[INFO] DEVICE: {device}")
    print(f"[INFO] Found {len(dataset)} Multicut instances.")
    
    fails = set()
    for sample in loader:
        name = sample["name"][0]    
        try:
            i = sample["i"]
            j = sample["j"]
            costs = sample["costs"] 
            normed_costs, factor = utils.normalise_costs(costs) 
            mp_data = rama_py.get_message_passing_data(i, j, normed_costs.tolist(), 3)
            edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = utils.extract_data(mp_data, device)

            with torch.no_grad():
                if cfg.test.model_type == "cpp":                    
                    #_, lb, _, _ = rama_py.rama_cuda(i, j, normed_costs.tolist(), opts)
                    mp = ClassicalMessagePassing(edge_costs, corr_12, corr_13, corr_23,
                                   t12_costs, t13_costs, t23_costs, edge_counter)
                    for _ in range(cfg.test.num_mp_iter):
                        mp.iteration()  
                    lb = mp.compute_lower_bound()
                elif cfg.test.model_type == "mlp":
                    #mp_data = rama_py.get_message_passing_data(i, j, normed_costs.tolist(), 3)
                    #edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = utils.extract_data(mp_data, device)
                
                    for _ in range(cfg.test.num_mp_iter):
                        updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                            edge_costs, t12_costs, t13_costs, t23_costs,
                            corr_12, corr_13, corr_23, edge_counter, dist=cfg.test.dist
                        )
                        edge_costs, t12_costs, t13_costs, t23_costs = updated_edge_costs, updated_t12, updated_t13, updated_t23

                    lb = utils.lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)
                    
                elif cfg.test.model_type == "gnn":
                    print("TODO")
                    
                else:
                    print("[ERROR] CANT FIND MODEL")
                    return
                
            save_results(name, lb, eval_dir)
            print(f"[SUCCESS] {name}")
        except Exception as e:
            print(f"[ERROR] Failed on {name}: {e}")
            fails.add(name)

    if fails:
        n = len(fails)
        print(f"[SUMMARY] {n} Failed instances:")
        for f in sorted(fails):
            print(f" - {f}")
    
    print("TESTING FINISHED")

if __name__ == "__main__":
    test()

