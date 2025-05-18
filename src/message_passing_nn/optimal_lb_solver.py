import os
import sys
from pathlib import Path
import rama_py
import nn_utils as utils
from torch.utils.data import DataLoader
from multicut_dataset import MulticutGraphDataset
import hydra
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
from config import Config

def save_results(name, lb, eval_dir):
    out_path = os.path.join(eval_dir, name.replace(".txt", ".out"))
    with open(out_path, "w") as f:
        f.write(f"{lb:.6f}\n")

def compute_optimal_lb(i, j, costs):

    opts = rama_py.multicut_solver_options("D")  
    opts.verbose = False 
    opts.num_dual_itr_lb = 1000000
    opts.max_cycle_length_lb = 3  
    opts.num_outer_itr_dual = 1
    
    _, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
    
    return lb

@hydra.main(version_base=None, config_name="config")
def solve_optimal_lbs(cfg: Config):
    utils.set_seed(cfg.seed)  
    
    max_dir = cfg.data.max_dir
    
    dataset = MulticutGraphDataset(cfg.data.test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"[INFO] Gefunden: {len(dataset)} Multicut Instanzen")
    success_count = 0
    
    for sample in loader:
        name = sample["name"][0]
        try:            
            i = sample["i"]
            j = sample["j"]
            costs = sample["costs"]
            normed_costs, factor = utils.normalise_costs(costs) 

            max_lb = compute_optimal_lb(i, j, normed_costs.tolist())
            
            save_results(name, max_lb, max_dir)
            
            print(f"[SUCCESS] {name}: MAX LB = {max_lb:.6f}")
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] ERROR {name}: {e}")
    
    print(f"[SUMMARY] {success_count}/{len(dataset)} SUCCESSFUL")

if __name__ == "__main__":
    solve_optimal_lbs() 

# DISABLE_MLP=1 /usr/bin/python3 /work/houraghene/RAMA/src/message_passing_nn/optimal_lb_solver.py