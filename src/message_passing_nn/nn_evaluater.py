import matplotlib.pyplot as plt
from pathlib import Path
import os
import wandb
import numpy as np
import hydra
import sys
import rama_py
import nn_utils as utils
from torch.utils.data import DataLoader
from multicut_dataset import MulticutGraphDataset
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
from config import Config

def compute_optimal_lb(i, j, costs):
    opts = rama_py.multicut_solver_options("D")  
    opts.verbose = False 
    opts.num_dual_itr_lb = 100000000
    opts.max_cycle_length_lb = 3  
    opts.num_outer_itr_dual = 1
    
    _, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
    
    return lb

@hydra.main(version_base=None, config_name="config")
def evaluate(cfg: Config):
   # wandb.init(project="rama-learned-mp", name="evaluate_models")

    cpp_lbs, mlp_lbs, max_lbs = [], [], []
    diff_mlp_cpp, diff_mlp_max, diff_cpp_max = [], [], []
    instance_names, compare_lines = [], []
    
    dataset = MulticutGraphDataset(cfg.data.test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    sample_data = {}
    for sample in loader:
        name = sample["name"][0]
        sample_data[name] = {
            'i': sample["i"],
            'j': sample["j"], 
            'costs': sample["costs"]
        }

    success_count = 0
    total_count = 0

    for f in sorted(Path(cfg.data.cpp_dir).glob("*.out")):
        cpp_path = f
        mlp_path = Path(cfg.data.mlp_dir) / f.name
        
        total_count += 1

        if not mlp_path.exists():
            print(f"[ERROR] Missing MLP output for {f.name}")
            continue
            
        instance_name = f.name.replace(".out", ".txt")
        if instance_name not in sample_data:
            print(f"[ERROR] Missing sample data for {instance_name}")
            continue
    
        try:
            with open(cpp_path) as f1, open(mlp_path) as f2:
                cpp_lb = float(f1.readline())
                mlp_lb = float(f2.readline())

            sample = sample_data[instance_name]
            i = sample['i']
            j = sample['j']
            costs = sample['costs']
            normed_costs, factor = utils.normalise_costs(costs)
            max_lb = compute_optimal_lb(i, j, normed_costs.tolist())

            diff_mlp_cpp_lb = 100 * (mlp_lb - cpp_lb) / abs(cpp_lb) if cpp_lb != 0 else 0
            diff_mlp_max_lb = 100 * (mlp_lb - max_lb) / abs(max_lb) if max_lb != 0 else 0
            diff_cpp_max_lb = 100 * (cpp_lb - max_lb) / abs(max_lb) if max_lb != 0 else 0
            
            compare_line = f"[COMPARE] {f.name:<15} CPP: {cpp_lb:<15.6f} MLP: {mlp_lb:<15.6f} MAX: {max_lb:<15.6f} %_DIFF_MLP_CPP: {diff_mlp_cpp_lb:<8.2f} %_DIFF_MLP_MAX: {diff_mlp_max_lb:<8.2f} %_DIFF_CPP_MAX: {diff_cpp_max_lb:<8.2f}"
            compare_lines.append(compare_line)
            
            cpp_lbs.append(cpp_lb)
            mlp_lbs.append(mlp_lb)
            max_lbs.append(max_lb)
            diff_mlp_cpp.append(diff_mlp_cpp_lb)
            diff_mlp_max.append(diff_mlp_max_lb)
            diff_cpp_max.append(diff_cpp_max_lb)
            instance_names.append(f.name)
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] Error processing {f.name}: {e}")

    plot_lower_bounds(instance_names, diff_mlp_cpp, cfg.data.output_plot_path)
    write_summary(diff_mlp_cpp, diff_mlp_max, diff_cpp_max, compare_lines, cfg.data.output_summary_path)
        
    print("[SUCCESS] EVALUATION FINISHED")
    print(f"[SUMMARY] {success_count}/{total_count} instances processed successfully")
   
    avg_diff_mlp_cpp = np.mean(diff_mlp_cpp)
    return avg_diff_mlp_cpp  

def plot_lower_bounds(instance_names, diff_mlp_cpp, output_path):
    plt.figure(figsize=(14, 6))
    plt.plot(instance_names, diff_mlp_cpp, label="MLP - CPP", marker='x')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Multicut Test Instances")
    plt.ylabel("Lower Bound Difference (%)")
    plt.title("Lower Bound Difference (MLP - CPP)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(output_path)

def write_summary(diff_mlp_cpp, diff_mlp_max, diff_cpp_max, compare_lines, output_path):
    count = len(diff_mlp_cpp)
    avg_diff_mlp_cpp = np.mean(diff_mlp_cpp)
    avg_diff_mlp_max = np.mean(diff_mlp_max)
    avg_diff_cpp_max = np.mean(diff_cpp_max)

    summary = []
    summary.append("===== COMPARISON RESULTS =====")
    summary.append(f"{'Instance':<30} {'CPP':<20} {'MLP':<20} {'MAX':<20} {'MLP-CPP(%)':<23} {'MLP-MAX(%)':<24} {'CPP-MAX(%)':<20}")
    summary.append("=" * 160)
    summary.extend(compare_lines)
    summary.append("\n===== SUMMARY =====")
    summary.append(f"[SUMMARY] Compared {count} graphs.")
    summary.append(f"[SUMMARY] On average, MLP LB is {abs(avg_diff_mlp_cpp):.2f}% {'better' if avg_diff_mlp_cpp > 0 else 'worse'} than CPP")
    summary.append(f"[SUMMARY] On average, MLP LB is {abs(avg_diff_mlp_max):.2f}% {'better' if avg_diff_mlp_max > 0 else 'worse'} than MAX")
    summary.append(f"[SUMMARY] On average, CPP LB is {abs(avg_diff_cpp_max):.2f}% {'better' if avg_diff_cpp_max > 0 else 'worse'} than MAX")
    
    summary.append(f"\n===== DETAILED STATISTICS =====")
    summary.append(f"MLP vs CPP - Mean: {avg_diff_mlp_cpp:.2f}%, Std: {np.std(diff_mlp_cpp):.2f}%, Min: {np.min(diff_mlp_cpp):.2f}%, Max: {np.max(diff_mlp_cpp):.2f}%")
    summary.append(f"MLP vs MAX - Mean: {avg_diff_mlp_max:.2f}%, Std: {np.std(diff_mlp_max):.2f}%, Min: {np.min(diff_mlp_max):.2f}%, Max: {np.max(diff_mlp_max):.2f}%")
    summary.append(f"CPP vs MAX - Mean: {avg_diff_cpp_max:.2f}%, Std: {np.std(diff_cpp_max):.2f}%, Min: {np.min(diff_cpp_max):.2f}%, Max: {np.max(diff_cpp_max):.2f}%")

    with open(output_path, "w") as f:
        f.write("\n".join(summary))

if __name__ == "__main__":
    evaluate()
