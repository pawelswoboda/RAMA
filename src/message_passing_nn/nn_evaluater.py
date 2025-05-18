import matplotlib.pyplot as plt
from pathlib import Path
import os
import wandb
import numpy as np
import hydra
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
from config import Config

@hydra.main(version_base=None, config_name="config")
def evaluate(cfg: Config):
   # wandb.init(project="rama-learned-mp", name="evaluate_models")

    cpp_lbs, mlp_lbs = [], []
    diff_mlp_cpp, diff_mlp_max = [], []
    instance_names, compare_lines = [], []

    for f in sorted(Path(cfg.data.cpp_dir).glob("*.out")):
        cpp_path = f
        mlp_path = Path(cfg.data.mlp_dir) / f.name
        max_path = Path(cfg.data.max_dir) / f.name

        if not mlp_path.exists():
            print(f"[ERROR] Missing MLP output for {f.name}")
            continue
    
        with open(cpp_path) as f1, open(mlp_path) as f2, open(max_path) as f3:
            cpp_lb = float(f1.readline())
            mlp_lb = float(f2.readline())
            max_lb = float(f3.readline())

        diff_mlp_cpp_lb = 100 * (mlp_lb - cpp_lb) / abs(cpp_lb)
        diff_mlp_max_lb = 100 * (mlp_lb - max_lb) / abs(max_lb)  
        compare_line = f"[COMPARE] {f.name:<15} CPP: {cpp_lb:<15.2f} MLP: {mlp_lb:<15.2f} MAX: {max_lb:<15.2f} %_DIFF_MLP_CPP: {diff_mlp_cpp_lb:<15.2f} %_DIFF_MLP_MAX: {diff_mlp_max_lb:<15.2f}"
        compare_lines.append(compare_line)

        cpp_lbs.append(cpp_lb)
        mlp_lbs.append(mlp_lb)
        diff_mlp_cpp.append(diff_mlp_cpp_lb)
        diff_mlp_max.append(diff_mlp_max_lb)  
        instance_names.append(f.name)

    plot_lower_bounds(instance_names, diff_mlp_cpp, cfg.data.output_plot_path)
    write_summary(diff_mlp_cpp, diff_mlp_max, compare_lines, cfg.data.output_summary_path)
    
    avg_diff_mlp_cpp = np.mean(diff_mlp_cpp)
    avg_diff_mlp_max = np.mean(diff_mlp_max) 
    print("[SUCCESS] EVALUATION FINISHED")
    print("[INFORMATION] AVERAGE DIFFERENCE MLP vs CPP: ", avg_diff_mlp_cpp)
    print("[INFORMATION] AVERAGE DIFFERENCE MLP vs MAX: ", avg_diff_mlp_max)
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

def write_summary(diff_mlp_cpp, diff_mlp_max, compare_lines, output_path):
    count = len(diff_mlp_cpp)
    avg_diff_mlp_cpp = np.mean(diff_mlp_cpp)
    avg_diff_mlp_max = np.mean(diff_mlp_max)  

    summary = []
    summary.append("===== COMPARISON RESULTS =====")
    summary.append(f"{'Instance':<30} {'CPP':<20} {'MLP':<20} {'MAX':<18} {'Diff MLP-CPP (%)':<30} {'Diff MLP-MAX (%)':<20}")
    summary.append("=" * 140)
    summary.extend(compare_lines)
    summary.append("\n===== SUMMARY =====")
    summary.append(f"[SUMMARY] Compared {count} graphs.")
    summary.append(f"[SUMMARY] On average, MLP LB is {abs(avg_diff_mlp_cpp):.2f}% {'better' if avg_diff_mlp_cpp > 0 else 'worse'} than CPP")
    summary.append(f"[SUMMARY] On average, MLP LB is {abs(avg_diff_mlp_max):.2f}% {'better' if avg_diff_mlp_max > 0 else 'worse'} than MAX")

    with open(output_path, "w") as f:
        f.write("\n".join(summary))

if __name__ == "__main__":
    evaluate()
