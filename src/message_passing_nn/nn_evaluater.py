import matplotlib.pyplot as plt
from pathlib import Path
import os
import wandb
import numpy as np

data_dir = "src/message_passing_nn/data"
cpp_dir = os.path.join(data_dir, "eval/cpp")
mlp_dir = os.path.join(data_dir, "eval/mlp")
gnn_dir = os.path.join(data_dir, "eval/gnn")
output_summary_path = os.path.join(data_dir, "eval/results/summary.txt")
output_plot_path = os.path.join(data_dir, "eval/results/lb_comparison.png")

def plot_lower_bounds(instance_names, diff_mlp, diff_gnn, output_path):
    plt.figure(figsize=(14, 6))
    plt.plot(instance_names, diff_mlp, label="MLP - CPP", marker='x')
    plt.plot(instance_names, diff_gnn, label="GNN - CPP", marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Multicut Test Instances")
    plt.ylabel("Lower Bound Difference (%)")
    plt.title("Lower Bound Difference to CPP (MLP & GNN)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(output_path)

def write_summary(diff_mlp, diff_gnn, compare_lines, output_path):
    count = len(diff_mlp)
    avg_diff_mlp = np.mean(diff_mlp)
    avg_diff_gnn = np.mean(diff_gnn)

    summary = []
    summary.append("===== COMPARISON RESULTS =====")
    summary.extend(compare_lines)
    summary.append("\n===== SUMMARY =====")
    summary.append(f"[SUMMARY] Compared {count} graphs.")
    summary.append(f"[SUMMARY] On average, MLP LB is {abs(avg_diff_mlp):.2f}% {'better' if avg_diff_mlp > 0 else 'worse'} than CPP")
    summary.append(f"[SUMMARY] On average, GNN LB is {abs(avg_diff_gnn):.2f}% {'better' if avg_diff_gnn > 0 else 'worse'} than CPP")

    with open(output_path, "w") as f:
        f.write("\n".join(summary))


def evaluate():
    wandb.init(project="rama-learned-mp", name="evaluate_models_v2")

    cpp_lbs, mlp_lbs, gnn_lbs = [], [], []
    diff_mlp, diff_gnn = [], []
    instance_names, compare_lines = [], []

    for f in sorted(Path(cpp_dir).glob("*.out")):
        cpp_path = f
        mlp_path = Path(mlp_dir) / f.name
        gnn_path = Path(gnn_dir) / f.name

        if not mlp_path.exists(): #or not gnn_path.exists():
            print(f"[ERROR] Missing MLP or GNN output for {f.name}")
            continue
    
        with open(cpp_path) as f1, open(mlp_path) as f2: #, open(gnn_path) as f3:
            cpp_lb = float(f1.readline())
            mlp_lb = float(f2.readline())
            #gnn_lb = float(f3.readline())

        diff_mlp_lb = 100 * (mlp_lb - cpp_lb) / abs(cpp_lb)
        #diff_gnn_lb = 100 * (gnn_lb - cpp_lb) / abs(cpp_lb)

        wandb.log({
            "MLP vs CPP (%)": diff_mlp_lb #,
           # "GNN vs CPP (%)": diff_gnn_lb
        })

        compare_line = f"[COMPARE] {f.name}: CPP={cpp_lb:.2f}, MLP={mlp_lb:.2f}, GNN={0:.2f}, DIFF_MLP={diff_mlp_lb:.2f}%, DIFF_GNN={0:.2f}%"
        compare_lines.append(compare_line)

        cpp_lbs.append(cpp_lb)
        mlp_lbs.append(mlp_lb)
        gnn_lbs.append(0) #gnn_lb
        diff_mlp.append(diff_mlp_lb)
        diff_gnn.append(0) #gnn_lb
        instance_names.append(f.name)

    plot_lower_bounds(instance_names, diff_mlp, diff_gnn, output_plot_path)
    write_summary(diff_mlp, diff_gnn, compare_lines, output_summary_path)
    print("[SUCCESS] EVALUATION FINISHED")

if __name__ == "__main__":
    evaluate()
