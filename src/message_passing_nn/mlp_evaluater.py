import matplotlib.pyplot as plt
from pathlib import Path
import os
import wandb
import numpy as np

data_dir = "src/message_passing_nn/data"
cpp_dir = os.path.join(data_dir, "eval/cpp")
mlp_dir = os.path.join(data_dir, "eval/mlp")
output_summary_path = os.path.join(data_dir, "eval/results/summary.txt")
output_plot_path = os.path.join(data_dir, "eval/results/lb_comparison.png")

def plot_lower_bounds(instance_names, cpp_lbs, mlp_lbs, diff_lb, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(instance_names, diff_lb, marker='x')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Multicut Test Instances")
    plt.ylabel("Lower Bound Difference in %")
    plt.title("Comparison of Lower Bounds Differences (MLP LB - CPP LB) in %")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(output_path)

def write_summary(diff_lbs, compare_lines, output_path):
    count = len(diff_lbs)
    avg_pct_diff = np.mean(diff_lbs)

    summary = []
    summary.append("===== COMPARISON RESULTS =====")
    summary.extend(compare_lines)
    summary.append("\n===== SUMMARY =====")
    summary.append(f"[SUMMARY] Compared {count} graphs.")
    summary.append(f"[SUMMARY] On average, MLP lower bound is {abs(avg_pct_diff):.2f}% {'better' if avg_pct_diff > 0 else 'worse'}")

    with open(output_path, "w") as f:
        f.write("\n".join(summary))


def evaluate():
    wandb.init(project="rama-mlp", name="evaluate_v2")

    cpp_lbs = []
    mlp_lbs = []
    diff_lbs = []
    instance_names = []
    compare_lines = []

    for f in sorted(Path(cpp_dir).glob("*.out")):
        cpp_path = f
        mlp_path = Path(mlp_dir) / f.name

        if not mlp_path.exists():
            print(f"[ERROR] Missing MLP output for {f.name}")
            continue

        with open(cpp_path) as f1, open(mlp_path) as f2:
            cpp_lines = f1.readlines()
            mlp_lines = f2.readlines()

        cpp_lb = float(cpp_lines[0])
        mlp_lb = float(mlp_lines[0])
        diff_lb = 100 * (mlp_lb - cpp_lb) / abs(cpp_lb)
        wandb.log({f"lower bound difference (mlp - cpp) in %": diff_lb})

        compare_line = f"[COMPARE] {f.name}: LB_CPP={cpp_lb:.2f}, LB_MLP={mlp_lb:.2f}, DIFF_in_%={diff_lb}"
        compare_lines.append(compare_line)

        cpp_lbs.append(cpp_lb)
        mlp_lbs.append(mlp_lb)
        diff_lbs.append(diff_lb)

        instance_names.append(f.name)

    plot_lower_bounds(instance_names, cpp_lbs, mlp_lbs, diff_lbs, output_plot_path)
    write_summary(diff_lbs, compare_lines, output_summary_path)

if __name__ == "__main__":
    evaluate()
