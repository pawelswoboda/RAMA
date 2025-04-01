import matplotlib.pyplot as plt
from pathlib import Path
import os

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
    plt.ylabel("Lower Bound Difference")
    plt.title("Comparison of Lower Bounds Differences (MLP LB - CPP LB)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(output_path)

def write_summary(cpp_lbs, mlp_lbs, compare_lines, output_path):
    total_pct_diff = 0.0
    total_abs_diff = 0.0
    count = len(cpp_lbs)

    for cpp, mlp in zip(cpp_lbs, mlp_lbs):
        if cpp != 0:
            diff_pct = 100 * (abs(cpp) - abs(mlp)) / abs(cpp)
            total_pct_diff += diff_pct
            total_abs_diff += (mlp - cpp)

    avg_pct_diff = total_pct_diff / count if count > 0 else 0

    summary = []
    summary.append("===== COMPARISON RESULTS =====")
    summary.extend(compare_lines)
    summary.append("\n===== SUMMARY =====")
    summary.append(f"[SUMMARY] Compared {count} graphs.")
    summary.append(f"[SUMMARY] On average, MLP lower bound is {abs(avg_pct_diff):.2f}% {'better' if avg_pct_diff > 0 else 'worse'}")

    with open(output_path, "w") as f:
        f.write("\n".join(summary))


def evaluate():
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
        diff_lb = float(mlp_lb-cpp_lb)

        compare_line = f"[COMPARE] {f.name}: LB_CPP={cpp_lb:.2f}, LB_MLP={mlp_lb:.2f}, DIFF={diff_lb}"
        compare_lines.append(compare_line)

        cpp_lbs.append(cpp_lb)
        mlp_lbs.append(mlp_lb)
        diff_lbs.append(diff_lb)

        instance_names.append(f.name)

    plot_lower_bounds(instance_names, cpp_lbs, mlp_lbs, diff_lbs, output_plot_path)
    write_summary(cpp_lbs, mlp_lbs, compare_lines, output_summary_path)

if __name__ == "__main__":
    evaluate()
