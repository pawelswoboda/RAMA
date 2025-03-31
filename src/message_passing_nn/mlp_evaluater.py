import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
import os

data_dir = "src/message_passing_nn/data"
cpp_dir = os.path.join(data_dir, "eval/cpp")
mlp_dir = os.path.join(data_dir, "eval/mlp")
output_summary_path = os.path.join(data_dir, "eval/results/summary.txt")
output_plot_path = os.path.join(data_dir, "eval/results/lb_comparison.png")

def plot_lower_bounds(instance_names, cpp_lbs, mlp_lbs, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(instance_names, cpp_lbs, label="CPP Lower Bound", marker='o')
    plt.plot(instance_names, mlp_lbs, label="MLP Lower Bound", marker='x')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Multicut Test Instances")
    plt.ylabel("Lower Bound")
    plt.title("Comparison of Lower Bounds (CPP vs MLP)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(output_path)

def write_summary(count, total_ari, cpp_lbs, mlp_lbs, compare_lines, output_path):
    avg_ari = total_ari / count if count else 0.0
    avg_diff_pct = 0.0
    for cpp, mlp in zip(cpp_lbs, mlp_lbs):
        if cpp != 0:
            diff_pct = 100 * (mlp - cpp) / abs(cpp)
            avg_diff_pct += diff_pct
    avg_diff_pct /= len(cpp_lbs) if cpp_lbs else 1

    summary = []
    summary.append("===== COMPARISON RESULTS =====")
    summary.extend(compare_lines)
    summary.append("\n===== SUMMARY =====")
    summary.append(f"[SUMMARY] Compared {count} graphs.")
    summary.append(f"[SUMMARY] Avg ARI: {avg_ari:.4f}")
    summary.append(f"[SUMMARY] On avg MLP-Lowerbound is {avg_diff_pct:.2f}% {'worse' if avg_diff_pct > 0 else 'better'}")

    with open(output_path, "w") as f:
        f.write("\n".join(summary))

def evaluate():
    matches = 0
    total_ari = 0.0
    count = 0
    cpp_lbs = []
    mlp_lbs = []
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

        cpp_cluster = list(map(int, cpp_lines[0].split()))
        mlp_cluster = list(map(int, mlp_lines[0].split()))
        cpp_lb = float(cpp_lines[1])
        mlp_lb = float(mlp_lines[1])

        ari = adjusted_rand_score(cpp_cluster, mlp_cluster)
        compare_line = f"[COMPARE] {f.name}: ARI={ari:.4f}, LB_CPP={cpp_lb:.2f}, LB_MLP={mlp_lb:.2f}"
        compare_lines.append(compare_line)

        total_ari += ari
        count += 1
        if ari >= 0.8:
            matches += 1

        cpp_lbs.append(cpp_lb)
        mlp_lbs.append(mlp_lb)
        instance_names.append(f.name)

    plot_lower_bounds(instance_names, cpp_lbs, mlp_lbs, output_plot_path)
    write_summary(count, total_ari, cpp_lbs, mlp_lbs, compare_lines, output_summary_path)

if __name__ == "__main__":
    evaluate()
