import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from multicut_dataset import MulticutGraphDataset
import rama_py

mlp = True  
data_dir = "src/message_passing_nn/data"
test_dir = os.path.join(data_dir, "test")
cpp_dir = os.path.join(data_dir, "eval/cpp")
mlp_dir = os.path.join(data_dir, "eval/mlp")
eval_dir = mlp_dir if mlp else cpp_dir

def save_results(name, mapping, lb):
    out_path = os.path.join(eval_dir, name.replace(".txt", ".out"))
    with open(out_path, "w") as f:
        f.write(" ".join(map(str, mapping)) + "\n")
        f.write(f"{lb:.6f}\n")


def run_tests():
    dataset = MulticutGraphDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    opts = rama_py.multicut_solver_options("PD")

    print(f"[INFO] Found {len(dataset)} Multicut instances.")

    for sample in loader:
        try:
            i = sample["i"]
            j = sample["j"]
            costs = sample["costs"]
            name = sample["name"][0]
            mapping, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
            save_results(name, mapping, lb)
            print(f"[SUCCESS] {name}: Clusters: {mapping}, LB: {lb}")
        except Exception as e:
            print(f"[ERROR] Failed on {name}: {e}")


def compare_mlp_cpp_results():
    mismatches = 0
    total_ari = 0.0
    count = 0

    for f in sorted(Path(cpp_dir).glob("*.out")):
        cpp_path = f
        mlp_path = Path(mlp_dir) / f.name

        if not mlp_path.exists():
            print(f"[ERROR] Missing MLP output for {f.name}")
            return

        with open(cpp_path) as f1, open(mlp_path) as f2:
            cpp_lines = f1.readlines()
            mlp_lines = f2.readlines()

        cpp_mapping = list(map(int, cpp_lines[0].split()))
        mlp_mapping = list(map(int, mlp_lines[0].split()))
        cpp_lb = float(cpp_lines[1])
        mlp_lb = float(mlp_lines[1])

        ari = adjusted_rand_score(cpp_mapping, mlp_mapping)
        print(f"[COMPARE] {f.name}: ARI={ari:.4f}, LB_CPP={cpp_lb:.2f}, LB_MLP={mlp_lb:.2f}")
        total_ari += ari
        count += 1
        if ari < 0.9999:
            mismatches += 1


    print(f"\n[SUMMARY] Compared {count} graphs.")
    print(f"[SUMMARY] Avg ARI: {total_ari / count:.4f}")
    print(f"[SUMMARY] Mismatches (ARI < 1.0): {mismatches}/{count}")


if __name__ == "__main__":
    run_tests()
    if mlp:
        compare_mlp_cpp_results()
