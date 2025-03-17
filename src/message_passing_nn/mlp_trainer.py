import rama_py
from pathlib import Path

DATA_DIR = Path("src/message_passing_nn/data")

opts = rama_py.multicut_solver_options("PD")

def load_graph_from_txt(filepath):

    with open(filepath, "r") as f:
        lines = f.readlines()

    assert lines[0].strip() == "MULTICUT"
    i, j, costs = [], [], []
    for line in lines[1:]:
        u, v, c = line.strip().split()
        i.append(int(u))
        j.append(int(v))
        costs.append(float(c))

    return i, j, costs

def train_all_graphs():
    all_files = list(DATA_DIR.glob("*.txt"))
    print(f"[INFO] Found {len(all_files)} Multicut instances.")

    for path in all_files:
        i, j, costs = load_graph_from_txt(path)
        rama_py.rama_cuda(i, j, costs, opts)
        

    print("Training finished.")


if __name__ == "__main__":
    train_all_graphs()
