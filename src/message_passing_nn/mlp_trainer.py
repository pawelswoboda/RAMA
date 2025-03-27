from torch.utils.data import DataLoader
import rama_py
from multicut_dataset import MulticutGraphDataset

def train():
    dataset = MulticutGraphDataset("src/message_passing_nn/data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  
    opts = rama_py.multicut_solver_options("PD")

    print(f"[INFO] Found {len(dataset)} Multicut instances.")

    for epoch in range(1):
        for sample in loader:
            try:
                i = sample["i"]
                j = sample["j"]
                costs = sample["costs"]
                name = sample["name"]
                mapping, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
                print(f"[SUCCESS] {name}: Clusters: {mapping}, LB: {lb}")
            except Exception as e:
                print(f"[ERROR] Failed on file {name}: {e}")

    print("Training finished.")

if __name__ == "__main__":
    train()
