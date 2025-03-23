from torch.utils.data import Dataset
from pathlib import Path
import rama_py
from torch.utils.data import DataLoader

class MulticutGraphDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("*.txt"))
        assert len(self.files) > 0, "No files found in dataset directory!"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        with open(filepath, "r") as f:
            lines = f.readlines()

        assert lines[0].strip() == "MULTICUT"
        i, j, costs = [], [], []
        for line in lines[1:]:
            u, v, c = line.strip().split()
            i.append(int(u))
            j.append(int(v))
            costs.append(float(c))

        return {
            "i": i,
            "j": j,
            "costs": costs,
            "name": filepath.name
        }

def train_all_graphs_with_loader():
   
    dataset = MulticutGraphDataset("src/message_passing_nn/data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  
    opts = rama_py.multicut_solver_options("PD")

    print(f"[INFO] Found {len(dataset)} Multicut instances.")

    for epoch in range(5):
        for sample in loader:
            i = sample["i"]
            j = sample["j"]
            costs = sample["costs"]
            name = sample["name"]
            res = rama_py.rama_cuda(i, j, costs, opts)
            print(res)

    print("Training finished.")

if __name__ == "__main__":
    train_all_graphs_with_loader()
