from torch.utils.data import Dataset
from pathlib import Path

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