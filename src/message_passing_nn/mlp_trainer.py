from torch.utils.data import DataLoader
import rama_py
from multicut_dataset import MulticutGraphDataset

# DISABLE_MLP=1 /bin/python3 /home/houraghene/RAMA/src/message_passing_nn/mlp_trainer.py 
# dont forget to put train = true in nn_message_passing.py
def train():
    dataset = MulticutGraphDataset("src/message_passing_nn/data2/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  
    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False
    print(f"[INFO] Found {len(dataset)} Multicut instances.")
    fails = set()
    for epoch in range(1):
        for sample in loader:
            name = sample["name"][0]
            print(f"[LOADING] {name}...")
            try:
                i = sample["i"]
                j = sample["j"]
                costs = sample["costs"]
                mapping, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
                print(f"[SUCCESS] {name}: Clusters: {mapping}, LB: {lb}")
                return
            except Exception as e:
                print(f"[ERROR] Failed on file {name}: {e}")
                fails.add(name)

    if fails:
        n = len(fails)
        print(f"[SUMMARY] {n} Failed instances:")
        for f in sorted(fails):
            print(f" - {f}")

    print("Training finished.")

if __name__ == "__main__":
    train()
