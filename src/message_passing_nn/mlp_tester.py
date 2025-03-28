import os
from torch.utils.data import DataLoader
from multicut_dataset import MulticutGraphDataset
import rama_py

mlp = False # DISABLE_MLP=1 /bin/python3 /home/houraghene/RAMA/src/message_passing_nn/mlp_tester.py  
# # dont forget to put train = false in nn_message_passing.py

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


def test():
    dataset = MulticutGraphDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False
    fails = set()
    print(f"[INFO] Found {len(dataset)} Multicut instances.")

    for sample in loader:
        try:
            i = sample["i"]
            j = sample["j"]
            costs = sample["costs"]   
            name = sample["name"][0]    
            mapping, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
            save_results(name, mapping, lb)
            print(f"[SUCCESS] {name}") #: Clusters: {mapping}, LB: {lb}")
        except Exception as e:
            print(f"[ERROR] Failed on {name}: {e}")
            fails.add(name)

    if fails:
        n = len(fails)
        print(f"[SUMMARY] {n} Failed instances:")
        for f in sorted(fails):
            print(f" - {f}")
    
    print("Testing finished.")

if __name__ == "__main__":
    test()
