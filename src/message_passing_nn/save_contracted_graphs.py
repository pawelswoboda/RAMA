from pathlib import Path
import rama_py
from multicut_dataset import MulticutGraphDataset
from torch.utils.data import DataLoader

def main():
    train_dir = Path("src/message_passing_nn/data/train")  
    
    opts = rama_py.multicut_solver_options("PD")  
    opts.verbose = False
    
    dataset = MulticutGraphDataset(train_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for graph in loader:
        name = graph["name"][0]
        i = graph["i"]
        j = graph["j"]
        costs = graph["costs"]
        print(f'Processing {name}...')
        rama_py.rama_cuda(i, j, costs, opts)

if __name__ == '__main__':
    main() 