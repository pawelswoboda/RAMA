from torch.utils.data import DataLoader
import rama_py
from multicut_dataset import MulticutGraphDataset
from mp.mlp_message_passing import MLPMessagePassing
import os
import torch
import wandb

def set_seed(seed=42):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_data(mp_data, device):
    edge_costs = torch.tensor(mp_data["edge_costs"], dtype=torch.float32).to(device)
    t12_costs = torch.tensor(mp_data["t12_costs"], dtype=torch.float32).to(device)
    t13_costs = torch.tensor(mp_data["t13_costs"], dtype=torch.float32).to(device)
    t23_costs = torch.tensor(mp_data["t23_costs"], dtype=torch.float32).to(device)
    corr_12 = torch.tensor(mp_data["tri_corr_12"], dtype=torch.long).to(device)
    corr_13 = torch.tensor(mp_data["tri_corr_13"], dtype=torch.long).to(device)
    corr_23 = torch.tensor(mp_data["tri_corr_23"], dtype=torch.long).to(device)
    edge_counter = torch.tensor(mp_data["edge_counter"], dtype=torch.int32).to(device)

    return edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter

def lower_bound(edge_costs, t12, t13, t23):
    edge_lb = torch.sum(torch.where(edge_costs < 0, edge_costs, torch.zeros_like(edge_costs)))
        
    a, b, c = t12, t13, t23
    zero = torch.zeros_like(a)

    lb = torch.stack([
        zero,
        a + b,
        a + c,
        b + c,
        a + b + c
    ])
    tri_lb = torch.min(lb, dim=0).values.sum()
    return edge_lb + tri_lb
   
def loss_fn(edge_costs, t12, t13, t23):
    neg_lb = -lower_bound(edge_costs, t12, t13, t23) 
    return neg_lb / edge_costs.numel()

def train():
    set_seed(42)

    wandb.init(project="rama-mlp", name="train_v2", config={
        "epochs": 10,
        "lr": 1e-3,
        "model": "MediumMLPMessagePassing",
        "hidden_dim": 64,
    })

    num_epochs = wandb.config["epochs"]
    lr = wandb.config["lr"]
    device = "cuda" if torch.cuda.is_available else "cpu"

    dataset = MulticutGraphDataset("src/message_passing_nn/data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  
    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False

    model = MLPMessagePassing().to(device)
    model.train()

    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    MODEL_PATH = "./mlp_model.pt"
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    print(f"[INFO] Found {len(dataset)} Multicut instances.")

    fails = set()
    losses = []
    lbs = []
    for epoch in range(num_epochs):
        print(f"[INFO] EPOCH: {epoch+1}")
        epoch_loss = 0
        epoch_lb = 0
        for sample in loader:
            name = sample["name"][0]
           # print(f"[LOADING] {name}...")
            try:
                i = sample["i"]
                j = sample["j"]
                costs = sample["costs"] 
                
                mp_data = rama_py.get_message_passing_data(i, j, costs, 3)

                edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = extract_data(mp_data, device)

                for _ in range(10):
                    updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                        edge_costs, t12_costs, t13_costs, t23_costs,
                        corr_12, corr_13, corr_23, edge_counter
                    )
                    edge_costs, t12_costs, t13_costs, t23_costs = updated_edge_costs, updated_t12, updated_t13, updated_t23
                
                loss = loss_fn(updated_edge_costs, updated_t12, updated_t13, updated_t23)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                lb = lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)
                epoch_lb += lb.item()
                print(f"[SUCCESS] {name}")
            except Exception as e:
                print(f"[ERROR] Failed on file {name}: {e}")
                fails.add(name)
        losses.append(epoch_loss / len(loader))
        lbs.append(epoch_lb / len(loader))
        wandb.log({
            "avg_loss": losses[-1],
            "avg_lower_bound": lbs[-1]
        }, step=epoch)


    
    torch.save(model.state_dict(), MODEL_PATH)
    wandb.save(MODEL_PATH)

    if fails:
        n = len(fails)
        print(f"[SUMMARY] {n} Failed instances:")
        for f in sorted(fails):
            print(f" - {f}")

    print("Training finished.")

if __name__ == "__main__":
    train()



#mapping, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts)
#print(f"[SUCCESS] {name}: Clusters: {mapping}, LB: {lb}")

                
# k = random.randint(0,10)
# with torch.no_grad():
#    for _ in range(k):
#        costs,triangle_costs = rama_py.message_passing(i, j, costs, triangles, triangle_costs)



# rama_py.message_passing(model,i,j,costs,triangles,triangle_costs)
# lb = rama_py.solve_dual(i,j,costs, opts)