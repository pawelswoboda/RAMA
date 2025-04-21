from torch.utils.data import DataLoader
import rama_py
from multicut_dataset import MulticutGraphDataset
from mlp.mlp_message_passing import MLPMessagePassing
from gnn.gnn_message_passing import GNNMessagePassing
import os
import torch
import wandb
import nn_utils as utils

def loss_fn(edge_costs, t12, t13, t23):
    neg_lb = -utils.lower_bound(edge_costs, t12, t13, t23) 
    return neg_lb / edge_costs.numel()

def train(model_type="mlp"):  # use "mlp" or "gnn"
    utils.set_seed(42)

    wandb.init(project="rama-learned-mp", name=f"train_{model_type}_v3", config={
        "epochs": 15,
        "lr": 1e-3,
        "model": f"{model_type}_v3",
    })

    num_epochs = wandb.config["epochs"]
    lr = wandb.config["lr"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MulticutGraphDataset("src/message_passing_nn/data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  
    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False

    if model_type == "mlp":
        model = MLPMessagePassing().to(device)
    elif model_type == "gnn":
        model = GNNMessagePassing().to(device)
    else:
        print("[ERROR] CANT FIND MODEL, USE mlp OR gnn")
        return
    
    model.train()

    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    MODEL_PATH = f"./{model_type}_model.pt"   
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    
    print(f"[INFO] MODEL: {model_type}")
    print(f"[INFO] DEVICE: {next(model.parameters()).device}")
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
                normed_costs, factor = utils.normalise_costs(costs)    

                mp_data = rama_py.get_message_passing_data(i, j, normed_costs.tolist(), 3)

                edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = utils.extract_data(mp_data, device)
                
                loss = 0
                if model_type == "mlp":
                    for _ in range(5):
                        updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                            edge_costs, t12_costs, t13_costs, t23_costs,
                            corr_12, corr_13, corr_23, edge_counter
                        )
                        loss += loss_fn(updated_edge_costs, updated_t12, updated_t13, updated_t23)
                        edge_costs, t12_costs, t13_costs, t23_costs = updated_edge_costs, updated_t12, updated_t13, updated_t23
                        

                elif model_type == "gnn":
                    print("TODO")
                    
                else:
                    print("[ERROR] CANT FIND MODEL")
                    return
 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                lb = utils.lower_bound(updated_edge_costs, updated_t12, updated_t13, updated_t23)
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