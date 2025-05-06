import sys
import os
from torch.utils.data import DataLoader
import rama_py
from multicut_dataset import MulticutGraphDataset
from mlp.mlp_message_passing import MLPMessagePassing
from gnn.gnn_message_passing import GNNMessagePassing
import os
import torch
import wandb
import nn_utils as utils
import hydra
from configuration.config import Config

def loss_fn(edge_costs, t12, t13, t23):
    neg_lb = -utils.lower_bound(edge_costs, t12, t13, t23) 
    return neg_lb / edge_costs.numel()

@hydra.main(version_base=None, config_name="config")
def train(cfg: Config):
    utils.set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="rama-learned-mp", name=f"train_{cfg.train.model_type}", config={
        "epochs": cfg.train.epochs,
        "lr": cfg.train.lr,
        "model_type": cfg.train.model_type,
        "model_config": cfg.model,
        "train_config": cfg.train
    })

    dataset = MulticutGraphDataset(cfg.data.train_dir)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    opts = rama_py.multicut_solver_options("PD")
    opts.verbose = False

    if cfg.train.model_type == "mlp":
        model = MLPMessagePassing(cfg.model).to(device)
    elif cfg.train.model_type == "gnn":
        model = GNNMessagePassing().to(device)
    else:
        print("[ERROR] CANT FIND MODEL, USE mlp OR gnn")
        return
    
    model.train()
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.train.lr, 
        weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.train.scheduler_step_size, 
        gamma=cfg.train.scheduler_gamma
    )

    model_path = cfg.model_save_path.format(model_type=cfg.train.model_type)

    #if os.path.exists(model_path):
     #   model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    print(f"[INFO] MODEL: {cfg.train.model_type}")
    print(f"[INFO] DEVICE: {device}")
    print(f"[INFO] Found {len(dataset)} Multicut instances.")

    fails = set()
    losses = []
    lbs = []
    for epoch in range(cfg.train.epochs):
        print(f"[INFO] EPOCH: {epoch+1}")
        epoch_loss = 0
        epoch_lb = 0
        for step_counter, sample in enumerate(loader, 1):
            name = sample["name"][0]
            try:
                i = sample["i"]
                j = sample["j"]
                costs = sample["costs"] 
                normed_costs, factor = utils.normalise_costs(costs)  

                mp_data = rama_py.get_message_passing_data(i, j, normed_costs.tolist(), 3)
                edge_costs, t12_costs, t13_costs, t23_costs, corr_12, corr_13, corr_23, edge_counter = utils.extract_data(mp_data, device)
                
                loss = 0
                if cfg.train.model_type == "mlp":
                    for _ in range(cfg.train.num_mp_iter):
                        updated_edge_costs, updated_t12, updated_t13, updated_t23 = model(
                            edge_costs, t12_costs, t13_costs, t23_costs,
                            corr_12, corr_13, corr_23, edge_counter, dist=cfg.train.dist
                        )
                        loss += loss_fn(updated_edge_costs, updated_t12, updated_t13, updated_t23)
                        edge_costs, t12_costs, t13_costs, t23_costs = updated_edge_costs, updated_t12, updated_t13, updated_t23
                        
                elif cfg.train.model_type == "gnn":
                    print("TODO")
                    
                else:
                    print("[ERROR] CANT FIND MODEL")
                    return
 
                loss.backward()
                
                if step_counter % cfg.train.gradient_acc_steps == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
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
        
        scheduler.step()

    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    if fails:
        n = len(fails)
        print(f"[SUMMARY] {n} Failed instances:")
        for f in sorted(fails):
            print(f" - {f}")

    print("TRAINING FINISHED")

if __name__ == "__main__":
    train()

