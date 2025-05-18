import wandb
import sys
import os
import optuna
from optuna.samplers import TPESampler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn_trainer import train
from nn_tester import test
from nn_evaluater import evaluate
from config import Config

def objective(trial):
    cfg = Config()
    cfg.train.sweep_active = True

    cfg.model.hidden_dim = trial.suggest_int("hidden_dim", 48, 64)
    cfg.model.num_res_blocks = trial.suggest_int("num_res_blocks", 4, 8)
    cfg.model.d_k = trial.suggest_int("d_k", 8, 16)
    cfg.model.weight_init_std = trial.suggest_float("weight_init_std", 0.01, 0.05, log=True)
    cfg.train.scheduler_step_size = trial.suggest_int("scheduler_step_size", 4, 9)
    cfg.train.scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.1, 0.5)
    cfg.train.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True)
    cfg.train.dist = trial.suggest_categorical("dist", ["attention", "mlp"])
    cfg.test.dist = cfg.train.dist
    
    wandb.init(
        project="rama-learned-mp",
        name=f"trial_{trial.number}_v5",
        config={
            "trial_number": trial.number,
            "hidden_dim": cfg.model.hidden_dim,
            "num_res_blocks": cfg.model.num_res_blocks,
            "d_k": cfg.model.d_k,
            "weight_init_std": cfg.model.weight_init_std,
            "scheduler_step_size": cfg.train.scheduler_step_size,
            "scheduler_gamma": cfg.train.scheduler_gamma,
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "dist": cfg.train.dist
        }
    )

    train(cfg)
    test(cfg)
    avg_diff = evaluate(cfg)

    wandb.log({"avg_mlp_diff": avg_diff})
    wandb.finish()

    return avg_diff

def main():
    study = optuna.create_study(
        study_name="rama_mp_optimization",
        sampler=TPESampler(),
        direction="maximize"
    )

    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 