from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class TrainingConfig:
    epochs: int = 20
    lr: float = 1e-4
    model_type: str = "mlp"  # "mlp" or "gnn"
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.3

@dataclass
class Config:
    training: TrainingConfig = TrainingConfig()
    seed: int = 42
    data_dir: str = "src/message_passing_nn/data/train"
    model_save_path: str = "./{model_type}_model.pt"

cs = ConfigStore.instance()
cs.store(name="config", node=Config) 