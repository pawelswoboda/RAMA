from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class ModelConfig:
    hidden_dim: int = 55  
    num_res_blocks: int = 6 
    weight_init_mean: float = 0.0
    weight_init_std: float = 0.012
    d_k: int = 12

@dataclass
class TrainConfig:
    model_type: str = "mlp"  
    gradient_clip_norm: float = 1.0
    gradient_acc_steps: int = 4
    scheduler_step_size: int = 20 
    scheduler_gamma: float = 0.3  
    epochs: int = 80
    lr: float = 1e-3  
    num_mp_iter: int = 5  
    dist: str = "mlp"  # "uniform" or "mlp" or "attention"
    batch_size: int = 1
    weight_decay: float = 1e-6
    sweep_active: bool = False

@dataclass
class TestConfig:
    model_type: str = "mlp"  # "mlp" or "cpp"
    num_mp_iter: int = 5
    batch_size: int = 1
    dist: str = "mlp"  # "uniform" or "mlp" or "attention"

@dataclass
class DataConfig:
    dir = "src/message_passing_nn/data"
    train_dir: str = dir + "/train"
    test_dir: str = dir + "/test"
    eval_dir: str = dir + "/eval"
    cpp_dir: str = dir + "/eval/cpp"
    mlp_dir: str = dir + "/eval/mlp"
    max_dir: str = dir + "/eval/max"
    output_summary_path: str = dir + "/eval/results/summary.txt"
    output_plot_path: str = dir + "/eval/results/lb_comparison.png"

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    test: TestConfig = TestConfig()
    data: DataConfig = DataConfig()
    seed: int = 42
    model_save_path: str = "./{model_type}_model.pt"

cs = ConfigStore.instance()
cs.store(name="config", node=Config) 