from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class ModelConfig:
    hidden_dim: int = MISSING  # 32
    num_res_blocks: int = MISSING # 6
    weight_init_mean: float = 0.0
    weight_init_std: float = MISSING # 0.02
    d_k: int = MISSING # 8

@dataclass
class TrainConfig:
    model_type: str = "mlp"  # "mlp" or "gnn"
    gradient_clip_norm: float = 1.0
    gradient_acc_steps: int = 4
    scheduler_step_size: int = MISSING  # 5
    scheduler_gamma: float = MISSING  # 0.3
    epochs: int = 20
    lr: float = MISSING  # 1e-3
    num_mp_iter: int = 5  
    dist: str = "attention"  # "uniform" or "mlp" or "attention"
    batch_size: int = 1
    weight_decay: float = MISSING  # 1e-8

@dataclass
class TestConfig:
    model_type: str = "mlp"  # "mlp", "gnn" or "cpp" with DISABLE_MLP=1
    num_mp_iter: int = 5
    batch_size: int = 1
    dist: str = "attention"  # "uniform" or "mlp" or "attention"

@dataclass
class DataConfig:
    train_dir: str = "src/message_passing_nn/data/train"
    test_dir: str = "src/message_passing_nn/data/test"
    eval_dir: str = "src/message_passing_nn/data/eval"
    cpp_dir: str = "src/message_passing_nn/data/eval/cpp"
    mlp_dir: str = "src/message_passing_nn/data/eval/mlp"
    gnn_dir: str = "src/message_passing_nn/data/eval/gnn"
    output_summary_path: str = "src/message_passing_nn/data/eval/results/summary.txt"
    output_plot_path: str = "src/message_passing_nn/data/eval/results/lb_comparison.png"

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