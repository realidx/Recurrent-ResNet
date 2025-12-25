from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    name: str = "dm_control/walker_walk"
    num_envs: int = 64
    frame_stack: int = 3


@dataclass
class TrainConfig:
    total_env_steps: int = 1_000_000
    utd_ratio: int = 10
    batch_size: int = 512
    update_every: int = 1
    log_interval: int = 1000
    eval_interval: int = 10_000


@dataclass
class EncoderConfig:
    type: str = "untied_resnet"
    depth: int = 64
    width: int = 256
    activation: str = "relu"


@dataclass
class RecurrentConfig:
    steps: int = 4
    tie_weights: bool = True
    depth_dropout: float = 0.0
    apply_to_actor: bool = False
    apply_to_critic: bool = True


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)


@dataclass
class LoggingConfig:
    backend: str = "tensorboard"
    project: str = "recurrent-resnet"


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cuda"
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
