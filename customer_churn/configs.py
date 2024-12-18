import torch

from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelConfigs(BaseModel):
    hidden_dims: list = [128, 64, 32]
    dropout_rate: float = 0.3
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    n_epochs: int = 50


class InferenceConfig(BaseSettings):
    MODEL_PATH: str = "models/model.pth"
    TRANSFORMS_PATH: str = "models/scaler.npy"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
