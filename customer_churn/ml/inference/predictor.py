from pathlib import Path

from customer_churn.ml.data.transforms import DataTransforms
from customer_churn.ml.models.model import ChurnModel

import torch
import numpy as np


class ChurnPredictor:
    def __init__(
        self, model: ChurnModel, transforms: DataTransforms, device: torch.device
    ):
        self.model = model
        self.transforms = transforms
        self.device = device
        self.model.eval()

    def predict(self, X: np.ndarray):
        X_scaled = self.transforms.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy()

    @classmethod
    def load(cls, model_path: Path, transforms_path: Path, device: torch.device):
        model = torch.load(model_path)
        transforms = DataTransforms.load(transforms_path)
        return cls(model, transforms, device)
