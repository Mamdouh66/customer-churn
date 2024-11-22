from typing import Optional


from customer_churn.configs import InferenceConfig
from customer_churn.ml.inference.predictor import ChurnPredictor

import torch


class ModelManager:
    _instance: Optional[ChurnPredictor] = None

    @classmethod
    def get_predictor(cls) -> ChurnPredictor:
        if cls._instance is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            settings = InferenceConfig()
            cls._instance = ChurnPredictor.load(
                model_path=settings.MODEL_PATH,
                transforms_path=settings.TRANSFORMS_PATH,
                device=device,
            )
        return cls._instance
