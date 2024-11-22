import numpy as np

from sklearn.preprocessing import StandardScaler


class DataTransforms:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def save(self, path):
        np.save(
            path / "scaler.npy",
            {"mean_": self.scaler.mean_, "scale_": self.scaler.scale_},
        )

    @classmethod
    def load(cls, path):
        scaler_dict = np.load(path, allow_pickle=True).item()
        instance = cls()
        instance.scaler.mean_ = scaler_dict["mean_"]
        instance.scaler.scale_ = scaler_dict["scale_"]
        return instance
