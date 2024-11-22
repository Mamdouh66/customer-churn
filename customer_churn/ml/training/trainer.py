import copy

from pathlib import Path

import torch
import numpy as np

from customer_churn.ml.configs import ModelConfigs
from customer_churn.ml.models.model import ChurnModel

from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ChurnTrainer:
    def __init__(
        self,
        model: ChurnModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: torch.device,
        config: ModelConfigs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.best_val_auc = 0
        self.best_model = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            total_loss += loss.item()
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)

        return {
            "loss": total_loss / len(data_loader),
            "accuracy": accuracy_score(targets, predictions > 0.5),
            "roc_auc": roc_auc_score(targets, predictions),
            "predictions": predictions,
            "targets": targets,
        }

    def train(self):
        no_improve_count = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_aucs = []

        for epoch in range(self.config.n_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_metrics["loss"])
            val_accuracies.append(val_metrics["accuracy"])
            val_aucs.append(val_metrics["roc_auc"])

            self.scheduler.step(val_metrics["loss"])

            if val_metrics["roc_auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["roc_auc"]
                self.best_model = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if (epoch + 1) % 5 == 0:
                self._print_metrics(epoch, train_loss, val_metrics)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "val_aucs": val_aucs,
            "best_model": self.best_model,
        }

    def _print_metrics(self, epoch, train_loss, val_metrics):
        print(f"Epoch {epoch+1}/{self.config.n_epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'Val ROC-AUC: {val_metrics["roc_auc"]:.4f}')
        print("-" * 50)
