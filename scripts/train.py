import torch
from torch.utils.data import DataLoader
import polars as pl
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

from customer_churn.ml.data.dataset import ChurnDataset
from customer_churn.ml.data.transforms import DataTransforms
from customer_churn.ml.models.model import ChurnModel
from customer_churn.ml.configs import ModelConfigs
from customer_churn.ml.training.trainer import ChurnTrainer
from customer_churn.ml.training.utils import (
    plot_training_metrics,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[128, 64, 32],
        help="Hidden layer dimensions",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="Number of training epochs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ModelConfigs(
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        n_epochs=args.n_epochs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = pl.scan_csv("data/dataframe_after_data_step.csv").collect()
    X = full_df.drop("churn").to_numpy()
    y = full_df.select("churn").to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    transforms = DataTransforms()
    X_train_scaled = transforms.fit_transform(X_train)
    X_test_scaled = transforms.transform(X_test)

    train_dataset = ChurnDataset(X_train_scaled, y_train)
    test_dataset = ChurnDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = ChurnModel(
        input_dim=X_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate,
    ).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    trainer = ChurnTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )

    metrics = trainer.train()

    plot_training_metrics(metrics)

    model.load_state_dict(metrics["best_model"])
    final_metrics = trainer.evaluate(test_loader)

    plot_confusion_matrix(final_metrics["targets"], final_metrics["predictions"])

    save_path = Path("models")
    save_path.mkdir(exist_ok=True)

    torch.save(model.state_dict(), save_path / "model.pth")
    # transforms.save(save_path)


if __name__ == "__main__":
    main()
