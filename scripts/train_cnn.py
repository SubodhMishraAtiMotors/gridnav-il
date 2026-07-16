import argparse
import os
import json
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from gridnav_il.dataset import GridExpertTorchDataset
from gridnav_il.models import CNNPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN policy for grid navigation imitation learning."
    )

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, default="checkpoints/cnn_policy.pt")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)

            batch_size = X_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def plot_loss_curves(train_losses, val_losses, out_path):
    epochs = np.arange(len(train_losses))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss on normalized actions")
    plt.title("CNN policy training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading dataset:", args.dataset)
    data = np.load(args.dataset, allow_pickle=True)

    X = data["X"]
    Y = data["Y"]

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)

    print("y_mean:", y_mean)
    print("y_std:", y_std)

    full_dataset = GridExpertTorchDataset(
        X=X,
        Y=Y,
        y_mean=y_mean,
        y_std=y_std,
    )

    num_total = len(full_dataset)
    num_train = int(args.train_fraction * num_total)
    num_val = num_total - num_train

    train_dataset, val_dataset = random_split(
        full_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = CNNPolicy(
        input_channels=X.shape[1],
        output_dim=Y.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = None

    for epoch in range(args.epochs):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = X_batch.shape[0]
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_loss = total_train_loss / max(total_train_samples, 1)
        val_loss = evaluate_model(model, val_loader, loss_fn, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f}"
        )

    if best_state_dict is None:
        raise RuntimeError("Training failed: no best model state was saved.")

    model.load_state_dict(best_state_dict)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std,
        "input_channels": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }

    torch.save(checkpoint, args.out)

    metrics_path = args.out.replace(".pt", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
            },
            f,
            indent=2,
        )

    plot_path = args.out.replace(".pt", "_loss_curve.png")

    plot_loss_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        out_path=plot_path,
    )

    print()
    print("Saved checkpoint:", args.out)
    print("Saved metrics:", metrics_path)
    print("Saved loss curve:", plot_path)
    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()