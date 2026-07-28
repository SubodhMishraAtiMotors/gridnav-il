import argparse
import os
import json
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from gridnav_il.dataset import GridExpertTorchDataset
from gridnav_il.models import CNNPolicy, CNNAttentionPolicy
from torch.utils.data import Dataset, DataLoader, Subset

class GridNavDataset(Dataset):
    def __init__(self, X, Y, indices, y_mean, y_std):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        # Keep the stored dataset float16 on CPU.
        # Convert only this sample to float32 for the model.
        x = torch.from_numpy(self.X[sample_idx].astype(np.float32))

        y = torch.from_numpy(self.Y[sample_idx].astype(np.float32))
        y = (y - self.y_mean) / self.y_std

        return x, y

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN policy for grid navigation imitation learning."
    )

    parser.add_argument("--dataset", type=str, required=True)

    # New preferred output style.
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints/cnn_policy_run",
        help="Directory where checkpoints, metrics, and plots will be saved.",
    )

    # Kept for backward compatibility. If supplied, we infer out_dir from this path.
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Deprecated. Use --out_dir instead. If provided, output folder is inferred from this path.",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "cnn_attention"],
        help="Model architecture to train.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=8,
        help="Stop if validation loss does not improve for this many epochs.",
    )

    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-5,
        help="Minimum validation loss improvement required to reset patience.",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save numbered checkpoints every N epochs. Use 0 to disable periodic checkpoints.",
    )


    return parser.parse_args()


def resolve_out_dir(args):
    if args.out is None:
        return args.out_dir

    # Backward compatibility:
    # --out checkpoints/foo.pt -> checkpoints/foo/
    root, ext = os.path.splitext(args.out)

    if ext == ".pt":
        return root

    return args.out


def split_indices_by_demo_ids(demo_ids, train_fraction, seed):
    rng = np.random.default_rng(seed)

    unique_demo_ids = np.unique(demo_ids)
    rng.shuffle(unique_demo_ids)

    num_train_demos = int(train_fraction * len(unique_demo_ids))
    num_train_demos = max(1, min(num_train_demos, len(unique_demo_ids) - 1))

    train_demo_ids = set(unique_demo_ids[:num_train_demos].tolist())
    val_demo_ids = set(unique_demo_ids[num_train_demos:].tolist())

    train_indices = np.array(
        [i for i, d in enumerate(demo_ids) if int(d) in train_demo_ids],
        dtype=np.int64,
    )

    val_indices = np.array(
        [i for i, d in enumerate(demo_ids) if int(d) in val_demo_ids],
        dtype=np.int64,
    )

    return train_indices, val_indices, train_demo_ids, val_demo_ids


def split_indices_random(num_samples, train_fraction, seed):
    rng = np.random.default_rng(seed)

    indices = np.arange(num_samples)
    rng.shuffle(indices)

    num_train = int(train_fraction * num_samples)
    num_train = max(1, min(num_train, num_samples - 1))

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    return train_indices, val_indices


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


def save_checkpoint(
    path,
    model,
    y_mean,
    y_std,
    input_channels,
    output_dim,
    train_losses,
    val_losses,
    epoch,
    best_epoch,
    best_val_loss,
    split_type,
    args,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std,
        "input_channels": int(input_channels),
        "output_dim": int(output_dim),
        "model_type": getattr(args, "model_type", "cnn"),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch": None if epoch is None else int(epoch),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "split_type": split_type,
        "args": vars(args),
    }

    torch.save(checkpoint, path)


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

class IndexedGridExpertTorchDataset(Dataset):
    def __init__(self, X, Y, indices, y_mean, y_std):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices, dtype=np.int64)

        # Keep these as numpy arrays so workers/dataloader do not hold GPU tensors.
        self.y_mean = y_mean.detach().cpu().numpy().astype(np.float32)
        self.y_std = y_std.detach().cpu().numpy().astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = int(self.indices[idx])

        # X is stored as float16 in the dataset.
        # Convert only this sample to float32.
        x = self.X[sample_idx].astype(np.float32, copy=False)

        y = self.Y[sample_idx].astype(np.float32, copy=False)
        y = (y - self.y_mean) / self.y_std

        return torch.from_numpy(x), torch.from_numpy(y)

def build_model(model_type, input_channels, output_dim):
    if model_type == "cnn":
        return CNNPolicy(
            input_channels=input_channels,
            output_dim=output_dim,
        )

    if model_type == "cnn_attention":
        return CNNAttentionPolicy(
            input_channels=input_channels,
            output_dim=output_dim,
            d_model=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

    raise ValueError(f"Unknown model_type: {model_type}")

def main():
    args = parse_args()

    out_dir = resolve_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading dataset:", args.dataset)
    data = np.load(args.dataset, allow_pickle=True)

    X = data["X"]
    Y = data["Y"]

    print("X shape:", X.shape, X.dtype)
    print("Y shape:", Y.shape, Y.dtype)

    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)

    print("y_mean:", y_mean)
    print("y_std:", y_std)

    num_samples = X.shape[0]

    if "demo_ids" in data.files:
        demo_ids = data["demo_ids"]

        (
            train_indices,
            val_indices,
            train_demo_ids,
            val_demo_ids,
        ) = split_indices_by_demo_ids(
            demo_ids=demo_ids,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )

        split_type = "demo_id"

        print("Using demo-level train/val split.")
        print("Train demos:", len(train_demo_ids))
        print("Val demos:", len(val_demo_ids))

    else:
        train_indices, val_indices = split_indices_random(
            num_samples=num_samples,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )

        split_type = "random_frame"

        print("WARNING: demo_ids not found. Using random frame-level split.")

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)

    print("Train samples:", len(train_indices))
    print("Val samples:", len(val_indices))

    train_dataset = IndexedGridExpertTorchDataset(
        X=X,
        Y=Y,
        indices=train_indices,
        y_mean=y_mean,
        y_std=y_std,
    )

    val_dataset = IndexedGridExpertTorchDataset(
        X=X,
        Y=Y,
        indices=val_indices,
        y_mean=y_mean,
        y_std=y_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(
        model_type=args.model_type,
        input_channels=X.shape[1],
        output_dim=Y.shape[1],
    ).to(device)

    print("Model type:", args.model_type)

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
    epochs_without_improvement = 0

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

        improved = val_loss < (best_val_loss - args.min_delta)

        if improved:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0

            best_path = os.path.join(out_dir, "best.pt")
            save_checkpoint(
                path=best_path,
                model=model,
                y_mean=y_mean,
                y_std=y_std,
                input_channels=X.shape[1],
                output_dim=Y.shape[1],
                train_losses=train_losses,
                val_losses=val_losses,
                epoch=epoch,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                split_type=split_type,
                args=args,
            )

        else:
            epochs_without_improvement += 1

        if args.save_every > 0 and (epoch % args.save_every == 0):
            epoch_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            save_checkpoint(
                path=epoch_path,
                model=model,
                y_mean=y_mean,
                y_std=y_std,
                input_channels=X.shape[1],
                output_dim=Y.shape[1],
                train_losses=train_losses,
                val_losses=val_losses,
                epoch=epoch,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                split_type=split_type,
                args=args,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | "
            f"best val: {best_val_loss:.6f} | "
            f"patience: {epochs_without_improvement}/{args.early_stopping_patience}"
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            print()
            print(
                "Early stopping triggered at epoch "
                f"{epoch}. Best epoch was {best_epoch}."
            )
            break

    if best_state_dict is None:
        raise RuntimeError("Training failed: no best model state was saved.")

    # Save final model as it exists at stopping.
    final_path = os.path.join(out_dir, "final.pt")
    save_checkpoint(
        path=final_path,
        model=model,
        y_mean=y_mean,
        y_std=y_std,
        input_channels=X.shape[1],
        output_dim=Y.shape[1],
        train_losses=train_losses,
        val_losses=val_losses,
        epoch=len(train_losses) - 1,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        split_type=split_type,
        args=args,
    )

    # Restore and save best again explicitly.
    model.load_state_dict(best_state_dict)

    best_path = os.path.join(out_dir, "best.pt")
    save_checkpoint(
        path=best_path,
        model=model,
        y_mean=y_mean,
        y_std=y_std,
        input_channels=X.shape[1],
        output_dim=Y.shape[1],
        train_losses=train_losses,
        val_losses=val_losses,
        epoch=best_epoch,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        split_type=split_type,
        args=args,
    )

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "split_type": split_type,
                "num_train_samples": int(len(train_indices)),
                "num_val_samples": int(len(val_indices)),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "out_dir": out_dir,
                "best_checkpoint": best_path,
                "final_checkpoint": final_path,
            },
            f,
            indent=2,
        )

    plot_path = os.path.join(out_dir, "loss_curve.png")

    plot_loss_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        out_path=plot_path,
    )

    print()
    print("Saved run directory:", out_dir)
    print("Saved best checkpoint:", best_path)
    print("Saved final checkpoint:", final_path)
    print("Saved metrics:", metrics_path)
    print("Saved loss curve:", plot_path)
    print("Saved config:", config_path)
    print("Split type:", split_type)
    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
