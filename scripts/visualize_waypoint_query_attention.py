#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gridnav_il.models import WaypointQueryAttentionPolicy


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError(
            "Could not find model weights. Expected key 'model_state_dict' or 'state_dict'."
        )

    input_channels = int(ckpt.get("input_channels", 11))
    output_dim = int(ckpt.get("output_dim", 20))
    model_type = ckpt.get("model_type", "waypoint_query_attention")

    if model_type != "waypoint_query_attention":
        print(f"[WARN] checkpoint model_type is {model_type}, but this script expects waypoint_query_attention.")

    model = WaypointQueryAttentionPolicy(
        input_channels=input_channels,
        output_dim=output_dim,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    y_mean = ckpt.get("y_mean", None)
    y_std = ckpt.get("y_std", None)

    if y_mean is not None:
        y_mean = torch.as_tensor(y_mean, dtype=torch.float32, device=device).view(1, -1)
    if y_std is not None:
        y_std = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(1, -1)

    return model, y_mean, y_std, ckpt


def normalize_image(img: np.ndarray):
    img = img.astype(np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)

    lo = np.percentile(img[finite], 2)
    hi = np.percentile(img[finite], 98)

    if hi <= lo + 1e-6:
        return np.zeros_like(img, dtype=np.float32)

    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def waypoint_to_pixel(x_local, y_local, crop_size, resolution):
    half = crop_size // 2
    row = half - x_local / resolution
    col = half - y_local / resolution
    return row, col


def draw_waypoints(ax, waypoints, crop_size, resolution, label, marker):
    rows = []
    cols = []

    for wp in waypoints:
        x_local = wp[0]
        y_local = wp[1]
        row, col = waypoint_to_pixel(x_local, y_local, crop_size, resolution)
        rows.append(row)
        cols.append(col)

    ax.plot(cols, rows, marker=marker, linewidth=2, markersize=5, label=label)


def extract_cross_attention(attn_dict):
    if not isinstance(attn_dict, dict):
        raise TypeError(
            "Expected attention output to be a dict with key 'cross_attention'."
        )

    if "cross_attention" not in attn_dict:
        raise KeyError(
            "Could not find 'cross_attention' in attention output."
        )

    cross_attn = attn_dict["cross_attention"]

    # Expected shape if average_attn_weights=False:
    # [B, num_heads, num_waypoints, num_tokens]
    #
    # But handle [B, num_waypoints, num_tokens] too, just in case.
    if cross_attn.ndim == 4:
        cross_attn = cross_attn.mean(dim=1)
    elif cross_attn.ndim == 3:
        pass
    else:
        raise ValueError(
            f"Unexpected cross attention shape: {tuple(cross_attn.shape)}"
        )

    # Now [B, num_waypoints, num_tokens]
    return cross_attn

def draw_token_grid(ax, crop_size, token_side):
    step = crop_size / token_side

    for i in range(token_side + 1):
        p = i * step
        ax.axhline(p, color="yellow", linewidth=1.0, alpha=0.8)
        ax.axvline(p, color="yellow", linewidth=1.0, alpha=0.8)

    for r in range(token_side):
        for c in range(token_side):
            ax.text(
                (c + 0.5) * step,
                (r + 0.5) * step,
                f"{r},{c}",
                color="yellow",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )

def plot_attention_visualization(
    x_np,
    y_target,
    y_pred,
    cross_attn,
    out_path,
    background_channel,
    resolution,
    title,
):
    crop_size = x_np.shape[-1]
    output_dim = y_pred.shape[0]

    if output_dim % 4 != 0:
        raise ValueError(f"Output dim must be divisible by 4, got {output_dim}")

    num_waypoints = output_dim // 4

    target_wps = y_target.reshape(num_waypoints, 4)
    pred_wps = y_pred.reshape(num_waypoints, 4)

    bg = normalize_image(x_np[background_channel])

    attn_one = cross_attn[0].detach().cpu().numpy()
    # shape: [num_waypoints, num_tokens]

    num_tokens = attn_one.shape[1]
    token_side = int(math.sqrt(num_tokens))

    if token_side * token_side != num_tokens:
        raise ValueError(f"Expected square token grid. Got {num_tokens} tokens.")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        squeeze=False,
    )

    all_axes = axes.ravel()

    # ------------------------------------------------------------
    # Panel 0: input crop with 4x4 token grid + waypoints
    # ------------------------------------------------------------
    ax = all_axes[0]
    ax.imshow(bg, origin="upper", cmap="gray")

    draw_token_grid(ax, crop_size=crop_size, token_side=token_side)

    draw_waypoints(
        ax,
        target_wps,
        crop_size=crop_size,
        resolution=resolution,
        label="target",
        marker="o",
    )
    draw_waypoints(
        ax,
        pred_wps,
        crop_size=crop_size,
        resolution=resolution,
        label="pred",
        marker="x",
    )

    ax.plot(crop_size // 2, crop_size // 2, "r+", markersize=10, markeredgewidth=2)
    ax.arrow(
        crop_size // 2,
        crop_size // 2,
        0,
        -8,
        color="red",
        width=0.7,
        head_width=3,
        length_includes_head=True,
    )

    ax.set_title(f"Input channel {background_channel}\nwith 4×4 token grid")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

    # ------------------------------------------------------------
    # Panels 1..5: clean 4x4 attention tables
    # ------------------------------------------------------------
    for i in range(num_waypoints):
        ax = all_axes[i + 1]

        attn_grid = attn_one[i].reshape(token_side, token_side)

        # Normalize only for color display.
        # Keep original values for printed numbers.
        attn_display = attn_grid / (attn_grid.max() + 1e-8)

        im = ax.imshow(attn_display, origin="upper", cmap="magma", vmin=0.0, vmax=1.0)

        max_idx = np.unravel_index(np.argmax(attn_grid), attn_grid.shape)

        for r in range(token_side):
            for c in range(token_side):
                value = attn_grid[r, c]

                is_max = (r == max_idx[0] and c == max_idx[1])

                text_color = "cyan" if is_max else "white"
                font_weight = "bold" if is_max else "normal"

                ax.text(
                    c,
                    r,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight=font_weight,
                )

        ax.set_xticks(range(token_side))
        ax.set_yticks(range(token_side))
        ax.set_xticklabels([f"c{c}" for c in range(token_side)])
        ax.set_yticklabels([f"r{r}" for r in range(token_side)])

        ax.set_title(
            f"Waypoint query {i + 1}\n"
            f"max token = row {max_idx[0]}, col {max_idx[1]}"
        )

        # Draw cell borders.
        ax.set_xticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=1000)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument(
        "--background_channel",
        type=int,
        default=1,
        help="Channel to use as background. For current setup: 0 occupancy, 1 raw cost-to-go, 10 goal mask.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="Grid resolution in meters/cell. Used only to draw waypoint coordinates on crop.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]

    if args.sample_index < 0 or args.sample_index >= len(X):
        raise IndexError(
            f"sample_index {args.sample_index} out of range for dataset with {len(X)} samples"
        )

    x_np = X[args.sample_index].astype(np.float32)
    y_target = Y[args.sample_index].astype(np.float32)

    model, y_mean, y_std, ckpt = load_checkpoint(args.checkpoint, device)

    x = torch.from_numpy(x_np).unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(x, return_attention=True)

        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "model(x, return_attention=True) should return (prediction, attention_dict)."
            )

        pred_norm, attn_dict = result

        if y_mean is not None and y_std is not None:
            pred = pred_norm * y_std + y_mean
        else:
            pred = pred_norm

        pred_np = pred[0].detach().cpu().numpy().astype(np.float32)

        cross_attn = extract_cross_attention(attn_dict)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = (
        f"Waypoint-query cross-attention | "
        f"sample {args.sample_index} | "
        f"{Path(args.checkpoint).name}"
    )

    plot_attention_visualization(
        x_np=x_np,
        y_target=y_target,
        y_pred=pred_np,
        cross_attn=cross_attn,
        out_path=str(out_path),
        background_channel=args.background_channel,
        resolution=args.resolution,
        title=title,
    )

    print("Saved:", out_path)

    print("\nPredicted waypoints:")
    print(pred_np.reshape(-1, 4))

    print("\nTarget waypoints:")
    print(y_target.reshape(-1, 4))


if __name__ == "__main__":
    main()