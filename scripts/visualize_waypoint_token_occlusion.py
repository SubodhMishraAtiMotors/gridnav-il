#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gridnav_il.models import CNNPolicy, WaypointQueryAttentionPolicy

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError("Expected checkpoint key 'model_state_dict' or 'state_dict'.")

    input_channels = int(ckpt.get("input_channels", 11))
    output_dim = int(ckpt.get("output_dim", 20))
    model_type = ckpt.get("model_type", "waypoint_query_attention")

    if model_type != "waypoint_query_attention":
        print(
            f"[WARN] checkpoint model_type is {model_type}, "
            "but this script expects waypoint_query_attention."
        )

    if model_type == "cnn":
        model = CNNPolicy(
            input_channels=input_channels,
            output_dim=output_dim,
        )
    elif model_type == "waypoint_query_attention":
        model = WaypointQueryAttentionPolicy(
            input_channels=input_channels,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unsupported model_type for occlusion visualization: {model_type}")

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


def denormalize_prediction(pred, y_mean, y_std):
    if y_mean is not None and y_std is not None:
        return pred * y_std + y_mean
    return pred


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


def predict_waypoints(model, x_np, y_mean, y_std, device):
    x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(x)

        # In case your forward can return tuple in some mode, guard it.
        if isinstance(pred_norm, tuple):
            pred_norm = pred_norm[0]

        pred = denormalize_prediction(pred_norm, y_mean, y_std)

    return pred[0].detach().cpu().numpy().astype(np.float32)


def make_occluded_copy(
    x_np,
    token_row,
    token_col,
    token_side,
    mode,
    fill_value,
    channels_to_occlude,
):
    x_occ = x_np.copy()
    crop_size = x_np.shape[-1]
    step = crop_size // token_side

    r0 = token_row * step
    r1 = (token_row + 1) * step
    c0 = token_col * step
    c1 = (token_col + 1) * step

    if channels_to_occlude is None:
        channel_indices = range(x_np.shape[0])
    else:
        channel_indices = channels_to_occlude

    for ch in channel_indices:
        patch = x_occ[ch, r0:r1, c0:c1]

        if mode == "zero":
            patch[...] = 0.0
        elif mode == "mean":
            patch[...] = float(np.mean(x_np[ch]))
        elif mode == "channel_mean":
            patch[...] = float(np.mean(x_np[ch]))
        elif mode == "constant":
            patch[...] = float(fill_value)
        else:
            raise ValueError(f"Unknown occlusion mode: {mode}")

    return x_occ


def compute_occlusion_influence(
    model,
    x_np,
    baseline_pred,
    y_mean,
    y_std,
    device,
    token_side,
    mode,
    fill_value,
    channels_to_occlude,
):
    num_waypoints = baseline_pred.shape[0] // 4
    baseline_wps = baseline_pred.reshape(num_waypoints, 4)

    influence_xy = np.zeros((num_waypoints, token_side, token_side), dtype=np.float32)
    influence_theta = np.zeros((num_waypoints, token_side, token_side), dtype=np.float32)
    influence_total = np.zeros((num_waypoints, token_side, token_side), dtype=np.float32)

    for r in range(token_side):
        for c in range(token_side):
            x_occ = make_occluded_copy(
                x_np=x_np,
                token_row=r,
                token_col=c,
                token_side=token_side,
                mode=mode,
                fill_value=fill_value,
                channels_to_occlude=channels_to_occlude,
            )

            pred_occ = predict_waypoints(model, x_occ, y_mean, y_std, device)
            occ_wps = pred_occ.reshape(num_waypoints, 4)

            for k in range(num_waypoints):
                dx = occ_wps[k, 0] - baseline_wps[k, 0]
                dy = occ_wps[k, 1] - baseline_wps[k, 1]

                xy_delta = math.sqrt(dx * dx + dy * dy)

                # Compare heading represented as sin/cos.
                dsin = occ_wps[k, 2] - baseline_wps[k, 2]
                dcos = occ_wps[k, 3] - baseline_wps[k, 3]
                theta_delta = math.sqrt(dsin * dsin + dcos * dcos)

                influence_xy[k, r, c] = xy_delta
                influence_theta[k, r, c] = theta_delta

                # Main score: mostly position, small heading contribution.
                influence_total[k, r, c] = xy_delta + 0.25 * theta_delta

    return influence_xy, influence_theta, influence_total


def plot_occlusion_visualization(
    x_np,
    y_target,
    baseline_pred,
    influence,
    out_path,
    background_channel,
    resolution,
    title,
):
    crop_size = x_np.shape[-1]
    output_dim = baseline_pred.shape[0]

    if output_dim % 4 != 0:
        raise ValueError(f"Output dim must be divisible by 4, got {output_dim}")

    num_waypoints = output_dim // 4
    target_wps = y_target.reshape(num_waypoints, 4)
    pred_wps = baseline_pred.reshape(num_waypoints, 4)

    token_side = influence.shape[-1]
    bg = normalize_image(x_np[background_channel])

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        squeeze=False,
    )

    all_axes = axes.ravel()

    # Panel 0: input with token grid.
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

    ax.set_title(f"Input channel {background_channel}\nwith 4×4 occlusion grid")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

    # Shared scale across waypoint influence maps.
    vmax = float(np.max(influence))
    if vmax <= 1e-8:
        vmax = 1.0

    for k in range(num_waypoints):
        ax = all_axes[k + 1]

        grid = influence[k]
        im = ax.imshow(grid, origin="upper", cmap="magma", vmin=0.0, vmax=vmax)

        max_idx = np.unravel_index(np.argmax(grid), grid.shape)

        for r in range(token_side):
            for c in range(token_side):
                value = grid[r, c]
                is_max = (r == max_idx[0] and c == max_idx[1])

                ax.text(
                    c,
                    r,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="cyan" if is_max else "white",
                    fontsize=9,
                    fontweight="bold" if is_max else "normal",
                )

        ax.set_xticks(range(token_side))
        ax.set_yticks(range(token_side))
        ax.set_xticklabels([f"c{c}" for c in range(token_side)])
        ax.set_yticklabels([f"r{r}" for r in range(token_side)])

        ax.set_title(
            f"Waypoint {k + 1} sensitivity\n"
            f"max token = row {max_idx[0]}, col {max_idx[1]}"
        )

        ax.set_xticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    # Add one colorbar for all influence plots.
    cbar = fig.colorbar(im, ax=all_axes[1:].tolist(), shrink=0.75)
    cbar.set_label("Waypoint prediction change after masking token")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_channels(text):
    if text is None or text.strip().lower() in {"all", "none"}:
        return None

    return [int(x) for x in text.split(",") if x.strip() != ""]

def plot_masked_waypoint_overlay(
    x_np,
    y_target,
    baseline_pred,
    occluded_pred,
    token_row,
    token_col,
    token_side,
    out_path,
    background_channel,
    resolution,
    title,
):
    crop_size = x_np.shape[-1]
    bg = normalize_image(x_np[background_channel])

    num_waypoints = baseline_pred.shape[0] // 4

    target_wps = y_target.reshape(num_waypoints, 4)
    baseline_wps = baseline_pred.reshape(num_waypoints, 4)
    occluded_wps = occluded_pred.reshape(num_waypoints, 4)

    step = crop_size // token_side
    r0 = token_row * step
    r1 = (token_row + 1) * step
    c0 = token_col * step
    c1 = (token_col + 1) * step

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(bg, origin="upper", cmap="gray")

    # Draw full token grid.
    draw_token_grid(ax, crop_size=crop_size, token_side=token_side)

    # Highlight the masked token region.
    rect = plt.Rectangle(
        (c0, r0),
        c1 - c0,
        r1 - r0,
        fill=True,
        facecolor="red",
        alpha=0.25,
        edgecolor="red",
        linewidth=3,
    )
    ax.add_patch(rect)

    # Draw waypoints.
    draw_waypoints(
        ax,
        target_wps,
        crop_size=crop_size,
        resolution=resolution,
        label="target / expert",
        marker="o",
    )

    draw_waypoints(
        ax,
        baseline_wps,
        crop_size=crop_size,
        resolution=resolution,
        label="original prediction",
        marker="x",
    )

    draw_waypoints(
        ax,
        occluded_wps,
        crop_size=crop_size,
        resolution=resolution,
        label=f"prediction after masking r{token_row},c{token_col}",
        marker="s",
    )

    # Robot center + forward arrow.
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

    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

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
        help="Channel to display. Current setup: 0 occupancy, 1 cost-to-go, 10 goal mask.",
    )
    parser.add_argument(
        "--channels_to_occlude",
        type=str,
        default="all",
        help=(
            "Comma-separated channel indices to occlude, e.g. '0,1,10'. "
            "Use 'all' to occlude all channels."
        ),
    )
    parser.add_argument(
        "--occlusion_mode",
        type=str,
        default="mean",
        choices=["zero", "mean", "channel_mean", "constant"],
        help="How to replace the masked token region.",
    )
    parser.add_argument("--fill_value", type=float, default=0.0)
    parser.add_argument("--token_side", type=int, default=4)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--inspect_token",
        type=str,
        default=None,
        help="Optional token to mask and visualize as row,col. Example: --inspect_token 1,1",
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

    channels_to_occlude = parse_channels(args.channels_to_occlude)

    model, y_mean, y_std, ckpt = load_checkpoint(args.checkpoint, device)

    baseline_pred = predict_waypoints(model, x_np, y_mean, y_std, device)

    influence_xy, influence_theta, influence_total = compute_occlusion_influence(
        model=model,
        x_np=x_np,
        baseline_pred=baseline_pred,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        token_side=args.token_side,
        mode=args.occlusion_mode,
        fill_value=args.fill_value,
        channels_to_occlude=channels_to_occlude,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = (
        f"Token occlusion sensitivity | "
        f"sample {args.sample_index} | "
        f"{Path(args.checkpoint).name}"
    )

    plot_occlusion_visualization(
        x_np=x_np,
        y_target=y_target,
        baseline_pred=baseline_pred,
        influence=influence_total,
        out_path=str(out_path),
        background_channel=args.background_channel,
        resolution=args.resolution,
        title=title,
    )

    # Save raw numbers too.
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        influence_xy=influence_xy,
        influence_theta=influence_theta,
        influence_total=influence_total,
        baseline_pred=baseline_pred,
        y_target=y_target,
        sample_index=np.array(args.sample_index, dtype=np.int32),
    )

    if args.inspect_token is not None:
        token_parts = args.inspect_token.split(",")
        if len(token_parts) != 2:
            raise ValueError(
                "--inspect_token should be formatted as row,col, for example: --inspect_token 1,1"
            )

        inspect_row = int(token_parts[0])
        inspect_col = int(token_parts[1])

        if not (0 <= inspect_row < args.token_side and 0 <= inspect_col < args.token_side):
            raise ValueError(
                f"inspect token ({inspect_row}, {inspect_col}) is outside "
                f"token grid size {args.token_side}x{args.token_side}"
            )

        x_occ = make_occluded_copy(
            x_np=x_np,
            token_row=inspect_row,
            token_col=inspect_col,
            token_side=args.token_side,
            mode=args.occlusion_mode,
            fill_value=args.fill_value,
            channels_to_occlude=channels_to_occlude,
        )

        occluded_pred = predict_waypoints(
            model=model,
            x_np=x_occ,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
        )

        overlay_path = out_path.with_name(
            out_path.stem + f"_masked_r{inspect_row}_c{inspect_col}_overlay.png"
        )

        overlay_title = (
            f"Waypoint shift after masking token r{inspect_row},c{inspect_col} | "
            f"sample {args.sample_index}"
        )

        plot_masked_waypoint_overlay(
            x_np=x_np,
            y_target=y_target,
            baseline_pred=baseline_pred,
            occluded_pred=occluded_pred,
            token_row=inspect_row,
            token_col=inspect_col,
            token_side=args.token_side,
            out_path=str(overlay_path),
            background_channel=args.background_channel,
            resolution=args.resolution,
            title=overlay_title,
        )

        print("Saved:", overlay_path)

        print(f"\nWaypoint shift after masking token r{inspect_row},c{inspect_col}:")
        base_wps = baseline_pred.reshape(-1, 4)
        occ_wps = occluded_pred.reshape(-1, 4)

        for k in range(base_wps.shape[0]):
            dx = occ_wps[k, 0] - base_wps[k, 0]
            dy = occ_wps[k, 1] - base_wps[k, 1]
            dxy = math.sqrt(dx * dx + dy * dy)

            print(
                f"waypoint {k + 1}: "
                f"dx={dx:+.4f}, dy={dy:+.4f}, position_shift={dxy:.4f}"
            )

    print("Saved:", out_path)
    print("Saved:", npz_path)

    print("\nBaseline predicted waypoints:")
    print(baseline_pred.reshape(-1, 4))

    print("\nTarget waypoints:")
    print(y_target.reshape(-1, 4))

    print("\nMax influence token per waypoint:")
    for k in range(influence_total.shape[0]):
        max_idx = np.unravel_index(np.argmax(influence_total[k]), influence_total[k].shape)
        max_val = influence_total[k, max_idx[0], max_idx[1]]
        print(f"waypoint {k + 1}: row {max_idx[0]}, col {max_idx[1]}, value {max_val:.4f}")


if __name__ == "__main__":
    main()