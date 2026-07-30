#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gridnav_il.models import CNNPolicy, WaypointQueryAttentionPolicy


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError("Expected checkpoint key 'model_state_dict' or 'state_dict'.")

    model_type = ckpt.get("model_type", None)
    input_channels = int(ckpt.get("input_channels", 11))
    output_dim = int(ckpt.get("output_dim", 20))

    if model_type == "cnn":
        model = CNNPolicy(input_channels=input_channels, output_dim=output_dim)
    elif model_type == "waypoint_query_attention":
        model = WaypointQueryAttentionPolicy(
            input_channels=input_channels,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    y_mean = ckpt.get("y_mean", None)
    y_std = ckpt.get("y_std", None)

    if y_mean is not None:
        y_mean = torch.as_tensor(y_mean, dtype=torch.float32, device=device).view(1, -1)
    if y_std is not None:
        y_std = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(1, -1)

    return model, model_type, y_mean, y_std


def predict(model, x_np, y_mean, y_std, device):
    x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]

        if y_mean is not None and y_std is not None:
            y = y * y_std + y_mean

    return y[0].detach().cpu().numpy().astype(np.float32)


def make_occluded_copy(
    x_np,
    token_row,
    token_col,
    token_side,
    channels_to_occlude,
    occlusion_mode,
    fill_value,
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
        if occlusion_mode == "zero":
            x_occ[ch, r0:r1, c0:c1] = 0.0
        elif occlusion_mode == "mean":
            x_occ[ch, r0:r1, c0:c1] = float(np.mean(x_np[ch]))
        elif occlusion_mode == "constant":
            x_occ[ch, r0:r1, c0:c1] = float(fill_value)
        else:
            raise ValueError(f"Unknown occlusion_mode: {occlusion_mode}")

    return x_occ


def compute_influence_for_model(
    model,
    x_np,
    y_mean,
    y_std,
    device,
    token_side,
    channels_to_occlude,
    occlusion_mode,
    fill_value,
):
    baseline = predict(model, x_np, y_mean, y_std, device)
    num_waypoints = baseline.shape[0] // 4
    baseline_wps = baseline.reshape(num_waypoints, 4)

    influence_xy = np.zeros((num_waypoints, token_side, token_side), dtype=np.float32)
    influence_heading = np.zeros((num_waypoints, token_side, token_side), dtype=np.float32)
    influence_total = np.zeros((num_waypoints, token_side, token_side), dtype=np.float32)

    occluded_predictions = {}

    for r in range(token_side):
        for c in range(token_side):
            x_occ = make_occluded_copy(
                x_np=x_np,
                token_row=r,
                token_col=c,
                token_side=token_side,
                channels_to_occlude=channels_to_occlude,
                occlusion_mode=occlusion_mode,
                fill_value=fill_value,
            )

            pred_occ = predict(model, x_occ, y_mean, y_std, device)
            occluded_predictions[(r, c)] = pred_occ

            occ_wps = pred_occ.reshape(num_waypoints, 4)

            for k in range(num_waypoints):
                dx = occ_wps[k, 0] - baseline_wps[k, 0]
                dy = occ_wps[k, 1] - baseline_wps[k, 1]

                dxy = math.sqrt(dx * dx + dy * dy)

                dsin = occ_wps[k, 2] - baseline_wps[k, 2]
                dcos = occ_wps[k, 3] - baseline_wps[k, 3]
                dheading = math.sqrt(dsin * dsin + dcos * dcos)

                influence_xy[k, r, c] = dxy
                influence_heading[k, r, c] = dheading
                influence_total[k, r, c] = dxy + 0.25 * dheading

    return {
        "baseline": baseline,
        "influence_xy": influence_xy,
        "influence_heading": influence_heading,
        "influence_total": influence_total,
        "occluded_predictions": occluded_predictions,
    }


def normalize_image(img):
    img = img.astype(np.float32)
    finite = np.isfinite(img)

    if not np.any(finite):
        return np.zeros_like(img)

    lo = np.percentile(img[finite], 2)
    hi = np.percentile(img[finite], 98)

    if hi <= lo + 1e-6:
        return np.zeros_like(img)

    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def waypoint_to_pixel(x_local, y_local, crop_size, resolution):
    half = crop_size // 2
    row = half - x_local / resolution
    col = half - y_local / resolution
    return row, col


def draw_waypoint_sequence(ax, waypoints, crop_size, resolution, label, marker):
    rows = []
    cols = []

    for wp in waypoints:
        row, col = waypoint_to_pixel(wp[0], wp[1], crop_size, resolution)
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


def plot_model_comparison(
    x_np,
    y_target,
    cnn_result,
    wqa_result,
    out_path,
    background_channel,
    resolution,
    token_side,
    sample_index,
):
    crop_size = x_np.shape[-1]
    bg = normalize_image(x_np[background_channel])

    num_waypoints = y_target.shape[0] // 4
    target_wps = y_target.reshape(num_waypoints, 4)
    cnn_wps = cnn_result["baseline"].reshape(num_waypoints, 4)
    wqa_wps = wqa_result["baseline"].reshape(num_waypoints, 4)

    cnn_inf = cnn_result["influence_total"]
    wqa_inf = wqa_result["influence_total"]
    diff_inf = wqa_inf - cnn_inf

    fig, axes = plt.subplots(4, 3, figsize=(15, 18), squeeze=False)

    # Panel 1: input and original predictions.
    ax = axes[0, 0]
    ax.imshow(bg, origin="upper", cmap="gray")
    draw_token_grid(ax, crop_size=crop_size, token_side=token_side)
    draw_waypoint_sequence(ax, target_wps, crop_size, resolution, "target", "o")
    draw_waypoint_sequence(ax, cnn_wps, crop_size, resolution, "CNN pred", "x")
    draw_waypoint_sequence(ax, wqa_wps, crop_size, resolution, "WQA pred", "s")
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
    ax.set_title(f"Sample {sample_index}: input + baseline waypoints")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

    # Hide unused top row panels but use them for summary text.
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    cnn_max_per_wp = cnn_inf.reshape(num_waypoints, -1).max(axis=1)
    wqa_max_per_wp = wqa_inf.reshape(num_waypoints, -1).max(axis=1)

    summary_text = "Max token-occlusion influence per waypoint\n\n"
    summary_text += "wp     CNN      WQA      WQA-CNN\n"
    for k in range(num_waypoints):
        summary_text += (
            f"{k+1:<5} "
            f"{cnn_max_per_wp[k]:.3f}    "
            f"{wqa_max_per_wp[k]:.3f}    "
            f"{(wqa_max_per_wp[k] - cnn_max_per_wp[k]):+.3f}\n"
        )

    axes[0, 1].text(
        0.0,
        1.0,
        summary_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    mean_cnn = float(np.mean(cnn_inf))
    mean_wqa = float(np.mean(wqa_inf))
    max_cnn = float(np.max(cnn_inf))
    max_wqa = float(np.max(wqa_inf))

    axes[0, 2].text(
        0.0,
        1.0,
        (
            "Overall sensitivity\n\n"
            f"CNN mean: {mean_cnn:.4f}\n"
            f"WQA mean: {mean_wqa:.4f}\n\n"
            f"CNN max:  {max_cnn:.4f}\n"
            f"WQA max:  {max_wqa:.4f}\n\n"
            "Positive WQA-CNN means\n"
            "WQA changed more after masking."
        ),
        va="top",
        ha="left",
        fontsize=11,
    )

    vmax = max(float(np.max(cnn_inf)), float(np.max(wqa_inf)), 1e-6)
    diff_abs = max(float(np.max(np.abs(diff_inf))), 1e-6)

    # Rows:
    # row 1: waypoint 1,2,3
    # row 2: waypoint 4,5,CNN-vs-WQA total
    # row 3: average maps
    plot_slots = [
        (1, 0, 0),
        (1, 1, 1),
        (1, 2, 2),
        (2, 0, 3),
        (2, 1, 4),
    ]

    for row, col, k in plot_slots:
        ax = axes[row, col]
        ax.imshow(wqa_inf[k], origin="upper", cmap="magma", vmin=0.0, vmax=vmax)

        max_idx = np.unravel_index(np.argmax(wqa_inf[k]), wqa_inf[k].shape)

        for r in range(token_side):
            for c in range(token_side):
                ax.text(
                    c,
                    r,
                    f"{wqa_inf[k, r, c]:.3f}",
                    ha="center",
                    va="center",
                    color="cyan" if (r, c) == max_idx else "white",
                    fontsize=9,
                    fontweight="bold" if (r, c) == max_idx else "normal",
                )

        ax.set_title(f"WQA influence: waypoint {k+1}\nmax r{max_idx[0]},c{max_idx[1]}")
        ax.set_xticks(range(token_side))
        ax.set_yticks(range(token_side))
        ax.set_xticklabels([f"c{i}" for i in range(token_side)])
        ax.set_yticklabels([f"r{i}" for i in range(token_side)])
        ax.set_xticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Difference map averaged across waypoints.
    ax = axes[2, 2]
    diff_mean = np.mean(diff_inf, axis=0)
    ax.imshow(diff_mean, origin="upper", cmap="coolwarm", vmin=-diff_abs, vmax=diff_abs)
    for r in range(token_side):
        for c in range(token_side):
            ax.text(
                c,
                r,
                f"{diff_mean[r, c]:+.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    ax.set_title("Mean influence difference\nWQA - CNN")
    ax.set_xticks(range(token_side))
    ax.set_yticks(range(token_side))
    ax.set_xticklabels([f"c{i}" for i in range(token_side)])
    ax.set_yticklabels([f"r{i}" for i in range(token_side)])
    ax.set_xticks(np.arange(-0.5, token_side, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, token_side, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Average CNN/WQA/Diff maps.
    maps = [
        ("CNN mean influence", np.mean(cnn_inf, axis=0), "magma", 0.0, vmax),
        ("WQA mean influence", np.mean(wqa_inf, axis=0), "magma", 0.0, vmax),
        ("WQA - CNN mean influence", diff_mean, "coolwarm", -diff_abs, diff_abs),
    ]

    for col, (name, grid, cmap, vmin, vmax_local) in enumerate(maps):
        ax = axes[3, col]
        ax.imshow(grid, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax_local)

        max_idx = np.unravel_index(np.argmax(np.abs(grid)), grid.shape)

        for r in range(token_side):
            for c in range(token_side):
                ax.text(
                    c,
                    r,
                    f"{grid[r, c]:+.3f}" if "WQA - CNN" in name else f"{grid[r, c]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )

        ax.set_title(name)
        ax.set_xticks(range(token_side))
        ax.set_yticks(range(token_side))
        ax.set_xticklabels([f"c{i}" for i in range(token_side)])
        ax.set_yticklabels([f"r{i}" for i in range(token_side)])
        ax.set_xticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, token_side, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle("CNN vs WQA token occlusion comparison", fontsize=15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_csv(out_csv, cnn_result, wqa_result, sample_index):
    cnn_inf = cnn_result["influence_total"]
    wqa_inf = wqa_result["influence_total"]

    num_waypoints, token_side, _ = cnn_inf.shape

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "waypoint",
                "token_row",
                "token_col",
                "cnn_influence",
                "wqa_influence",
                "wqa_minus_cnn",
            ],
        )
        writer.writeheader()

        for k in range(num_waypoints):
            for r in range(token_side):
                for c in range(token_side):
                    writer.writerow(
                        {
                            "sample_index": sample_index,
                            "waypoint": k + 1,
                            "token_row": r,
                            "token_col": c,
                            "cnn_influence": float(cnn_inf[k, r, c]),
                            "wqa_influence": float(wqa_inf[k, r, c]),
                            "wqa_minus_cnn": float(wqa_inf[k, r, c] - cnn_inf[k, r, c]),
                        }
                    )


def parse_channels(text):
    if text is None or text.strip().lower() in {"all", "none"}:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cnn_checkpoint", type=str, required=True)
    parser.add_argument("--wqa_checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--background_channel", type=int, default=1)
    parser.add_argument("--channels_to_occlude", type=str, default="all")
    parser.add_argument(
        "--occlusion_mode",
        type=str,
        default="mean",
        choices=["zero", "mean", "constant"],
    )
    parser.add_argument("--fill_value", type=float, default=0.0)
    parser.add_argument("--token_side", type=int, default=4)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    channels_to_occlude = parse_channels(args.channels_to_occlude)

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]

    if args.sample_index < 0 or args.sample_index >= len(X):
        raise IndexError(
            f"sample_index {args.sample_index} out of range for dataset with {len(X)} samples"
        )

    x_np = X[args.sample_index].astype(np.float32)
    y_target = Y[args.sample_index].astype(np.float32)

    cnn_model, cnn_type, cnn_y_mean, cnn_y_std = load_model(args.cnn_checkpoint, device)
    wqa_model, wqa_type, wqa_y_mean, wqa_y_std = load_model(args.wqa_checkpoint, device)

    if cnn_type != "cnn":
        raise ValueError(f"--cnn_checkpoint should be model_type cnn, got {cnn_type}")
    if wqa_type != "waypoint_query_attention":
        raise ValueError(
            f"--wqa_checkpoint should be model_type waypoint_query_attention, got {wqa_type}"
        )

    cnn_result = compute_influence_for_model(
        model=cnn_model,
        x_np=x_np,
        y_mean=cnn_y_mean,
        y_std=cnn_y_std,
        device=device,
        token_side=args.token_side,
        channels_to_occlude=channels_to_occlude,
        occlusion_mode=args.occlusion_mode,
        fill_value=args.fill_value,
    )

    wqa_result = compute_influence_for_model(
        model=wqa_model,
        x_np=x_np,
        y_mean=wqa_y_mean,
        y_std=wqa_y_std,
        device=device,
        token_side=args.token_side,
        channels_to_occlude=channels_to_occlude,
        occlusion_mode=args.occlusion_mode,
        fill_value=args.fill_value,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(
        x_np=x_np,
        y_target=y_target,
        cnn_result=cnn_result,
        wqa_result=wqa_result,
        out_path=str(out_path),
        background_channel=args.background_channel,
        resolution=args.resolution,
        token_side=args.token_side,
        sample_index=args.sample_index,
    )

    csv_path = out_path.with_suffix(".csv")
    npz_path = out_path.with_suffix(".npz")

    write_csv(csv_path, cnn_result, wqa_result, args.sample_index)

    np.savez_compressed(
        npz_path,
        cnn_influence_total=cnn_result["influence_total"],
        wqa_influence_total=wqa_result["influence_total"],
        cnn_baseline=cnn_result["baseline"],
        wqa_baseline=wqa_result["baseline"],
        y_target=y_target,
        sample_index=np.array(args.sample_index, dtype=np.int32),
    )

    print("Saved:", out_path)
    print("Saved:", csv_path)
    print("Saved:", npz_path)

    print("\nMax influence per waypoint:")
    for name, result in [("CNN", cnn_result), ("WQA", wqa_result)]:
        inf = result["influence_total"]
        print(f"\n{name}")
        for k in range(inf.shape[0]):
            max_idx = np.unravel_index(np.argmax(inf[k]), inf[k].shape)
            max_val = inf[k, max_idx[0], max_idx[1]]
            print(
                f"  waypoint {k + 1}: "
                f"row {max_idx[0]}, col {max_idx[1]}, value {max_val:.4f}"
            )


if __name__ == "__main__":
    main()