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

    influence_total = np.zeros((num_waypoints, token_side, token_side), dtype=np.float32)

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
            occ_wps = pred_occ.reshape(num_waypoints, 4)

            for k in range(num_waypoints):
                dx = occ_wps[k, 0] - baseline_wps[k, 0]
                dy = occ_wps[k, 1] - baseline_wps[k, 1]
                dxy = math.sqrt(dx * dx + dy * dy)

                dsin = occ_wps[k, 2] - baseline_wps[k, 2]
                dcos = occ_wps[k, 3] - baseline_wps[k, 3]
                dheading = math.sqrt(dsin * dsin + dcos * dcos)

                influence_total[k, r, c] = dxy + 0.25 * dheading

    return influence_total


def parse_channels(text):
    if text is None or text.strip().lower() in {"all", "none"}:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def max_token_info(influence_for_waypoint):
    max_idx = np.unravel_index(
        np.argmax(influence_for_waypoint),
        influence_for_waypoint.shape,
    )
    max_val = float(influence_for_waypoint[max_idx[0], max_idx[1]])
    return int(max_idx[0]), int(max_idx[1]), max_val


def safe_ratio(a, b):
    if abs(b) < 1e-8:
        return float("nan")
    return float(a / b)


def write_summary_csv(rows, out_csv):
    fieldnames = [
        "sample_index",
        "waypoint",
        "cnn_max_row",
        "cnn_max_col",
        "cnn_max_influence",
        "wqa_max_row",
        "wqa_max_col",
        "wqa_max_influence",
        "wqa_minus_cnn",
        "wqa_div_cnn",
        "same_max_token",
        "wqa_greater_than_cnn",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, out_path):
    waypoints = sorted(set(row["waypoint"] for row in rows))

    cnn_means = []
    wqa_means = []
    cnn_stds = []
    wqa_stds = []

    for wp in waypoints:
        cnn_vals = np.array(
            [row["cnn_max_influence"] for row in rows if row["waypoint"] == wp],
            dtype=np.float32,
        )
        wqa_vals = np.array(
            [row["wqa_max_influence"] for row in rows if row["waypoint"] == wp],
            dtype=np.float32,
        )

        cnn_means.append(float(np.mean(cnn_vals)))
        wqa_means.append(float(np.mean(wqa_vals)))
        cnn_stds.append(float(np.std(cnn_vals)))
        wqa_stds.append(float(np.std(wqa_vals)))

    x = np.arange(len(waypoints))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        cnn_means,
        width,
        yerr=cnn_stds,
        label="CNN",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        wqa_means,
        width,
        yerr=wqa_stds,
        label="WQA",
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"wp{wp}" for wp in waypoints])
    ax.set_ylabel("Max token-occlusion influence")
    ax.set_title("Mean max occlusion influence per waypoint")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_aggregate(rows):
    waypoints = sorted(set(row["waypoint"] for row in rows))
    sample_indices = sorted(set(row["sample_index"] for row in rows))

    print("\n================================================================================")
    print("Occlusion window summary")
    print("================================================================================")
    print(f"Num samples: {len(sample_indices)}")
    print(f"Samples: {sample_indices[0]} to {sample_indices[-1]}")

    print("\nMean max influence per waypoint:")
    print("wp    CNN_mean    WQA_mean    WQA-CNN    WQA>CNN_frac    same_token_frac")

    for wp in waypoints:
        wp_rows = [row for row in rows if row["waypoint"] == wp]

        cnn_vals = np.array([r["cnn_max_influence"] for r in wp_rows], dtype=np.float32)
        wqa_vals = np.array([r["wqa_max_influence"] for r in wp_rows], dtype=np.float32)

        wqa_gt = np.mean([r["wqa_greater_than_cnn"] for r in wp_rows])
        same = np.mean([r["same_max_token"] for r in wp_rows])

        print(
            f"{wp:<5} "
            f"{np.mean(cnn_vals):.4f}      "
            f"{np.mean(wqa_vals):.4f}      "
            f"{(np.mean(wqa_vals) - np.mean(cnn_vals)):+.4f}      "
            f"{wqa_gt:.2f}            "
            f"{same:.2f}"
        )

    # Horizon ratio wp5 / wp1 per sample.
    ratios_cnn = []
    ratios_wqa = []

    for sample in sample_indices:
        s_rows = [r for r in rows if r["sample_index"] == sample]

        wp1 = [r for r in s_rows if r["waypoint"] == 1][0]
        wp5 = [r for r in s_rows if r["waypoint"] == 5][0]

        ratios_cnn.append(
            safe_ratio(wp5["cnn_max_influence"], wp1["cnn_max_influence"])
        )
        ratios_wqa.append(
            safe_ratio(wp5["wqa_max_influence"], wp1["wqa_max_influence"])
        )

    ratios_cnn = np.array(ratios_cnn, dtype=np.float32)
    ratios_wqa = np.array(ratios_wqa, dtype=np.float32)

    print("\nFar/near sensitivity ratio, wp5_max / wp1_max:")
    print(f"CNN mean ratio: {np.nanmean(ratios_cnn):.3f}")
    print(f"WQA mean ratio: {np.nanmean(ratios_wqa):.3f}")
    print(f"WQA ratio > CNN ratio fraction: {np.nanmean(ratios_wqa > ratios_cnn):.2f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cnn_checkpoint", type=str, required=True)
    parser.add_argument("--wqa_checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--end_index", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--channels_to_occlude", type=str, default="all")
    parser.add_argument(
        "--occlusion_mode",
        type=str,
        default="mean",
        choices=["zero", "mean", "constant"],
    )
    parser.add_argument("--fill_value", type=float, default=0.0)
    parser.add_argument("--token_side", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    if args.end_index < args.start_index:
        raise ValueError("--end_index must be >= --start_index")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    channels_to_occlude = parse_channels(args.channels_to_occlude)

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]

    if args.start_index < 0 or args.end_index >= len(X):
        raise IndexError(
            f"Requested range [{args.start_index}, {args.end_index}] "
            f"outside dataset with {len(X)} samples"
        )

    cnn_model, cnn_type, cnn_y_mean, cnn_y_std = load_model(args.cnn_checkpoint, device)
    wqa_model, wqa_type, wqa_y_mean, wqa_y_std = load_model(args.wqa_checkpoint, device)

    if cnn_type != "cnn":
        raise ValueError(f"--cnn_checkpoint should be model_type cnn, got {cnn_type}")
    if wqa_type != "waypoint_query_attention":
        raise ValueError(
            f"--wqa_checkpoint should be model_type waypoint_query_attention, got {wqa_type}"
        )

    rows = []

    for sample_index in range(args.start_index, args.end_index + 1):
        print(f"Processing sample {sample_index}")

        x_np = X[sample_index].astype(np.float32)

        cnn_inf = compute_influence_for_model(
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

        wqa_inf = compute_influence_for_model(
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

        num_waypoints = cnn_inf.shape[0]

        for k in range(num_waypoints):
            cnn_r, cnn_c, cnn_val = max_token_info(cnn_inf[k])
            wqa_r, wqa_c, wqa_val = max_token_info(wqa_inf[k])

            rows.append(
                {
                    "sample_index": sample_index,
                    "waypoint": k + 1,
                    "cnn_max_row": cnn_r,
                    "cnn_max_col": cnn_c,
                    "cnn_max_influence": cnn_val,
                    "wqa_max_row": wqa_r,
                    "wqa_max_col": wqa_c,
                    "wqa_max_influence": wqa_val,
                    "wqa_minus_cnn": wqa_val - cnn_val,
                    "wqa_div_cnn": safe_ratio(wqa_val, cnn_val),
                    "same_max_token": int(cnn_r == wqa_r and cnn_c == wqa_c),
                    "wqa_greater_than_cnn": int(wqa_val > cnn_val),
                }
            )

    out_csv = out_dir / "occlusion_window_summary.csv"
    out_png = out_dir / "occlusion_window_summary.png"

    write_summary_csv(rows, out_csv)
    plot_summary(rows, out_png)
    print_aggregate(rows)

    print("\nSaved:", out_csv)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()