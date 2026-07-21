import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize CNN input channels from a gridnav-il dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to .npz dataset.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/cnn_input_tiles",
        help="Output folder for individual visualization images.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of random samples to visualize.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --indices is not provided.",
    )

    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific dataset indices to visualize.",
    )

    parser.add_argument(
        "--channel_order",
        type=str,
        default="occupancy,cost_to_go,goal_mask",
        help=(
            "Comma-separated names describing dataset channel order. "
            "For current occupancy experiment use: occupancy,cost_to_go,goal_mask"
        ),
    )

    parser.add_argument(
        "--display_order",
        type=str,
        default="cost_to_go,occupancy,goal_mask",
        help="Comma-separated channel names for plotting order.",
    )

    parser.add_argument(
        "--draw_robot",
        action="store_true",
        help="Draw robot and robot-frame axes at the crop center.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )

    parser.add_argument(
        "--goal_visible_only",
        action="store_true",
        help="Only visualize samples where the goal mask is visible in the local crop.",
    )

    parser.add_argument(
        "--goal_mask_threshold",
        type=float,
        default=0.2,
        help="Minimum max value of goal mask for a sample to be considered goal-visible.",
    )

    return parser.parse_args()


def get_channel_display_settings(channel_name):
    if channel_name == "occupancy":
        return {
            "title": "Occupancy",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "gray_r",
        }

    if channel_name == "cost_to_go":
        return {
            "title": "Cost-to-go",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "viridis",
        }

    if channel_name == "goal_mask":
        return {
            "title": "Goal mask",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "magma",
        }

    if channel_name == "clearance":
        return {
            "title": "Clearance",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "viridis",
        }

    if channel_name == "traversability":
        return {
            "title": "Traversability",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "viridis",
        }

    return {
        "title": channel_name,
        "vmin": None,
        "vmax": None,
        "cmap": "viridis",
    }


def draw_robot_and_axes(ax, image_shape):
    height, width = image_shape

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    robot_radius = 2.0
    axis_len = 9.0

    # Robot body.
    body = Circle(
        (cx, cy),
        radius=robot_radius,
        fill=False,
        linewidth=1.8,
    )
    ax.add_patch(body)

    # In robot-frame crop:
    # image up is robot +x / forward.
    # image left is robot +y / left.
    forward_arrow = FancyArrow(
        cx,
        cy,
        0.0,
        -axis_len,
        width=0.4,
        head_width=2.0,
        head_length=2.5,
        length_includes_head=True,
    )
    ax.add_patch(forward_arrow)

    left_arrow = FancyArrow(
        cx,
        cy,
        -axis_len,
        0.0,
        width=0.4,
        head_width=2.0,
        head_length=2.5,
        length_includes_head=True,
    )
    ax.add_patch(left_arrow)

    ax.text(
        cx + 1.5,
        cy - axis_len - 2.0,
        "+x",
        fontsize=9,
        ha="left",
        va="center",
    )

    ax.text(
        cx - axis_len - 2.0,
        cy + 2.0,
        "+y",
        fontsize=9,
        ha="center",
        va="bottom",
    )


def save_one_sample(
    X,
    Y,
    sample_idx,
    channel_to_index,
    display_order,
    out_path,
    draw_robot,
    dpi,
):
    sample = X[sample_idx].astype(np.float32)

    num_cols = len(display_order)

    fig, axes = plt.subplots(
        1,
        num_cols,
        figsize=(4.0 * num_cols, 4.2),
        squeeze=False,
    )

    axes = axes[0]

    for col_idx, channel_name in enumerate(display_order):
        ax = axes[col_idx]

        channel_idx = channel_to_index[channel_name]
        image = sample[channel_idx]

        settings = get_channel_display_settings(channel_name)

        ax.imshow(
            image,
            origin="upper",
            cmap=settings["cmap"],
            vmin=settings["vmin"],
            vmax=settings["vmax"],
        )

        ax.set_title(settings["title"], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        if draw_robot:
            draw_robot_and_axes(ax, image.shape)

    if Y is not None:
        v, omega = Y[sample_idx]
        title = (
            f"CNN input in robot frame | idx={sample_idx} | "
            f"expert v={v:.3f}, omega={omega:.3f}"
        )
    else:
        title = f"CNN input in robot frame | idx={sample_idx}"

    fig.suptitle(
        title + "\ncenter = robot, +x = forward/up, +y = left",
        fontsize=12,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    Y = data["Y"] if "Y" in data.files else None

    channel_order = [c.strip() for c in args.channel_order.split(",")]
    display_order = [c.strip() for c in args.display_order.split(",")]

    if X.ndim != 4:
        raise ValueError(f"Expected X shape [N, C, H, W], got {X.shape}")

    if len(channel_order) != X.shape[1]:
        raise ValueError(
            f"channel_order has {len(channel_order)} entries, "
            f"but dataset has {X.shape[1]} channels. "
            f"X shape = {X.shape}"
        )

    channel_to_index = {
        name: idx for idx, name in enumerate(channel_order)
    }

    for name in display_order:
        if name not in channel_to_index:
            raise ValueError(
                f"Requested display channel '{name}' not found in channel_order: "
                f"{channel_order}"
            )

    if args.indices is not None and len(args.indices) > 0:
        indices = np.array(args.indices, dtype=np.int64)
    else:
        candidate_indices = np.arange(X.shape[0])

        if args.goal_visible_only:
            if "goal_mask" not in channel_to_index:
                raise ValueError(
                    "--goal_visible_only requires 'goal_mask' to be present in channel_order."
                )

            goal_mask_channel_idx = channel_to_index["goal_mask"]
            goal_mask_max = X[:, goal_mask_channel_idx].astype(np.float32).max(axis=(1, 2))

            candidate_indices = candidate_indices[
                goal_mask_max >= args.goal_mask_threshold
            ]

            if len(candidate_indices) == 0:
                raise ValueError(
                    "No samples found with visible goal mask. "
                    f"Try lowering --goal_mask_threshold below {args.goal_mask_threshold}."
                )

            print(
                f"Found {len(candidate_indices)} samples with "
                f"goal_mask.max >= {args.goal_mask_threshold}"
            )

        rng = np.random.default_rng(args.seed)
        num_samples = min(args.num_samples, len(candidate_indices))
        indices = rng.choice(candidate_indices, size=num_samples, replace=False)

    for sample_idx in indices:
        out_path = os.path.join(
            args.out_dir,
            f"cnn_input_idx_{int(sample_idx):06d}.png",
        )

        save_one_sample(
            X=X,
            Y=Y,
            sample_idx=int(sample_idx),
            channel_to_index=channel_to_index,
            display_order=display_order,
            out_path=out_path,
            draw_robot=args.draw_robot,
            dpi=args.dpi,
        )

    print("Saved folder:", args.out_dir)
    print("Dataset:", args.dataset)
    print("X shape:", X.shape)
    print("Displayed indices:", indices.tolist())
    print("Dataset channel order:", channel_order)
    print("Display order:", display_order)
    print("Robot drawn:", args.draw_robot)


if __name__ == "__main__":
    main()
