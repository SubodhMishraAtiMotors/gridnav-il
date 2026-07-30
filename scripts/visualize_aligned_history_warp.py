import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from gridnav_il.geometry import ControlLimits, SimConfig, states_to_array
from gridnav_il.controllers import (
    PurePursuitConfig,
    PurePursuitController,
    rollout_nav_controller,
    make_nav_context,
    make_start_state_from_problem,
)
from gridnav_il.grid_world import create_valid_planning_problem, sample_problem_kwargs
from gridnav_il.observations import (
    LocalObservationConfig,
    extract_local_grid_observation,
)


def pose_to_array(state):
    return np.array([state.x, state.y, state.theta], dtype=np.float32)


def warp_old_crop_to_current_frame(
    old_crop,
    old_pose,
    current_pose,
    resolution,
    fill_value=0.0,
):
    """
    Warp an old robot-frame crop into the current robot frame.

    old_crop:
        [C, H, W]

    old_pose/current_pose:
        [x_world, y_world, theta_world]

    Output:
        [C, H, W], expressed in current robot-frame crop coordinates.

    Convention from observations.py:
        image up   = robot forward
        image left = robot left
        x_body     = forward
        y_body     = left
    """
    channels, height, width = old_crop.shape

    if height != width:
        raise ValueError("Expected square crops.")

    crop_size = height
    half = crop_size // 2

    rows, cols = np.meshgrid(
        np.arange(crop_size),
        np.arange(crop_size),
        indexing="ij",
    )

    # Output pixel coordinates interpreted in CURRENT robot frame.
    x_cur = (half - rows).astype(np.float32) * resolution
    y_cur = (half - cols).astype(np.float32) * resolution

    cur_x, cur_y, cur_theta = current_pose
    old_x, old_y, old_theta = old_pose

    c_cur = np.cos(cur_theta)
    s_cur = np.sin(cur_theta)

    # Current robot frame -> world frame.
    x_world = cur_x + c_cur * x_cur - s_cur * y_cur
    y_world = cur_y + s_cur * x_cur + c_cur * y_cur

    dx = x_world - old_x
    dy = y_world - old_y

    c_old = np.cos(old_theta)
    s_old = np.sin(old_theta)

    # World frame -> OLD robot frame.
    x_old = c_old * dx + s_old * dy
    y_old = -s_old * dx + c_old * dy

    # OLD robot frame -> old crop image coordinates.
    old_rows = half - x_old / resolution
    old_cols = half - y_old / resolution

    warped_channels = []

    for c in range(channels):
        warped = map_coordinates(
            old_crop[c].astype(np.float32),
            coordinates=[old_rows, old_cols],
            order=0,
            mode="constant",
            cval=float(fill_value),
        )
        warped_channels.append(warped.astype(np.float32))

    return np.stack(warped_channels, axis=0)


def make_demo(seed, args, obs_config):
    rng = np.random.default_rng(seed)

    problem_kwargs = sample_problem_kwargs(
        seed=int(rng.integers(0, 1_000_000_000))
    )

    problem = create_valid_planning_problem(
        **problem_kwargs,
        seed=int(rng.integers(0, 1_000_000_000)),
        planner_connectivity=args.planner_connectivity,
    )

    nav_context = make_nav_context(problem)
    start_state = make_start_state_from_problem(problem)

    controller = PurePursuitController(
        config=PurePursuitConfig(
            lookahead_distance=args.lookahead_distance,
            desired_speed=args.desired_speed,
            goal_slowdown_distance=args.goal_slowdown_distance,
        ),
        limits=ControlLimits(
            v_max=args.v_max,
            omega_max=args.omega_max,
        ),
    )

    states, controls = rollout_nav_controller(
        start_state=start_state,
        controller=controller,
        nav_context=nav_context,
        sim_config=SimConfig(
            dt=args.dt,
            max_steps=args.max_steps,
            goal_position_tol=args.goal_position_tol,
        ),
    )

    return problem, nav_context, states, controls


def plot_warp_debug(
    raw_crops,
    warped_crops,
    state_indices,
    channel_index,
    out_path,
):
    num_history = len(raw_crops)

    fig, axes = plt.subplots(
        2,
        num_history,
        figsize=(4 * num_history, 8),
        squeeze=False,
    )

    for j in range(num_history):
        raw = raw_crops[j][channel_index]
        warped = warped_crops[j][channel_index]

        axes[0, j].imshow(raw, origin="upper", cmap="gray")
        axes[0, j].set_title(f"Raw crop\nstep {state_indices[j]}")
        axes[0, j].axis("off")

        axes[1, j].imshow(warped, origin="upper", cmap="gray")
        axes[1, j].set_title(f"Warped to current\nstep {state_indices[-1]}")
        axes[1, j].axis("off")

    fig.suptitle(
        f"SE(2)-aligned history warp | channel {channel_index}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overlay_debug(
    current_crop,
    warped_crops,
    channel_index,
    out_path,
):
    plt.figure(figsize=(7, 7))

    # Current observation in grayscale.
    plt.imshow(
        current_crop[channel_index],
        origin="upper",
        cmap="gray",
        alpha=1.0,
    )

    # Overlay older warped crops as contours.
    for i, crop in enumerate(warped_crops[:-1]):
        channel = crop[channel_index]
        plt.contour(
            channel,
            levels=[0.5],
            linewidths=1.0,
            alpha=0.7,
        )

    plt.title(
        f"Current crop with warped history contours | channel {channel_index}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_merged_occupancy_debug(
    current_crop,
    warped_crops,
    channel_index,
    out_path,
):
    current_occ = current_crop[channel_index]

    # Merge occupancy memory.
    # Since occupancy is 0/1, max means:
    # if any aligned history frame saw obstacle, keep it.
    warped_stack = np.stack(
        [crop[channel_index] for crop in warped_crops],
        axis=0,
    )

    merged_occ = np.max(warped_stack, axis=0)

    # What history adds beyond current observation.
    added_memory = np.clip(merged_occ - current_occ, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

    axes[0, 0].imshow(current_occ, origin="upper", cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Current occupancy")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(merged_occ, origin="upper", cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Merged aligned occupancy history")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(added_memory, origin="upper", cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("History-only added occupancy")
    axes[0, 2].axis("off")

    # Mark robot center.
    h, w = current_occ.shape
    for ax in axes[0]:
        ax.plot(w // 2, h // 2, "rx", markersize=8, markeredgewidth=2)
        ax.arrow(
            w // 2,
            h // 2,
            0,
            -8,
            color="red",
            width=0.8,
            head_width=3,
            length_includes_head=True,
        )

    fig.suptitle("Merged SE(2)-aligned occupancy memory", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SE(2)-aligned local crop history warping."
    )

    parser.add_argument("--out_dir", type=str, default="outputs/warp_debug")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--current_step", type=int, default=120)
    parser.add_argument("--history_stride", type=int, default=5)

    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--channel_index", type=int, default=0)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--goal_position_tol", type=float, default=0.25)

    parser.add_argument("--lookahead_distance", type=float, default=0.4)
    parser.add_argument("--desired_speed", type=float, default=0.4)
    parser.add_argument("--goal_slowdown_distance", type=float, default=0.8)

    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--omega_max", type=float, default=2.0)

    parser.add_argument("--planner_connectivity", type=int, default=8, choices=[4, 8])

    parser.add_argument("--max_clearance_m", type=float, default=2.0)
    parser.add_argument("--cost_to_go_normalization", type=str, default="local")

    parser.add_argument("--include_goal_mask", action="store_true")
    parser.add_argument("--include_occupancy_channel", action="store_true")
    parser.add_argument("--include_cost_to_go_encoding", action="store_true")
    parser.add_argument("--cost_encoding_frequencies", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])

    parser.add_argument("--no_traversability_channel", action="store_true")
    parser.add_argument("--no_clearance_channel", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    obs_config = LocalObservationConfig(
        crop_size_cells=args.crop_size,
        normalize_cost=True,
        unknown_cost_value=1.0,
        include_traversability_channel=not args.no_traversability_channel,
        include_occupancy_channel=args.include_occupancy_channel,
        include_clearance_channel=not args.no_clearance_channel,
        max_clearance_m=args.max_clearance_m,
        include_goal_mask_channel=args.include_goal_mask,
        goal_mask_sigma_cells=2.0,
        include_cost_to_go_encoding=args.include_cost_to_go_encoding,
        cost_encoding_frequencies=tuple(args.cost_encoding_frequencies),
        cost_to_go_normalization=args.cost_to_go_normalization,
    )

    problem, nav_context, states, controls = make_demo(
        seed=args.seed,
        args=args,
        obs_config=obs_config,
    )

    if len(states) < args.current_step + 1:
        raise ValueError(
            f"Demo has only {len(states)} states, but current_step={args.current_step}."
        )

    current_step = args.current_step

    state_indices = [
        max(0, current_step - (args.history_length - 1 - i) * args.history_stride)
        for i in range(args.history_length)
    ]

    raw_crops = []
    warped_crops = []

    current_pose = pose_to_array(states[current_step])

    for idx in state_indices:
        crop = extract_local_grid_observation(
            robot_state=states[idx],
            nav_context=nav_context,
            obs_config=obs_config,
        )

        old_pose = pose_to_array(states[idx])

        warped = warp_old_crop_to_current_frame(
            old_crop=crop,
            old_pose=old_pose,
            current_pose=current_pose,
            resolution=nav_context.resolution,
            fill_value=1.0,
        )

        raw_crops.append(crop)
        warped_crops.append(warped)

    grid_out_path = os.path.join(
        args.out_dir,
        f"warp_grid_step_{current_step:04d}_ch_{args.channel_index}.png",
    )

    overlay_out_path = os.path.join(
        args.out_dir,
        f"warp_overlay_step_{current_step:04d}_ch_{args.channel_index}.png",
    )

    merged_out_path = os.path.join(
        args.out_dir,
        f"merged_occupancy_step_{current_step:04d}_ch_{args.channel_index}.png",
    )

    plot_warp_debug(
        raw_crops=raw_crops,
        warped_crops=warped_crops,
        state_indices=state_indices,
        channel_index=args.channel_index,
        out_path=grid_out_path,
    )

    plot_overlay_debug(
        current_crop=raw_crops[-1],
        warped_crops=warped_crops,
        channel_index=args.channel_index,
        out_path=overlay_out_path,
    )

    plot_merged_occupancy_debug(
        current_crop=raw_crops[-1],
        warped_crops=warped_crops,
        channel_index=args.channel_index,
        out_path=merged_out_path,
    )

    print("Saved:", grid_out_path)
    print("Saved:", overlay_out_path)
    print("Saved:", merged_out_path)
    print("State indices:", state_indices)
    print("Current step:", current_step)
    print("Observation shape:", raw_crops[-1].shape)


if __name__ == "__main__":
    main()