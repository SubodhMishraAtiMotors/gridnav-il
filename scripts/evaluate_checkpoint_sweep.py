import argparse
import os
import json
import glob
import csv

import torch

from gridnav_il.geometry import ControlLimits, SimConfig
from gridnav_il.controllers import PurePursuitConfig
from gridnav_il.observations import LocalObservationConfig
from gridnav_il.models import CNNPolicy
from gridnav_il.evaluation import (
    run_closed_loop_test_suite,
    summarize_closed_loop_test_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple saved checkpoints from one training run."
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing best.pt, final.pt, and checkpoint_epoch_*.pt",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where sweep results will be saved.",
    )

    parser.add_argument("--num_tests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2000)

    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument(
        "--cost_to_go_normalization",
        type=str,
        default="local",
        choices=["local", "global", "none"],
    )
    parser.add_argument("--include_goal_mask", action="store_true")
    parser.add_argument("--max_clearance_m", type=float, default=2.0)
    parser.add_argument("--goal_mask_sigma_cells", type=float, default=2.0)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--goal_position_tol", type=float, default=0.25)

    parser.add_argument("--lookahead_distance", type=float, default=0.4)
    parser.add_argument("--desired_speed", type=float, default=0.4)
    parser.add_argument("--goal_slowdown_distance", type=float, default=0.8)

    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--omega_max", type=float, default=2.0)

    parser.add_argument(
        "--no_traversability_channel",
        action="store_true",
        help="Remove traversability channel from local observation.",
    )

    parser.add_argument(
        "--no_clearance_channel",
        action="store_true",
        help="Remove clearance channel from local observation.",
    )

    parser.add_argument(
        "--planner_connectivity",
        type=int,
        default=4,
        choices=[4, 8],
        help="Planner connectivity for Dijkstra and path extraction during evaluation.",
    )

    parser.add_argument(
        "--include_occupancy_channel",
        action="store_true",
        help="Include raw occupancy grid channel in local observation.",
    )

    parser.add_argument(
        "--include_cost_to_go_encoding",
        action="store_true",
        help="Add sinusoidal encoding channels for normalized cost-to-go.",
    )

    parser.add_argument(
        "--cost_encoding_frequencies",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0],
        help="Frequencies used for sinusoidal cost-to-go encoding.",
    )

    return parser.parse_args()


def checkpoint_sort_key(path):
    name = os.path.basename(path)

    if name == "best.pt":
        return (10_000_000, name)

    if name == "final.pt":
        return (10_000_001, name)

    if name.startswith("checkpoint_epoch_") and name.endswith(".pt"):
        epoch_str = name.replace("checkpoint_epoch_", "").replace(".pt", "")
        try:
            return (int(epoch_str), name)
        except ValueError:
            return (9_000_000, name)

    return (9_999_999, name)


def find_checkpoints(checkpoint_dir):
    paths = []

    paths.extend(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")))

    best_path = os.path.join(checkpoint_dir, "best.pt")
    final_path = os.path.join(checkpoint_dir, "final.pt")

    if os.path.exists(best_path):
        paths.append(best_path)

    if os.path.exists(final_path):
        paths.append(final_path)

    paths = sorted(paths, key=checkpoint_sort_key)

    if len(paths) == 0:
        raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")

    return paths


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = CNNPolicy(
        input_channels=checkpoint["input_channels"],
        output_dim=checkpoint["output_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_mean = checkpoint["y_mean"].to(device)
    y_std = checkpoint["y_std"].to(device)

    return model, y_mean, y_std, checkpoint


def flatten_summary(checkpoint_name, checkpoint_meta, summary):
    nn = summary["nn"]
    pp = summary["pp"]
    ratio = summary.get("nn_vs_pp", {})

    return {
        "checkpoint": checkpoint_name,
        "epoch": checkpoint_meta.get("epoch", None),
        "best_epoch_in_file": checkpoint_meta.get("best_epoch", None),
        "best_val_loss_in_file": checkpoint_meta.get("best_val_loss", None),

        "nn_success_rate": nn["success_rate"],
        "nn_collision_rate": nn["collision_rate"],
        "nn_success_no_collision_rate": nn["success_no_collision_rate"],
        "nn_success_with_collision_rate": nn["success_with_collision_rate"],
        "nn_failure_with_collision_rate": nn["failure_with_collision_rate"],
        "nn_failure_without_collision_rate": nn["failure_without_collision_rate"],

        "nn_mean_final_position_error": nn["mean_final_position_error"],
        "nn_median_final_position_error": nn["median_final_position_error"],
        "nn_mean_min_goal_distance": nn["mean_min_goal_distance_during_rollout"],

        "nn_overshoot_rate": nn["overshoot_goal_region_rate"],
        "nn_mean_steps": nn["mean_steps"],
        "nn_mean_min_clearance_m": nn["mean_min_clearance_m"],

        "nn_mean_path_length_m": nn["mean_rollout_path_length_m"],
        "nn_median_path_length_m": nn["median_rollout_path_length_m"],
        "nn_mean_distance_to_dijkstra_path_m": nn["mean_distance_to_dijkstra_path_m"],
        "nn_median_distance_to_dijkstra_path_m": nn["median_distance_to_dijkstra_path_m"],
        "nn_mean_max_distance_to_dijkstra_path_m": nn["mean_max_distance_to_dijkstra_path_m"],

        "mean_path_length_ratio": ratio.get("mean_path_length_ratio", None),
        "median_path_length_ratio": ratio.get("median_path_length_ratio", None),
        "max_path_length_ratio": ratio.get("max_path_length_ratio", None),
        "num_cases_ratio_gt_1_2": ratio.get("num_cases_ratio_gt_1_2", None),
        "num_cases_ratio_gt_1_5": ratio.get("num_cases_ratio_gt_1_5", None),
        "num_cases_ratio_gt_2_0": ratio.get("num_cases_ratio_gt_2_0", None),

        "pp_success_rate": pp["success_rate"],
        "pp_collision_rate": pp["collision_rate"],
    }


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint_paths = find_checkpoints(args.checkpoint_dir)

    print("Found checkpoints:")
    for path in checkpoint_paths:
        print(" ", path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    obs_config = LocalObservationConfig(
        crop_size_cells=args.crop_size,
        normalize_cost=True,
        unknown_cost_value=1.0,
        include_traversability_channel=not args.no_traversability_channel,
        include_occupancy_channel=args.include_occupancy_channel,
        include_clearance_channel=not args.no_clearance_channel,
        max_clearance_m=args.max_clearance_m,
        include_goal_mask_channel=args.include_goal_mask,
        goal_mask_sigma_cells=args.goal_mask_sigma_cells,
        cost_to_go_normalization=args.cost_to_go_normalization,
        include_cost_to_go_encoding=args.include_cost_to_go_encoding,
        cost_encoding_frequencies=tuple(args.cost_encoding_frequencies),
    )

    limits = ControlLimits(
        v_max=args.v_max,
        omega_max=args.omega_max,
    )

    sim_config = SimConfig(
        dt=args.dt,
        max_steps=args.max_steps,
        goal_position_tol=args.goal_position_tol,
    )

    pp_config = PurePursuitConfig(
        lookahead_distance=args.lookahead_distance,
        desired_speed=args.desired_speed,
        goal_slowdown_distance=args.goal_slowdown_distance,
    )

    rows = []
    all_summaries = {}

    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)

        print()
        print("=" * 80)
        print("Evaluating:", checkpoint_name)
        print("=" * 80)

        model, y_mean, y_std, checkpoint_meta = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        results = run_closed_loop_test_suite(
            model=model,
            y_mean=y_mean,
            y_std=y_std,
            obs_config=obs_config,
            num_tests=args.num_tests,
            base_seed=args.seed,
            device=device,
            pp_config=pp_config,
            limits=limits,
            sim_config=sim_config,
            verbose=False,
            planner_connectivity=args.planner_connectivity,
        )

        summary = summarize_closed_loop_test_suite(results)

        all_summaries[checkpoint_name] = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_meta": {
                "epoch": checkpoint_meta.get("epoch", None),
                "best_epoch": checkpoint_meta.get("best_epoch", None),
                "best_val_loss": checkpoint_meta.get("best_val_loss", None),
                "split_type": checkpoint_meta.get("split_type", None),
            },
            "summary": summary,
        }

        row = flatten_summary(
            checkpoint_name=checkpoint_name,
            checkpoint_meta=checkpoint_meta,
            summary=summary,
        )

        rows.append(row)

        print(json.dumps(row, indent=2))

    summary_json_path = os.path.join(args.out_dir, "checkpoint_summaries.json")
    with open(summary_json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    csv_path = os.path.join(args.out_dir, "checkpoint_comparison.csv")

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Select a practical best checkpoint.
    # Priority:
    #   1. high success_no_collision
    #   2. low collision
    #   3. low mean final error
    #   4. low path length ratio
    ranked = sorted(
        rows,
        key=lambda r: (
            -r["nn_success_no_collision_rate"],
            r["nn_collision_rate"],
            r["nn_mean_final_position_error"],
            r["mean_path_length_ratio"] if r["mean_path_length_ratio"] is not None else 999.0,
        ),
    )

    best_closed_loop = ranked[0]

    best_closed_loop_path = os.path.join(args.out_dir, "best_closed_loop.json")
    with open(best_closed_loop_path, "w") as f:
        json.dump(best_closed_loop, f, indent=2)

    print()
    print("=" * 80)
    print("Sweep complete")
    print("=" * 80)
    print("Saved:", summary_json_path)
    print("Saved:", csv_path)
    print("Saved:", best_closed_loop_path)
    print()
    print("Best closed-loop checkpoint:")
    print(json.dumps(best_closed_loop, indent=2))


if __name__ == "__main__":
    main()
