import argparse
import os
import json
import numpy as np

from gridnav_il.geometry import ControlLimits, SimConfig
from gridnav_il.controllers import PurePursuitConfig
from gridnav_il.observations import LocalObservationConfig
from gridnav_il.dataset import generate_expert_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Pure Pursuit expert dataset for grid navigation imitation learning."
    )

    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/gridnav_pp_200.npz")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--min_clearance_m", type=float, default=0.05)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--goal_position_tol", type=float, default=0.25)

    parser.add_argument("--lookahead_distance", type=float, default=0.4)
    parser.add_argument("--desired_speed", type=float, default=0.4)
    parser.add_argument("--goal_slowdown_distance", type=float, default=0.8)

    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--omega_max", type=float, default=2.0)

    parser.add_argument(
        "--max_clearance_m",
        type=float,
        default=2.0,
        help="Clearance distance that maps to 1.0 in the clearance channel.",
    )

    parser.add_argument(
        "--cost_to_go_normalization",
        type=str,
        default="local",
        choices=["local", "global", "none"],
        help="How to normalize the cost-to-go channel.",
    )

    parser.add_argument(
        "--include_goal_mask",
        action="store_true",
        help="Include goal mask as an extra observation channel.",
    )

    parser.add_argument(
        "--goal_mask_sigma_cells",
        type=float,
        default=2.0,
        help="Gaussian sigma for the goal mask in pixels/cells.",
    )

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
        help="Planner connectivity for Dijkstra and path extraction.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    obs_config = LocalObservationConfig(
        crop_size_cells=args.crop_size,
        normalize_cost=True,
        unknown_cost_value=1.0,
        include_traversability_channel=not args.no_traversability_channel,
        include_clearance_channel=not args.no_clearance_channel,
        max_clearance_m=args.max_clearance_m,
        include_goal_mask_channel=args.include_goal_mask,
        goal_mask_sigma_cells=args.goal_mask_sigma_cells,
        cost_to_go_normalization=args.cost_to_go_normalization,
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

    dataset = generate_expert_dataset(
        num_demos=args.num_demos,
        pp_config=pp_config,
        limits=limits,
        sim_config=sim_config,
        obs_config=obs_config,
        base_seed=args.seed,
        max_attempts=args.max_attempts,
        keep_failed_demos=False,
        require_collision_free=True,
        min_clearance_m=args.min_clearance_m,
        planner_connectivity=args.planner_connectivity,
        verbose=True,
    )

    X = dataset["X"]
    Y = dataset["Y"]
    stats = dataset["stats"]

    np.savez_compressed(
        args.out,
        X=X,
        Y=Y,
        demo_ids=dataset["demo_ids"],
        stats_json=json.dumps(stats),
    )

    print()
    print("Saved dataset:", args.out)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("demo_ids shape:", dataset["demo_ids"].shape)
    print("num unique demos:", len(np.unique(dataset["demo_ids"])))
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
