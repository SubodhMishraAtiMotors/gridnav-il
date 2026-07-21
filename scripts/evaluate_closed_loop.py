import argparse
import os
import json

import torch

from gridnav_il.geometry import ControlLimits, SimConfig
from gridnav_il.controllers import PurePursuitConfig
from gridnav_il.observations import LocalObservationConfig
from gridnav_il.models import CNNPolicy
from gridnav_il.evaluation import (
    run_closed_loop_test_suite,
    summarize_closed_loop_test_suite,
    get_failure_case_indices,
)
from gridnav_il.plotting import (
    plot_expert_vs_nn_rollout,
    plot_expert_vs_nn_rollout_on_cost_to_go,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN policy in closed loop against Pure Pursuit."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_tests", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="outputs/eval_cnn")

    parser.add_argument("--crop_size", type=int, default=64)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--goal_position_tol", type=float, default=0.25)

    parser.add_argument("--lookahead_distance", type=float, default=0.4)
    parser.add_argument("--desired_speed", type=float, default=0.4)
    parser.add_argument("--goal_slowdown_distance", type=float, default=0.8)

    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--omega_max", type=float, default=2.0)

    parser.add_argument("--num_plots", type=int, default=10)

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

    parser.add_argument(
        "--num_waypoints",
        type=int,
        default=0,
        help=(
            "Number of predicted waypoints. "
            "If 0, use direct v, omega prediction."
        ),
    )

    parser.add_argument(
        "--waypoint_tracking_index",
        type=int,
        default=0,
        help="Which predicted waypoint to track. Usually 0 for the first waypoint.",
    )

    parser.add_argument(
        "--waypoint_kx",
        type=float,
        default=0.8,
        help="Forward gain for waypoint tracking controller.",
    )

    parser.add_argument(
        "--waypoint_ky",
        type=float,
        default=1.5,
        help="Lateral gain for waypoint tracking controller.",
    )

    parser.add_argument(
        "--waypoint_ktheta",
        type=float,
        default=0.8,
        help="Heading gain for waypoint tracking controller.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_waypoints < 0:
        raise ValueError("--num_waypoints must be >= 0.")

    if args.num_waypoints == 0:
        print(
            "WARNING: --num_waypoints is 0. "
            "Closed-loop evaluation will use direct v, omega prediction."
        )

    if args.num_waypoints > 0:
        if args.waypoint_tracking_index < 0:
            raise ValueError("--waypoint_tracking_index must be >= 0.")

        if args.waypoint_tracking_index >= args.num_waypoints:
            raise ValueError(
                "--waypoint_tracking_index must be smaller than --num_waypoints."
            )

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint = torch.load(
        args.checkpoint,
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
        planner_connectivity=args.planner_connectivity,
        num_waypoints=args.num_waypoints,
        waypoint_tracking_index=args.waypoint_tracking_index,
        waypoint_kx=args.waypoint_kx,
        waypoint_ky=args.waypoint_ky,
        waypoint_ktheta=args.waypoint_ktheta,
        verbose=True,
    )

    summary = summarize_closed_loop_test_suite(results)

    failure_cases = get_failure_case_indices(
        results=results,
        controller_name="nn",
    )

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    failure_cases_path = os.path.join(args.out_dir, "failure_cases.json")
    with open(failure_cases_path, "w") as f:
        json.dump(failure_cases, f, indent=2)

    print()
    print("Closed-loop summary")
    print("-------------------")
    print(json.dumps(summary, indent=2))
    print()
    print("Saved failure cases:", failure_cases_path)
    print()
    print("NN failure case indices")
    print("-----------------------")
    for k, v in failure_cases.items():
        print(f"{k}: {v}")

    num_plots = min(args.num_plots, len(results))

    for i in range(num_plots):
        result = results[i]

        pp = result["pp_summary"]
        nn = result["nn_summary"]

        title = (
            f"Test {i:03d} | "
            f"PP success={pp['reached_goal']} collision={pp['has_collision']} | "
            f"NN success={nn['reached_goal']} collision={nn['has_collision']}"
        )

        occupancy_out_path = os.path.join(
            args.out_dir,
            f"rollout_{i:03d}_occupancy.png",
        )

        cost_to_go_out_path = os.path.join(
            args.out_dir,
            f"rollout_{i:03d}_cost_to_go.png",
        )

        plot_expert_vs_nn_rollout(
            problem=result["problem"],
            pp_states_np=result["pp_states_np"],
            nn_states_np=result["nn_states_np"],
            title=title,
            out_path=occupancy_out_path,
        )

        plot_expert_vs_nn_rollout_on_cost_to_go(
            problem=result["problem"],
            pp_states_np=result["pp_states_np"],
            nn_states_np=result["nn_states_np"],
            title=title + " | Cost-to-go",
            out_path=cost_to_go_out_path,
        )

    print("Saved rollout plots in:", args.out_dir)


if __name__ == "__main__":
    main()
