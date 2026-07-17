import argparse
import os

import torch

from gridnav_il.geometry import ControlLimits, SimConfig
from gridnav_il.controllers import PurePursuitConfig
from gridnav_il.observations import LocalObservationConfig
from gridnav_il.models import CNNPolicy
from gridnav_il.evaluation import run_one_closed_loop_test
from gridnav_il.plotting import plot_global_rollout_with_local_greedy_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot local greedy cost-to-go descents over the global rollout."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_index", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="outputs/global_greedy_debug")

    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 50, 100, 150, 200, 250, 300, 350, 400],
    )

    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument(
        "--cost_to_go_normalization",
        type=str,
        default="local",
        choices=["local", "global"],
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

    return parser.parse_args()


def main():
    args = parse_args()

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
        include_clearance_channel=True,
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

    seed = args.base_seed + args.test_index

    result = run_one_closed_loop_test(
        model=model,
        y_mean=y_mean,
        y_std=y_std,
        obs_config=obs_config,
        seed=seed,
        device=device,
        pp_config=pp_config,
        limits=limits,
        sim_config=sim_config,
    )

    out_path = os.path.join(
        args.out_dir,
        f"test_{args.test_index:03d}_global_greedy.png",
    )

    title = (
        f"Test {args.test_index:03d} | "
        f"NN success={result['nn_summary']['reached_goal']} "
        f"collision={result['nn_summary']['has_collision']}"
    )

    plot_global_rollout_with_local_greedy_paths(
        problem=result["problem"],
        pp_states_np=result["pp_states_np"],
        nn_states_np=result["nn_states_np"],
        obs_config=obs_config,
        timesteps=args.timesteps,
        title=title,
        out_path=out_path,
    )

    print("Saved:", out_path)
    print("NN summary:", result["nn_summary"])


if __name__ == "__main__":
    main()
