import numpy as np
import torch
from torch.utils.data import Dataset

from gridnav_il.geometry import (
    ControlLimits,
    SimConfig,
    states_to_array,
    controls_to_array,
    world_to_grid,
)
from gridnav_il.grid_world import create_valid_planning_problem, sample_problem_kwargs
from gridnav_il.controllers import (
    PurePursuitConfig,
    PurePursuitController,
    rollout_nav_controller,
    make_nav_context,
    make_start_state_from_problem,
)
from gridnav_il.observations import extract_local_grid_observation


def create_dataset_from_rollout(
    states,
    controls,
    nav_context,
    obs_config,
    stride=2,
    x_dtype=np.float16,
):
    assert len(states) == len(controls) + 1

    observations = []
    actions = []

    for i in range(0, len(controls), stride):
        state = states[i]
        cmd = controls[i]

        obs = extract_local_grid_observation(
            robot_state=state,
            nav_context=nav_context,
            obs_config=obs_config,
        )

        action = cmd.as_array()

        observations.append(obs.astype(x_dtype))
        actions.append(action)

    X = np.stack(observations, axis=0).astype(x_dtype)
    Y = np.stack(actions, axis=0).astype(np.float32)

    return X, Y


def check_collision_for_states(states_np, occupancy_grid, resolution, origin_world):
    collision_indices = []

    height, width = occupancy_grid.shape

    for i, state in enumerate(states_np):
        x, y, _ = state

        row, col = world_to_grid(
            point_world=np.array([x, y], dtype=np.float32),
            resolution=resolution,
            origin_world=origin_world,
        )

        if row < 0 or row >= height or col < 0 or col >= width:
            collision_indices.append(i)
            continue

        if occupancy_grid[row, col] == 1:
            collision_indices.append(i)

    has_collision = len(collision_indices) > 0

    return has_collision, collision_indices


def compute_min_obstacle_distance_along_rollout(
    states_np,
    distance_to_obstacle,
    resolution,
    origin_world,
):
    distances_m = []

    height, width = distance_to_obstacle.shape

    for state in states_np:
        x, y, _ = state

        row, col = world_to_grid(
            point_world=np.array([x, y], dtype=np.float32),
            resolution=resolution,
            origin_world=origin_world,
        )

        if row < 0 or row >= height or col < 0 or col >= width:
            distances_m.append(0.0)
        else:
            distances_m.append(distance_to_obstacle[row, col] * resolution)

    return np.array(distances_m, dtype=np.float32)

def compute_rollout_path_length(states_np):
    if len(states_np) < 2:
        return 0.0

    deltas = states_np[1:, :2] - states_np[:-1, :2]
    segment_lengths = np.linalg.norm(deltas, axis=1)

    return float(np.sum(segment_lengths))


def compute_distance_to_reference_path(states_np, path_world):
    if path_world is None or len(path_world) == 0:
        distances = np.full((len(states_np),), np.nan, dtype=np.float32)
        return distances

    robot_xy = states_np[:, :2]

    # Simple vectorized nearest-point distance to discrete path points.
    # Shape: [num_states, num_path_points]
    diff = robot_xy[:, None, :] - path_world[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    min_distances = np.min(distances, axis=1)

    return min_distances.astype(np.float32)

def summarize_rollout(problem, states_np, controls_np, sim_config):
    goal_world = problem["goal_world"]

    goal_distances = np.linalg.norm(
        states_np[:, :2] - goal_world[None, :],
        axis=1,
    )

    final_position_error = float(goal_distances[-1])
    min_goal_distance_during_rollout = float(np.min(goal_distances))

    reached_goal = final_position_error <= sim_config.goal_position_tol
    ever_reached_goal_region = (
        min_goal_distance_during_rollout <= sim_config.goal_position_tol
    )

    has_collision, collision_indices = check_collision_for_states(
        states_np=states_np,
        occupancy_grid=problem["occupancy_grid"],
        resolution=problem["resolution"],
        origin_world=problem["origin_world"],
    )

    obstacle_distances_m = compute_min_obstacle_distance_along_rollout(
        states_np=states_np,
        distance_to_obstacle=problem["distance_to_obstacle"],
        resolution=problem["resolution"],
        origin_world=problem["origin_world"],
    )

    rollout_path_length_m = compute_rollout_path_length(states_np)

    distance_to_path_m = compute_distance_to_reference_path(
        states_np=states_np,
        path_world=problem["path_world"],
    )

    mean_distance_to_path_m = float(np.nanmean(distance_to_path_m))
    max_distance_to_path_m = float(np.nanmax(distance_to_path_m))

    # Overshoot means:
    #   The robot came inside the goal tolerance at some point,
    #   but the rollout did not end successfully there.
    #
    # This captures cases where the policy reaches the goal region,
    # leaves it, loops around, and fails or collides later.
    overshoot_goal_region = bool(
        ever_reached_goal_region and not reached_goal
    )

    return {
        "steps": int(len(controls_np)),
        "final_position_error": final_position_error,
        "min_goal_distance_during_rollout": min_goal_distance_during_rollout,
        "reached_goal": bool(reached_goal),
        "ever_reached_goal_region": bool(ever_reached_goal_region),
        "overshoot_goal_region": bool(overshoot_goal_region),
        "has_collision": bool(has_collision),
        "num_collision_states": int(len(collision_indices)),
        "min_obstacle_distance_m": float(obstacle_distances_m.min()),
        "mean_obstacle_distance_m": float(obstacle_distances_m.mean()),
        "rollout_path_length_m": float(rollout_path_length_m),
        "mean_distance_to_dijkstra_path_m": mean_distance_to_path_m,
        "max_distance_to_dijkstra_path_m": max_distance_to_path_m,
    }


def generate_one_random_expert_demo(
    demo_seed,
    pp_config: PurePursuitConfig,
    limits: ControlLimits,
    sim_config: SimConfig,
    obs_config,
    keep_failed_demos=False,
    require_collision_free=True,
    min_clearance_m=0.05,
    planner_connectivity: int = 4,
):
    rng = np.random.default_rng(demo_seed)

    problem_kwargs = sample_problem_kwargs(
        seed=int(rng.integers(0, 1_000_000_000))
    )

    problem = create_valid_planning_problem(
        **problem_kwargs,
        seed=int(rng.integers(0, 1_000_000_000)),
        planner_connectivity=planner_connectivity,
    )

    nav_context = make_nav_context(problem)
    start_state = make_start_state_from_problem(problem)

    controller = PurePursuitController(
        config=pp_config,
        limits=limits,
    )

    states, controls = rollout_nav_controller(
        start_state=start_state,
        controller=controller,
        nav_context=nav_context,
        sim_config=sim_config,
    )

    states_np = states_to_array(states)
    controls_np = controls_to_array(controls)

    summary = summarize_rollout(
        problem=problem,
        states_np=states_np,
        controls_np=controls_np,
        sim_config=sim_config,
    )

    accepted = True
    rejection_reason = None

    if not keep_failed_demos and not summary["reached_goal"]:
        accepted = False
        rejection_reason = "did_not_reach_goal"

    if require_collision_free and summary["has_collision"]:
        accepted = False
        rejection_reason = "collision"

    if summary["min_obstacle_distance_m"] < min_clearance_m:
        accepted = False
        rejection_reason = "low_clearance"

    if not accepted:
        return None

    X, Y = create_dataset_from_rollout(
        states=states,
        controls=controls,
        nav_context=nav_context,
        obs_config=obs_config,
        stride=2,
        x_dtype=np.float16,
    )

    return {
        "demo_seed": demo_seed,
        "problem_kwargs": problem_kwargs,
        "problem": problem,
        "nav_context": nav_context,
        "states": states,
        "controls": controls,
        "states_np": states_np,
        "controls_np": controls_np,
        "summary": summary,
        "X": X,
        "Y": Y,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
    }


def generate_expert_dataset(
    num_demos,
    pp_config: PurePursuitConfig,
    limits: ControlLimits,
    sim_config: SimConfig,
    obs_config,
    base_seed=0,
    max_attempts=None,
    keep_failed_demos=False,
    require_collision_free=True,
    min_clearance_m=0.05,
    planner_connectivity: int = 4,
    verbose=True,
):
    if max_attempts is None:
        max_attempts = num_demos * 5

    rng = np.random.default_rng(base_seed)

    accepted_demos = []
    num_attempts = 0

    while len(accepted_demos) < num_demos and num_attempts < max_attempts:
        num_attempts += 1

        demo_seed = int(rng.integers(0, 1_000_000_000))

        demo = generate_one_random_expert_demo(
            demo_seed=demo_seed,
            pp_config=pp_config,
            limits=limits,
            sim_config=sim_config,
            obs_config=obs_config,
            keep_failed_demos=keep_failed_demos,
            require_collision_free=require_collision_free,
            min_clearance_m=min_clearance_m,
            planner_connectivity=planner_connectivity,
        )

        if demo is not None:
            demo["demo_id"] = len(accepted_demos)
            accepted_demos.append(demo)

        if verbose and num_attempts % 10 == 0:
            print(
                f"Attempts: {num_attempts}, "
                f"accepted: {len(accepted_demos)} / {num_demos}"
            )

    if len(accepted_demos) == 0:
        raise RuntimeError("No demos were accepted.")

    X = np.concatenate([demo["X"] for demo in accepted_demos], axis=0)
    Y = np.concatenate([demo["Y"] for demo in accepted_demos], axis=0)

    demo_ids = np.concatenate(
        [
            np.full(
                shape=(len(demo["X"]),),
                fill_value=demo["demo_id"],
                dtype=np.int32,
            )
            for demo in accepted_demos
        ],
        axis=0,
    )

    steps_per_demo = np.array(
        [demo["summary"]["steps"] for demo in accepted_demos],
        dtype=np.float32,
    )

    samples_per_demo = np.array(
        [len(demo["X"]) for demo in accepted_demos],
        dtype=np.float32,
    )

    final_errors = np.array(
        [demo["summary"]["final_position_error"] for demo in accepted_demos],
        dtype=np.float32,
    )

    min_clearances = np.array(
        [demo["summary"]["min_obstacle_distance_m"] for demo in accepted_demos],
        dtype=np.float32,
    )

    stats = {
        "requested_num_demos": int(num_demos),
        "accepted_num_demos": int(len(accepted_demos)),
        "num_attempts": int(num_attempts),
        "acceptance_rate": float(len(accepted_demos) / num_attempts),
        "total_samples": int(len(X)),
        "mean_steps_per_demo": float(np.mean(steps_per_demo)),
        "median_steps_per_demo": float(np.median(steps_per_demo)),
        "mean_samples_per_demo": float(np.mean(samples_per_demo)),
        "median_samples_per_demo": float(np.median(samples_per_demo)),
        "mean_final_position_error": float(np.mean(final_errors)),
        "max_final_position_error": float(np.max(final_errors)),
        "mean_min_clearance_m": float(np.mean(min_clearances)),
        "min_clearance_m": float(np.min(min_clearances)),
    }

    return {
        "X": X,
        "Y": Y,
        "demo_ids": demo_ids,
        "demos": accepted_demos,
        "stats": stats,
    }


class GridExpertTorchDataset(Dataset):
    def __init__(self, X, Y, y_mean=None, y_std=None):
        assert len(X) == len(Y)

        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

        if y_mean is None:
            y_mean = self.Y.mean(dim=0)

        if y_std is None:
            y_std = self.Y.std(dim=0) + 1e-6

        self.y_mean = y_mean.float()
        self.y_std = y_std.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        y_norm = (y - self.y_mean) / self.y_std

        return x, y_norm
