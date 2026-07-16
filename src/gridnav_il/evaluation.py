import numpy as np

from gridnav_il.geometry import (
    ControlLimits,
    SimConfig,
    states_to_array,
    controls_to_array,
)
from gridnav_il.grid_world import create_valid_planning_problem, sample_problem_kwargs
from gridnav_il.controllers import (
    PurePursuitConfig,
    PurePursuitController,
    NeuralGridController,
    rollout_nav_controller,
    make_nav_context,
    make_start_state_from_problem,
)
from gridnav_il.dataset import summarize_rollout


def run_one_closed_loop_test(
    model,
    y_mean,
    y_std,
    obs_config,
    seed,
    device,
    pp_config=None,
    limits=None,
    sim_config=None,
):
    if pp_config is None:
        pp_config = PurePursuitConfig()

    if limits is None:
        limits = ControlLimits()

    if sim_config is None:
        sim_config = SimConfig()

    problem_kwargs = sample_problem_kwargs(seed=seed)
    problem = create_valid_planning_problem(**problem_kwargs, seed=seed + 10000)

    nav_context = make_nav_context(problem)
    start_state = make_start_state_from_problem(problem)

    pp_controller = PurePursuitController(
        config=pp_config,
        limits=limits,
    )

    nn_controller = NeuralGridController(
        model=model,
        y_mean=y_mean,
        y_std=y_std,
        obs_config=obs_config,
        limits=limits,
        device=device,
    )

    pp_states, pp_controls = rollout_nav_controller(
        start_state=start_state,
        controller=pp_controller,
        nav_context=nav_context,
        sim_config=sim_config,
    )

    nn_states, nn_controls = rollout_nav_controller(
        start_state=start_state,
        controller=nn_controller,
        nav_context=nav_context,
        sim_config=sim_config,
    )

    pp_states_np = states_to_array(pp_states)
    pp_controls_np = controls_to_array(pp_controls)

    nn_states_np = states_to_array(nn_states)
    nn_controls_np = controls_to_array(nn_controls)

    pp_summary = summarize_rollout(
        problem=problem,
        states_np=pp_states_np,
        controls_np=pp_controls_np,
        sim_config=sim_config,
    )

    nn_summary = summarize_rollout(
        problem=problem,
        states_np=nn_states_np,
        controls_np=nn_controls_np,
        sim_config=sim_config,
    )

    return {
        "seed": seed,
        "problem": problem,
        "problem_kwargs": problem_kwargs,
        "start_state": start_state,
        "pp_states_np": pp_states_np,
        "pp_controls_np": pp_controls_np,
        "nn_states_np": nn_states_np,
        "nn_controls_np": nn_controls_np,
        "pp_summary": pp_summary,
        "nn_summary": nn_summary,
    }


def run_closed_loop_test_suite(
    model,
    y_mean,
    y_std,
    obs_config,
    num_tests,
    base_seed,
    device,
    pp_config=None,
    limits=None,
    sim_config=None,
    verbose=True,
):
    results = []

    for i in range(num_tests):
        seed = base_seed + i

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

        results.append(result)

        if verbose:
            pp = result["pp_summary"]
            nn = result["nn_summary"]

            print(
                f"[{i + 1:03d}/{num_tests:03d}] "
                f"PP success={pp['reached_goal']} collision={pp['has_collision']} "
                f"err={pp['final_position_error']:.3f} | "
                f"NN success={nn['reached_goal']} collision={nn['has_collision']} "
                f"err={nn['final_position_error']:.3f}"
            )

    return results


def summarize_closed_loop_test_suite(results):
    def collect(controller_name, key):
        return np.array(
            [r[f"{controller_name}_summary"][key] for r in results],
            dtype=np.float32,
        )

    summary = {}

    for name in ["pp", "nn"]:
        reached = collect(name, "reached_goal")
        collision = collect(name, "has_collision")
        final_error = collect(name, "final_position_error")
        steps = collect(name, "steps")
        min_clearance = collect(name, "min_obstacle_distance_m")

        summary[name] = {
            "success_rate": float(np.mean(reached)),
            "collision_rate": float(np.mean(collision)),
            "mean_final_position_error": float(np.mean(final_error)),
            "median_final_position_error": float(np.median(final_error)),
            "mean_steps": float(np.mean(steps)),
            "median_steps": float(np.median(steps)),
            "mean_min_clearance_m": float(np.mean(min_clearance)),
            "min_clearance_m": float(np.min(min_clearance)),
        }

    return summary
