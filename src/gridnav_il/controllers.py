from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from gridnav_il.geometry import (
    RobotState,
    ControlCommand,
    ControlLimits,
    SimConfig,
    clip_control,
    unicycle_step,
)


@dataclass
class NavContext:
    occupancy_grid: np.ndarray
    traversal_cost: np.ndarray
    cost_to_go: np.ndarray
    path_world: np.ndarray
    goal_world: np.ndarray
    resolution: float
    origin_world: np.ndarray


@dataclass
class PurePursuitConfig:
    lookahead_distance: float = 0.4
    desired_speed: float = 0.4
    goal_slowdown_distance: float = 0.8


class BaseNavController(ABC):
    @abstractmethod
    def __call__(self, robot_state: RobotState, nav_context: NavContext) -> ControlCommand:
        pass

    def reset(self):
        pass


class PurePursuitController(BaseNavController):
    def __init__(
        self,
        config: PurePursuitConfig,
        limits: ControlLimits,
    ):
        self.config = config
        self.limits = limits

    def __call__(self, robot_state: RobotState, nav_context: NavContext) -> ControlCommand:
        path = nav_context.path_world

        if path is None or len(path) == 0:
            return ControlCommand(0.0, 0.0)

        robot_xy = np.array([robot_state.x, robot_state.y], dtype=np.float32)

        distances = np.linalg.norm(path - robot_xy[None, :], axis=1)
        nearest_idx = int(np.argmin(distances))

        lookahead_idx = len(path) - 1
        for i in range(nearest_idx, len(path)):
            dist = np.linalg.norm(path[i] - robot_xy)
            if dist >= self.config.lookahead_distance:
                lookahead_idx = i
                break

        target = path[lookahead_idx]

        dx_world = target[0] - robot_state.x
        dy_world = target[1] - robot_state.y

        dx_body = (
            np.cos(robot_state.theta) * dx_world
            + np.sin(robot_state.theta) * dy_world
        )

        dy_body = (
            -np.sin(robot_state.theta) * dx_world
            + np.cos(robot_state.theta) * dy_world
        )

        distance_to_goal = np.linalg.norm(nav_context.goal_world - robot_xy)

        v = self.config.desired_speed

        if distance_to_goal < self.config.goal_slowdown_distance:
            v = self.config.desired_speed * (
                distance_to_goal / self.config.goal_slowdown_distance
            )

        v = max(0.0, v)

        lookahead_dist_sq = dx_body**2 + dy_body**2 + 1e-6
        curvature = 2.0 * dy_body / lookahead_dist_sq

        omega = v * curvature

        return clip_control(v, omega, self.limits)


def rollout_nav_controller(
    start_state: RobotState,
    controller: BaseNavController,
    nav_context: NavContext,
    sim_config: SimConfig,
):
    controller.reset()

    states = [start_state]
    controls = []

    state = start_state

    for _ in range(sim_config.max_steps):
        cmd = controller(state, nav_context)
        controls.append(cmd)

        next_state = unicycle_step(
            robot_state=state,
            cmd=cmd,
            dt=sim_config.dt,
        )

        states.append(next_state)
        state = next_state

        robot_xy = np.array([state.x, state.y], dtype=np.float32)
        position_error = np.linalg.norm(robot_xy - nav_context.goal_world)

        if position_error <= sim_config.goal_position_tol:
            break

    return states, controls


def make_nav_context(problem) -> NavContext:
    return NavContext(
        occupancy_grid=problem["occupancy_grid"],
        traversal_cost=problem["traversal_cost"],
        cost_to_go=problem["cost_to_go"],
        path_world=problem["path_world"],
        goal_world=problem["goal_world"],
        resolution=problem["resolution"],
        origin_world=problem["origin_world"],
    )


def make_start_state_from_problem(problem) -> RobotState:
    start_world = problem["start_world"]
    path_world = problem["path_world"]

    if len(path_world) > 5:
        direction = path_world[5] - path_world[0]
    else:
        direction = path_world[-1] - path_world[0]

    start_theta = np.arctan2(direction[1], direction[0])

    return RobotState(
        x=float(start_world[0]),
        y=float(start_world[1]),
        theta=float(start_theta),
    )
