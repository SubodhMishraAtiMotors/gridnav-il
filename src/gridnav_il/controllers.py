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
    distance_to_obstacle: np.ndarray
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


class NeuralGridController(BaseNavController):
    def __init__(
        self,
        model,
        y_mean,
        y_std,
        obs_config,
        limits: ControlLimits,
        device,
        num_waypoints: int = 0,
        waypoint_tracking_index: int = 0,
        waypoint_kx: float = 0.8,
        waypoint_ky: float = 1.5,
        waypoint_ktheta: float = 0.8,
        sequence_length: int = 1,
    ):
        self.model = model
        self.model.eval()

        self.y_mean = y_mean.to(device)
        self.y_std = y_std.to(device)

        self.obs_config = obs_config
        self.limits = limits
        self.device = device

        self.num_waypoints = int(num_waypoints)
        self.waypoint_tracking_index = int(waypoint_tracking_index)

        self.waypoint_kx = float(waypoint_kx)
        self.waypoint_ky = float(waypoint_ky)
        self.waypoint_ktheta = float(waypoint_ktheta)

        self.sequence_length = int(sequence_length)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0.")

        self.obs_history = []

        self.predicted_waypoints_world_history = []
        self.last_predicted_waypoints_robot = None
        self.last_predicted_waypoints_world = None

        if self.num_waypoints < 0:
            raise ValueError("num_waypoints must be >= 0.")

        if self.num_waypoints == 0:
            print(
                "WARNING: NeuralGridController num_waypoints is 0. "
                "Using direct v, omega prediction."
            )

        if self.num_waypoints > 0:
            if self.waypoint_tracking_index < 0:
                raise ValueError("waypoint_tracking_index must be >= 0.")

            if self.waypoint_tracking_index >= self.num_waypoints:
                raise ValueError(
                    "waypoint_tracking_index must be smaller than num_waypoints. "
                    f"Got waypoint_tracking_index={self.waypoint_tracking_index}, "
                    f"num_waypoints={self.num_waypoints}."
                )

    def reset(self):
        self.obs_history = []
        self.predicted_waypoints_world_history = []
        self.last_predicted_waypoints_robot = None
        self.last_predicted_waypoints_world = None

    def _robot_waypoints_to_world(self, robot_waypoints, current_state):
        """
        Convert predicted waypoints from current robot frame to world frame.

        robot_waypoints shape:
            (num_waypoints, 4)

        Each row:
            [x_robot, y_robot, sin(theta_robot), cos(theta_robot)]

        Output shape:
            (num_waypoints, 3)

        Each row:
            [x_world, y_world, theta_world]
        """
        c = float(np.cos(current_state.theta))
        s = float(np.sin(current_state.theta))

        world_waypoints = []

        for wp in robot_waypoints:
            x_robot = float(wp[0])
            y_robot = float(wp[1])
            theta_robot = float(np.arctan2(float(wp[2]), float(wp[3])))

            x_world = current_state.x + c * x_robot - s * y_robot
            y_world = current_state.y + s * x_robot + c * y_robot
            theta_world = current_state.theta + theta_robot

            world_waypoints.append([x_world, y_world, theta_world])

        return np.asarray(world_waypoints, dtype=np.float32)

    def _waypoints_to_control(self, pred_waypoints) -> ControlCommand:
        """
        Convert predicted waypoint sequence into v, omega.

        pred_waypoints is flattened:
            [x1, y1, sin(theta1), cos(theta1),
             x2, y2, sin(theta2), cos(theta2),
             ...]

        The waypoint is expressed in the current robot frame:
            x: forward
            y: left
            theta: heading relative to current robot heading
        """
        expected_dim = 4 * self.num_waypoints

        if pred_waypoints.shape[0] != expected_dim:
            raise ValueError(
                "Waypoint model output dimension mismatch. "
                f"Expected {expected_dim}, got {pred_waypoints.shape[0]}."
            )

        waypoints = pred_waypoints.reshape(self.num_waypoints, 4)

        wp = waypoints[self.waypoint_tracking_index]

        x_target = float(wp[0])
        y_target = float(wp[1])
        sin_theta = float(wp[2])
        cos_theta = float(wp[3])

        theta_target = float(np.arctan2(sin_theta, cos_theta))

        distance = float(np.sqrt(x_target * x_target + y_target * y_target))

        # Direction of the selected waypoint in the robot frame.
        # x is forward, y is left.
        heading_error = float(np.arctan2(y_target, max(x_target, 1e-6)))

        # Turn toward the waypoint and also respect the predicted target orientation.
        omega = (
            self.waypoint_ky * heading_error
            + self.waypoint_ktheta * theta_target
        )

        # Move fast only when the waypoint is in front.
        # If the waypoint is sideways, turn first and slow down.
        alignment_scale = max(0.0, float(np.cos(heading_error)))

        # Avoid amplifying tiny waypoint prediction noise.
        # If the waypoint is very close, slow down.
        distance_scale = float(np.clip(distance / 0.4, 0.0, 1.0))

        # Speed command.
        v = self.waypoint_kx * distance

        # Cap speed before applying slowdowns.
        v = min(v, self.limits.v_max)

        # Apply safety/smoothness slowdowns.
        v = v * alignment_scale * distance_scale

        return clip_control(v, omega, self.limits)

    def __call__(self, robot_state: RobotState, nav_context: NavContext) -> ControlCommand:
        import torch
        from gridnav_il.observations import extract_local_grid_observation

        obs = extract_local_grid_observation(
            robot_state=robot_state,
            nav_context=nav_context,
            obs_config=self.obs_config,
        )

        if self.sequence_length > 1:
            self.obs_history.append(obs.astype(np.float32, copy=True))

            if len(self.obs_history) > self.sequence_length:
                self.obs_history = self.obs_history[-self.sequence_length:]

            if len(self.obs_history) < self.sequence_length:
                pad_count = self.sequence_length - len(self.obs_history)
                padded_history = [self.obs_history[0]] * pad_count + self.obs_history
            else:
                padded_history = self.obs_history

            obs_seq = np.stack(padded_history, axis=0)  # [K, C, H, W]
            obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(self.device)

        else:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(obs_tensor)[0]
            pred = pred_norm * self.y_std + self.y_mean

        pred_np = pred.detach().cpu().numpy().astype(np.float32)

        if self.num_waypoints > 0:
            robot_waypoints = pred_np.reshape(self.num_waypoints, 4)

            self.last_predicted_waypoints_robot = robot_waypoints.copy()
            self.last_predicted_waypoints_world = self._robot_waypoints_to_world(
                robot_waypoints=robot_waypoints,
                current_state=robot_state,
            )

            self.predicted_waypoints_world_history.append(
                self.last_predicted_waypoints_world.copy()
            )

            return self._waypoints_to_control(pred_np)

        if pred_np.shape[0] < 2:
            raise ValueError(
                "Direct v, omega prediction requires model output dimension >= 2. "
                f"Got {pred_np.shape[0]}."
            )

        v = float(pred_np[0])
        omega = float(pred_np[1])

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
        distance_to_obstacle=problem["distance_to_obstacle"],
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
