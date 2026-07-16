from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    x: float
    y: float
    theta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=np.float32)


@dataclass
class ControlCommand:
    v: float
    omega: float

    def as_array(self) -> np.ndarray:
        return np.array([self.v, self.omega], dtype=np.float32)


@dataclass
class ControlLimits:
    v_max: float = 1.0
    omega_max: float = 2.0


@dataclass
class SimConfig:
    dt: float = 0.1
    max_steps: int = 1000
    goal_position_tol: float = 0.25


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def grid_to_world(cell, resolution: float, origin_world: np.ndarray) -> np.ndarray:
    row, col = cell

    x = origin_world[0] + (col + 0.5) * resolution
    y = origin_world[1] + (row + 0.5) * resolution

    return np.array([x, y], dtype=np.float32)


def world_to_grid(point_world: np.ndarray, resolution: float, origin_world: np.ndarray):
    x, y = point_world

    col = int((x - origin_world[0]) / resolution)
    row = int((y - origin_world[1]) / resolution)

    return row, col


def unicycle_step(
    robot_state: RobotState,
    cmd: ControlCommand,
    dt: float,
) -> RobotState:
    x_next = robot_state.x + cmd.v * np.cos(robot_state.theta) * dt
    y_next = robot_state.y + cmd.v * np.sin(robot_state.theta) * dt
    theta_next = wrap_angle(robot_state.theta + cmd.omega * dt)

    return RobotState(
        x=float(x_next),
        y=float(y_next),
        theta=float(theta_next),
    )


def clip_control(v: float, omega: float, limits: ControlLimits) -> ControlCommand:
    v = np.clip(v, -limits.v_max, limits.v_max)
    omega = np.clip(omega, -limits.omega_max, limits.omega_max)

    return ControlCommand(
        v=float(v),
        omega=float(omega),
    )


def states_to_array(states) -> np.ndarray:
    return np.array(
        [[s.x, s.y, s.theta] for s in states],
        dtype=np.float32,
    )


def controls_to_array(controls) -> np.ndarray:
    return np.array(
        [[c.v, c.omega] for c in controls],
        dtype=np.float32,
    )
