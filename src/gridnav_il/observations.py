from dataclasses import dataclass

import numpy as np

from gridnav_il.geometry import RobotState
from gridnav_il.controllers import NavContext


@dataclass
class LocalObservationConfig:
    crop_size_cells: int = 64
    normalize_cost: bool = True
    unknown_cost_value: float = 1.0


def normalize_map_crop(crop, unknown_value=1.0):
    crop = crop.astype(np.float32).copy()

    finite_mask = np.isfinite(crop)

    normalized = np.full_like(
        crop,
        fill_value=unknown_value,
        dtype=np.float32,
    )

    if np.any(finite_mask):
        finite_values = crop[finite_mask]

        min_value = np.min(finite_values)
        max_value = np.max(finite_values)

        denom = max(max_value - min_value, 1e-6)

        normalized[finite_mask] = (finite_values - min_value) / denom

    return normalized


def sample_grid_nearest(grid, row_float, col_float, unknown_value):
    height, width = grid.shape

    row = int(np.round(row_float))
    col = int(np.round(col_float))

    if row < 0 or row >= height or col < 0 or col >= width:
        return unknown_value

    return grid[row, col]


def extract_robot_frame_grid_observation(
    robot_state: RobotState,
    nav_context: NavContext,
    obs_config: LocalObservationConfig,
):
    """
    Robot-frame local observation.

    Convention:
      - robot is at crop center
      - image up = robot forward
      - image left = robot left
      - channel 0 = traversal cost
      - channel 1 = cost-to-go
    """
    crop_size = obs_config.crop_size_cells
    assert crop_size % 2 == 0, "Use an even crop size."

    half = crop_size // 2
    resolution = nav_context.resolution

    traversal_crop = np.full(
        (crop_size, crop_size),
        fill_value=np.inf,
        dtype=np.float32,
    )

    cost_to_go_crop = np.full(
        (crop_size, crop_size),
        fill_value=np.inf,
        dtype=np.float32,
    )

    cos_theta = np.cos(robot_state.theta)
    sin_theta = np.sin(robot_state.theta)

    for r in range(crop_size):
        for c in range(crop_size):
            # Body frame convention:
            # x_body = forward
            # y_body = left
            #
            # Image convention:
            # row decreases upward, col decreases leftward.
            x_body = (half - r) * resolution
            y_body = (half - c) * resolution

            x_world = (
                robot_state.x
                + cos_theta * x_body
                - sin_theta * y_body
            )

            y_world = (
                robot_state.y
                + sin_theta * x_body
                + cos_theta * y_body
            )

            row_float = (y_world - nav_context.origin_world[1]) / resolution
            col_float = (x_world - nav_context.origin_world[0]) / resolution

            traversal_crop[r, c] = sample_grid_nearest(
                grid=nav_context.traversal_cost,
                row_float=row_float,
                col_float=col_float,
                unknown_value=np.inf,
            )

            cost_to_go_crop[r, c] = sample_grid_nearest(
                grid=nav_context.cost_to_go,
                row_float=row_float,
                col_float=col_float,
                unknown_value=np.inf,
            )

    if obs_config.normalize_cost:
        traversal_crop = normalize_map_crop(
            traversal_crop,
            unknown_value=obs_config.unknown_cost_value,
        )

        cost_to_go_crop = normalize_map_crop(
            cost_to_go_crop,
            unknown_value=obs_config.unknown_cost_value,
        )
    else:
        traversal_crop[~np.isfinite(traversal_crop)] = obs_config.unknown_cost_value
        cost_to_go_crop[~np.isfinite(cost_to_go_crop)] = obs_config.unknown_cost_value

    obs = np.stack(
        [
            traversal_crop,
            cost_to_go_crop,
        ],
        axis=0,
    ).astype(np.float32)

    return obs


def extract_local_grid_observation(
    robot_state: RobotState,
    nav_context: NavContext,
    obs_config: LocalObservationConfig,
):
    return extract_robot_frame_grid_observation(
        robot_state=robot_state,
        nav_context=nav_context,
        obs_config=obs_config,
    )
