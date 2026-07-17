from dataclasses import dataclass

import numpy as np

from gridnav_il.geometry import RobotState
from gridnav_il.controllers import NavContext


@dataclass
class LocalObservationConfig:
    crop_size_cells: int = 64
    normalize_cost: bool = True
    unknown_cost_value: float = 1.0

    include_traversability_channel: bool = True

    include_clearance_channel: bool = True
    max_clearance_m: float = 2.0

    include_goal_mask_channel: bool = True
    goal_mask_sigma_cells: float = 2.0

    cost_to_go_normalization: str = "local"


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


def normalize_cost_to_go_crop_global(
    cost_to_go_crop,
    full_cost_to_go,
    unknown_value=1.0,
):
    cost_to_go_crop = cost_to_go_crop.astype(np.float32).copy()

    finite_map_mask = np.isfinite(full_cost_to_go)

    normalized = np.full_like(
        cost_to_go_crop,
        fill_value=unknown_value,
        dtype=np.float32,
    )

    if not np.any(finite_map_mask):
        return normalized

    global_max = np.max(full_cost_to_go[finite_map_mask])
    denom = max(float(global_max), 1e-6)

    finite_crop_mask = np.isfinite(cost_to_go_crop)

    normalized[finite_crop_mask] = np.clip(
        cost_to_go_crop[finite_crop_mask] / denom,
        0.0,
        1.0,
    )

    return normalized


def normalize_clearance_crop(
    clearance_crop_cells,
    resolution,
    max_clearance_m,
    unknown_value=0.0,
):
    """
    Convert distance-to-obstacle from cells to normalized clearance.

    Output convention:
      0.0 = obstacle / very unsafe / unknown
      1.0 = at least max_clearance_m away from obstacle
    """
    clearance_m = clearance_crop_cells.astype(np.float32) * resolution

    finite_mask = np.isfinite(clearance_m)

    normalized = np.full_like(
        clearance_m,
        fill_value=unknown_value,
        dtype=np.float32,
    )

    normalized[finite_mask] = np.clip(
        clearance_m[finite_mask] / max_clearance_m,
        0.0,
        1.0,
    )

    return normalized


_LOCAL_GRID_CACHE = {}


def get_robot_frame_offsets(crop_size, resolution):
    """
    Precompute local body-frame coordinates for every crop pixel.

    Convention:
      image up    = robot forward
      image left  = robot left
      x_body      = forward
      y_body      = left
    """
    key = (crop_size, float(resolution))

    if key in _LOCAL_GRID_CACHE:
        return _LOCAL_GRID_CACHE[key]

    assert crop_size % 2 == 0, "Use an even crop size."

    half = crop_size // 2

    rows, cols = np.meshgrid(
        np.arange(crop_size),
        np.arange(crop_size),
        indexing="ij",
    )

    x_body = (half - rows).astype(np.float32) * resolution
    y_body = (half - cols).astype(np.float32) * resolution

    _LOCAL_GRID_CACHE[key] = (x_body, y_body)

    return x_body, y_body


def sample_grid_nearest_vectorized(grid, row_float, col_float, unknown_value):
    height, width = grid.shape

    rows = np.rint(row_float).astype(np.int32)
    cols = np.rint(col_float).astype(np.int32)

    valid = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )

    sampled = np.full(
        row_float.shape,
        fill_value=unknown_value,
        dtype=np.float32,
    )

    sampled[valid] = grid[rows[valid], cols[valid]]

    return sampled


def compute_goal_mask(
    robot_state: RobotState,
    nav_context: NavContext,
    crop_size: int,
    resolution: float,
    sigma_cells: float,
):
    """
    Create a robot-frame Gaussian goal mask.

    Output:
      [crop_size, crop_size]

    Convention:
      goal inside local crop  -> Gaussian blob
      goal outside local crop -> all zeros
    """
    half = crop_size // 2

    dx_world = nav_context.goal_world[0] - robot_state.x
    dy_world = nav_context.goal_world[1] - robot_state.y

    cos_theta = np.cos(robot_state.theta)
    sin_theta = np.sin(robot_state.theta)

    # World to robot/body frame.
    # x_body = forward
    # y_body = left
    goal_x_body = cos_theta * dx_world + sin_theta * dy_world
    goal_y_body = -sin_theta * dx_world + cos_theta * dy_world

    # Body frame to image coordinates.
    # x_body positive forward -> row decreases.
    # y_body positive left    -> col decreases.
    goal_row = half - goal_x_body / resolution
    goal_col = half - goal_y_body / resolution

    if (
        goal_row < 0
        or goal_row >= crop_size
        or goal_col < 0
        or goal_col >= crop_size
    ):
        return np.zeros((crop_size, crop_size), dtype=np.float32)

    rows, cols = np.meshgrid(
        np.arange(crop_size),
        np.arange(crop_size),
        indexing="ij",
    )

    dist_sq = (rows - goal_row) ** 2 + (cols - goal_col) ** 2

    sigma_sq = max(float(sigma_cells) ** 2, 1e-6)

    goal_mask = np.exp(-0.5 * dist_sq / sigma_sq).astype(np.float32)

    return goal_mask


def extract_robot_frame_grid_observation(
    robot_state: RobotState,
    nav_context: NavContext,
    obs_config: LocalObservationConfig,
):
    """
    Fast vectorized robot-frame local observation.

    Convention:
      - robot is at crop center
      - image up = robot forward
      - image left = robot left

    Channels:
      channel 0 = traversal cost
      channel 1 = cost-to-go
      channel 2 = clearance, optional
      channel 3 = goal mask, optional
    """
    crop_size = obs_config.crop_size_cells
    resolution = nav_context.resolution

    x_body, y_body = get_robot_frame_offsets(
        crop_size=crop_size,
        resolution=resolution,
    )

    cos_theta = np.cos(robot_state.theta)
    sin_theta = np.sin(robot_state.theta)

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

    traversal_crop = sample_grid_nearest_vectorized(
        grid=nav_context.traversal_cost,
        row_float=row_float,
        col_float=col_float,
        unknown_value=np.inf,
    )

    cost_to_go_crop = sample_grid_nearest_vectorized(
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

        if obs_config.cost_to_go_normalization == "local":
            cost_to_go_crop = normalize_map_crop(
                cost_to_go_crop,
                unknown_value=obs_config.unknown_cost_value,
            )

        elif obs_config.cost_to_go_normalization == "global":
            cost_to_go_crop = normalize_cost_to_go_crop_global(
                cost_to_go_crop=cost_to_go_crop,
                full_cost_to_go=nav_context.cost_to_go,
                unknown_value=obs_config.unknown_cost_value,
            )

        elif obs_config.cost_to_go_normalization == "none":
            # Use raw global cost-to-go values.
            # Unknown / unreachable cells are assigned a large finite value.
            finite_map_mask = np.isfinite(nav_context.cost_to_go)

            if np.any(finite_map_mask):
                unknown_ctg_value = float(np.max(nav_context.cost_to_go[finite_map_mask]))
            else:
                unknown_ctg_value = obs_config.unknown_cost_value

            cost_to_go_crop = cost_to_go_crop.astype(np.float32)
            cost_to_go_crop[~np.isfinite(cost_to_go_crop)] = unknown_ctg_value

        else:
            raise ValueError(
                "Unknown cost_to_go_normalization: "
                f"{obs_config.cost_to_go_normalization}. "
                "Use 'local', 'global', or 'none'."
            )

    else:
        traversal_crop[~np.isfinite(traversal_crop)] = obs_config.unknown_cost_value
        cost_to_go_crop[~np.isfinite(cost_to_go_crop)] = obs_config.unknown_cost_value

    channels = []

    if obs_config.include_traversability_channel:
        channels.append(traversal_crop)

    channels.append(cost_to_go_crop)

    if obs_config.include_clearance_channel:
        clearance_crop_cells = sample_grid_nearest_vectorized(
            grid=nav_context.distance_to_obstacle,
            row_float=row_float,
            col_float=col_float,
            unknown_value=0.0,
        )

        clearance_crop = normalize_clearance_crop(
            clearance_crop_cells=clearance_crop_cells,
            resolution=resolution,
            max_clearance_m=obs_config.max_clearance_m,
            unknown_value=0.0,
        )

        channels.append(clearance_crop)

    if obs_config.include_goal_mask_channel:
        goal_mask = compute_goal_mask(
            robot_state=robot_state,
            nav_context=nav_context,
            crop_size=crop_size,
            resolution=resolution,
            sigma_cells=obs_config.goal_mask_sigma_cells,
        )

        channels.append(goal_mask)

    obs = np.stack(channels, axis=0).astype(np.float32)

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
