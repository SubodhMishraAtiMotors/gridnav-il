import numpy as np
from scipy.ndimage import distance_transform_edt

from gridnav_il.geometry import grid_to_world
from gridnav_il.planning import dijkstra_cost_to_go, extract_path_from_cost_to_go


def create_random_grid_world(
    height=256,
    width=256,
    num_rectangles=60,
    min_rect_size=5,
    max_rect_size=30,
    seed=None,
):
    rng = np.random.default_rng(seed)

    occupancy = np.zeros((height, width), dtype=np.uint8)

    occupancy[0, :] = 1
    occupancy[-1, :] = 1
    occupancy[:, 0] = 1
    occupancy[:, -1] = 1

    for _ in range(num_rectangles):
        rect_h = int(rng.integers(min_rect_size, max_rect_size + 1))
        rect_w = int(rng.integers(min_rect_size, max_rect_size + 1))

        if rect_h >= height - 2 or rect_w >= width - 2:
            continue

        row = int(rng.integers(1, height - rect_h - 1))
        col = int(rng.integers(1, width - rect_w - 1))

        occupancy[row:row + rect_h, col:col + rect_w] = 1

    return occupancy


def compute_obstacle_proximity_cost(
    occupancy_grid,
    max_influence_distance_cells=8,
    obstacle_weight=8.0,
):
    free_mask = (occupancy_grid == 0).astype(np.uint8)

    distance_to_obstacle = distance_transform_edt(free_mask)

    normalized_distance = np.clip(
        distance_to_obstacle / max_influence_distance_cells,
        0.0,
        1.0,
    )

    proximity_penalty = obstacle_weight * (1.0 - normalized_distance) ** 2

    traversal_cost = 1.0 + proximity_penalty
    traversal_cost[occupancy_grid == 1] = np.inf

    return traversal_cost.astype(np.float32), distance_to_obstacle.astype(np.float32)


def sample_free_cell(occupancy_grid, rng):
    free_cells = np.argwhere(occupancy_grid == 0)

    if len(free_cells) == 0:
        raise RuntimeError("No free cells available.")

    idx = rng.integers(0, len(free_cells))
    row, col = free_cells[idx]

    return int(row), int(col)


def sample_start_goal_cells(
    occupancy_grid,
    min_separation_cells=30,
    seed=None,
):
    rng = np.random.default_rng(seed)

    for _ in range(1000):
        start = sample_free_cell(occupancy_grid, rng)
        goal = sample_free_cell(occupancy_grid, rng)

        dist = np.linalg.norm(np.array(start) - np.array(goal))

        if dist >= min_separation_cells:
            return start, goal

    raise RuntimeError("Could not sample valid start/goal with enough separation.")


def create_valid_planning_problem(
    height=256,
    width=256,
    num_rectangles=40,
    min_rect_size=6,
    max_rect_size=24,
    min_separation_cells=30,
    resolution=0.1,
    max_influence_distance_cells=8,
    obstacle_weight=8.0,
    seed=None,
    planner_connectivity: int = 4,
):
    rng = np.random.default_rng(seed)

    for _ in range(100):
        occupancy = create_random_grid_world(
            height=height,
            width=width,
            num_rectangles=num_rectangles,
            min_rect_size=min_rect_size,
            max_rect_size=max_rect_size,
            seed=int(rng.integers(0, 1_000_000_000)),
        )

        traversal_cost, distance_to_obstacle = compute_obstacle_proximity_cost(
            occupancy_grid=occupancy,
            max_influence_distance_cells=max_influence_distance_cells,
            obstacle_weight=obstacle_weight,
        )

        start_cell, goal_cell = sample_start_goal_cells(
            occupancy,
            min_separation_cells=min_separation_cells,
            seed=int(rng.integers(0, 1_000_000_000)),
        )

        cost_to_go = dijkstra_cost_to_go(
            occupancy_grid=occupancy,
            goal_cell=goal_cell,
            traversal_cost=traversal_cost,
            connectivity=planner_connectivity,
        )

        path_cells = extract_path_from_cost_to_go(
            cost_to_go=cost_to_go,
            occupancy_grid=occupancy,
            start_cell=start_cell,
            goal_cell=goal_cell,
            connectivity=planner_connectivity,
        )

        if path_cells is not None and len(path_cells) > 2:
            origin_world = np.array(
                [
                    -0.5 * width * resolution,
                    -0.5 * height * resolution,
                ],
                dtype=np.float32,
            )

            path_world = np.array(
                [
                    grid_to_world(cell, resolution, origin_world)
                    for cell in path_cells
                ],
                dtype=np.float32,
            )

            start_world = grid_to_world(start_cell, resolution, origin_world)
            goal_world = grid_to_world(goal_cell, resolution, origin_world)

            return {
                "occupancy_grid": occupancy,
                "traversal_cost": traversal_cost,
                "distance_to_obstacle": distance_to_obstacle,
                "cost_to_go": cost_to_go,
                "path_cells": path_cells,
                "path_world": path_world,
                "start_cell": start_cell,
                "goal_cell": goal_cell,
                "start_world": start_world,
                "goal_world": goal_world,
                "resolution": resolution,
                "origin_world": origin_world,
            }

    raise RuntimeError("Could not create valid planning problem.")


def sample_problem_kwargs(seed=None):
    rng = np.random.default_rng(seed)

    height = 256
    width = 256
    resolution = 0.1

    num_rectangles = int(rng.integers(25, 80))

    min_rect_size = int(rng.integers(4, 10))
    max_rect_size = int(rng.integers(12, 35))
    max_rect_size = max(max_rect_size, min_rect_size + 3)

    min_separation_cells = int(rng.integers(40, 120))

    max_influence_distance_cells = int(rng.integers(8, 14))
    obstacle_weight = float(rng.uniform(8.0, 20.0))

    return dict(
        height=height,
        width=width,
        num_rectangles=num_rectangles,
        min_rect_size=min_rect_size,
        max_rect_size=max_rect_size,
        min_separation_cells=min_separation_cells,
        resolution=resolution,
        max_influence_distance_cells=max_influence_distance_cells,
        obstacle_weight=obstacle_weight,
    )
