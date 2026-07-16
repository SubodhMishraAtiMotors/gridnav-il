import heapq
import numpy as np


def get_4_connected_neighbors(cell, height: int, width: int):
    row, col = cell

    candidates = [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ]

    valid = []

    for r, c in candidates:
        if 0 <= r < height and 0 <= c < width:
            valid.append((r, c))

    return valid


def dijkstra_cost_to_go(occupancy_grid, goal_cell, traversal_cost=None):
    height, width = occupancy_grid.shape

    cost_to_go = np.full((height, width), np.inf, dtype=np.float32)

    if occupancy_grid[goal_cell] == 1:
        raise ValueError("Goal cell is occupied.")

    if traversal_cost is None:
        traversal_cost = np.ones((height, width), dtype=np.float32)
        traversal_cost[occupancy_grid == 1] = np.inf

    cost_to_go[goal_cell] = 0.0

    pq = []
    heapq.heappush(pq, (0.0, goal_cell))

    while len(pq) > 0:
        current_cost, current_cell = heapq.heappop(pq)

        if current_cost > cost_to_go[current_cell]:
            continue

        for neighbor in get_4_connected_neighbors(current_cell, height, width):
            if occupancy_grid[neighbor] == 1:
                continue

            step_cost = traversal_cost[neighbor]

            if not np.isfinite(step_cost):
                continue

            new_cost = current_cost + step_cost

            if new_cost < cost_to_go[neighbor]:
                cost_to_go[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))

    return cost_to_go


def extract_path_from_cost_to_go(
    cost_to_go,
    occupancy_grid,
    start_cell,
    goal_cell,
):
    height, width = occupancy_grid.shape

    if not np.isfinite(cost_to_go[start_cell]):
        return None

    path = [start_cell]
    current = start_cell

    max_steps = height * width

    for _ in range(max_steps):
        if current == goal_cell:
            return path

        neighbors = get_4_connected_neighbors(current, height, width)

        valid_neighbors = [
            n for n in neighbors
            if occupancy_grid[n] == 0 and np.isfinite(cost_to_go[n])
        ]

        if len(valid_neighbors) == 0:
            return None

        next_cell = min(valid_neighbors, key=lambda n: cost_to_go[n])

        if cost_to_go[next_cell] >= cost_to_go[current]:
            return None

        path.append(next_cell)
        current = next_cell

    return None
