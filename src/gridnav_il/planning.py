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


def get_8_connected_neighbors(cell, height: int, width: int):
    row, col = cell

    candidates = [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
        (row - 1, col - 1),
        (row - 1, col + 1),
        (row + 1, col - 1),
        (row + 1, col + 1),
    ]

    valid = []

    for r, c in candidates:
        if 0 <= r < height and 0 <= c < width:
            valid.append((r, c))

    return valid


def get_neighbors(cell, height: int, width: int, connectivity: int):
    if connectivity == 4:
        return get_4_connected_neighbors(cell, height, width)

    if connectivity == 8:
        return get_8_connected_neighbors(cell, height, width)

    raise ValueError(f"Unknown connectivity: {connectivity}. Use 4 or 8.")


def move_distance(current_cell, neighbor_cell):
    row, col = current_cell
    nr, nc = neighbor_cell

    dr = abs(nr - row)
    dc = abs(nc - col)

    if dr == 1 and dc == 1:
        return np.sqrt(2.0)

    return 1.0


def dijkstra_cost_to_go(
    occupancy_grid,
    goal_cell,
    traversal_cost=None,
    connectivity: int = 4,
):
    height, width = occupancy_grid.shape

    if connectivity not in [4, 8]:
        raise ValueError(f"Unknown connectivity: {connectivity}. Use 4 or 8.")

    cost_to_go = np.full((height, width), np.inf, dtype=np.float32)

    if occupancy_grid[goal_cell] == 1:
        raise ValueError("Goal cell is occupied.")

    if traversal_cost is None:
        traversal_cost = np.ones((height, width), dtype=np.float32)
        traversal_cost[occupancy_grid == 1] = np.inf

    traversal_cost = traversal_cost.astype(np.float32)

    if connectivity == 4:
        neighbor_steps = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
        ]
    else:
        sqrt2 = float(np.sqrt(2.0))
        neighbor_steps = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, sqrt2),
            (-1, 1, sqrt2),
            (1, -1, sqrt2),
            (1, 1, sqrt2),
        ]

    cost_to_go[goal_cell] = 0.0

    pq = []
    heapq.heappush(pq, (0.0, goal_cell))

    while pq:
        current_cost, current_cell = heapq.heappop(pq)
        row, col = current_cell

        if current_cost > cost_to_go[row, col]:
            continue

        current_traversal_cost = traversal_cost[row, col]

        if not np.isfinite(current_traversal_cost):
            continue

        for dr, dc, move_dist in neighbor_steps:
            nr = row + dr
            nc = col + dc

            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue

            if occupancy_grid[nr, nc] == 1:
                continue

            neighbor_traversal_cost = traversal_cost[nr, nc]

            if not np.isfinite(neighbor_traversal_cost):
                continue

            step_cost = move_dist * 0.5 * (
                float(current_traversal_cost) + float(neighbor_traversal_cost)
            )

            new_cost = current_cost + step_cost

            if new_cost < cost_to_go[nr, nc]:
                cost_to_go[nr, nc] = new_cost
                heapq.heappush(pq, (new_cost, (nr, nc)))

    return cost_to_go


def extract_path_from_cost_to_go(
    cost_to_go,
    occupancy_grid,
    start_cell,
    goal_cell,
    connectivity: int = 4,
):
    height, width = occupancy_grid.shape

    if connectivity not in [4, 8]:
        raise ValueError(f"Unknown connectivity: {connectivity}. Use 4 or 8.")

    if not np.isfinite(cost_to_go[start_cell]):
        return None

    path = [start_cell]
    current = start_cell

    max_steps = height * width

    for _ in range(max_steps):
        if current == goal_cell:
            return path

        neighbors = get_neighbors(
            cell=current,
            height=height,
            width=width,
            connectivity=connectivity,
        )

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
