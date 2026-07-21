import numpy as np
import matplotlib.pyplot as plt

def plot_predicted_waypoints_world(
    nn_predicted_waypoints_world,
    waypoint_draw_stride=25,
    label_prefix="Predicted waypoints",
):
    """
    Overlay predicted waypoint chains on the current matplotlib axes.

    nn_predicted_waypoints_world:
        list/array of length T
        each entry has shape (num_waypoints, 3)
        each waypoint is [x_world, y_world, theta_world]
    """
    if nn_predicted_waypoints_world is None:
        return

    if len(nn_predicted_waypoints_world) == 0:
        return

    waypoint_draw_stride = max(1, int(waypoint_draw_stride))

    first_chain = True
    first_tracked = True

    for t in range(0, len(nn_predicted_waypoints_world), waypoint_draw_stride):
        waypoints = np.asarray(nn_predicted_waypoints_world[t])

        if waypoints.ndim != 2:
            continue

        if waypoints.shape[0] == 0 or waypoints.shape[1] < 2:
            continue

        chain_label = label_prefix if first_chain else None

        plt.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            marker=".",
            markersize=3,
            linewidth=0.8,
            alpha=0.35,
            label=chain_label,
        )

        tracked_label = "Tracked predicted waypoint" if first_tracked else None

        plt.scatter(
            waypoints[0, 0],
            waypoints[0, 1],
            marker="x",
            s=30,
            alpha=0.7,
            label=tracked_label,
        )

        first_chain = False
        first_tracked = False

def plot_grid_path_and_rollout(
    problem,
    states_np=None,
    background="occupancy",
    show_rollout_arrows=True,
    show_path_arrows=True,
    arrow_stride=25,
    arrow_length=0.5,
):
    occupancy = problem["occupancy_grid"]
    path_world = problem["path_world"]
    start_world = problem["start_world"]
    goal_world = problem["goal_world"]
    resolution = problem["resolution"]
    origin_world = problem["origin_world"]

    height, width = occupancy.shape

    extent = [
        origin_world[0],
        origin_world[0] + width * resolution,
        origin_world[1],
        origin_world[1] + height * resolution,
    ]

    if background == "occupancy":
        display = occupancy
        cmap = "gray_r"
        title = "Occupancy grid"

    elif background == "traversal_cost":
        display = problem["traversal_cost"].copy()
        display[~np.isfinite(display)] = np.nan
        cmap = "viridis"
        title = "Traversal cost map"

    elif background == "cost_to_go":
        display = problem["cost_to_go"].copy()
        display[~np.isfinite(display)] = np.nan
        cmap = "viridis"
        title = "Dijkstra cost-to-go"

    elif background == "distance_to_obstacle":
        display = problem["distance_to_obstacle"].copy()
        display[~np.isfinite(display)] = np.nan
        cmap = "viridis"
        title = "Distance to obstacle"

    else:
        raise ValueError(
            f"Unknown background '{background}'. "
            "Use one of: occupancy, traversal_cost, cost_to_go, distance_to_obstacle."
        )

    plt.figure(figsize=(9, 9))

    plt.imshow(
        display,
        cmap=cmap,
        extent=extent,
        origin="lower",
    )

    if background != "occupancy":
        plt.colorbar(label=background)

    if path_world is not None and len(path_world) > 0:
        plt.plot(
            path_world[:, 0],
            path_world[:, 1],
            linewidth=2.0,
            label="Planned path",
        )

        if show_path_arrows and len(path_world) > arrow_stride:
            for i in range(0, len(path_world) - 1, arrow_stride):
                dx = path_world[i + 1, 0] - path_world[i, 0]
                dy = path_world[i + 1, 1] - path_world[i, 1]

                norm = np.sqrt(dx**2 + dy**2) + 1e-9
                dx = arrow_length * dx / norm
                dy = arrow_length * dy / norm

                plt.arrow(
                    path_world[i, 0],
                    path_world[i, 1],
                    dx,
                    dy,
                    head_width=0.15,
                    head_length=0.15,
                    length_includes_head=True,
                    alpha=0.8,
                )

    if states_np is not None and len(states_np) > 0:
        plt.plot(
            states_np[:, 0],
            states_np[:, 1],
            linewidth=2.0,
            linestyle="--",
            label="Executed rollout",
        )

        if show_rollout_arrows:
            for i in range(0, len(states_np), arrow_stride):
                x = states_np[i, 0]
                y = states_np[i, 1]
                theta = states_np[i, 2]

                fwd_dx = arrow_length * np.cos(theta)
                fwd_dy = arrow_length * np.sin(theta)

                left_dx = 0.6 * arrow_length * np.cos(theta + np.pi / 2.0)
                left_dy = 0.6 * arrow_length * np.sin(theta + np.pi / 2.0)

                plt.arrow(
                    x,
                    y,
                    fwd_dx,
                    fwd_dy,
                    head_width=0.15,
                    head_length=0.15,
                    length_includes_head=True,
                    alpha=0.95,
                )

                plt.arrow(
                    x,
                    y,
                    left_dx,
                    left_dy,
                    head_width=0.12,
                    head_length=0.12,
                    length_includes_head=True,
                    alpha=0.95,
                    linestyle="--",
                )

        plt.scatter(
            states_np[-1, 0],
            states_np[-1, 1],
            marker="s",
            s=70,
            label="Final robot pose",
        )

    plt.scatter(
        start_world[0],
        start_world[1],
        marker="o",
        s=90,
        label="Start",
    )

    plt.scatter(
        goal_world[0],
        goal_world[1],
        marker="x",
        s=120,
        label="Goal",
    )

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title + " with path and rollout")
    plt.legend()
    plt.show()


def plot_local_observation(obs):
    num_channels = obs.shape[0]

    channel_titles = [
        "Robot-frame traversal cost",
        "Robot-frame cost-to-go",
        "Robot-frame clearance",
        "Robot-frame goal mask",
    ]

    crop_size = obs.shape[1]
    robot_row = crop_size // 2
    robot_col = crop_size // 2

    arrow_len = crop_size * 0.18

    plt.figure(figsize=(5.5 * num_channels, 4.8))

    for i in range(num_channels):
        channel = obs[i]
        title = channel_titles[i] if i < len(channel_titles) else f"Channel {i}"

        plt.subplot(1, num_channels, i + 1)

        plt.imshow(channel, cmap="viridis", origin="upper")

        plt.scatter(
            robot_col,
            robot_row,
            marker="x",
            s=90,
            label="Robot",
        )

        plt.arrow(
            robot_col,
            robot_row,
            0,
            -arrow_len,
            head_width=2.5,
            head_length=3.0,
            length_includes_head=True,
        )

        plt.arrow(
            robot_col,
            robot_row,
            -arrow_len,
            0,
            head_width=2.5,
            head_length=3.0,
            length_includes_head=True,
            linestyle="--",
        )

        plt.text(robot_col + 2, robot_row - arrow_len, "F", fontsize=12)
        plt.text(robot_col - arrow_len - 5, robot_row + 4, "L", fontsize=12)
        plt.text(robot_col + arrow_len - 2, robot_row + 4, "R", fontsize=10)
        plt.text(robot_col + 2, robot_row + arrow_len + 4, "B", fontsize=10)

        plt.title(title + "\nrobot frame: forward = up, left = left")
        plt.xlabel("local image column")
        plt.ylabel("local image row")
        plt.colorbar()
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_expert_vs_nn_rollout(
    problem,
    pp_states_np,
    nn_states_np,
    title="Pure Pursuit vs Neural Policy",
    out_path=None,
    nn_predicted_waypoints_world=None,
    waypoint_draw_stride=25,
):
    occupancy = problem["occupancy_grid"]
    path_world = problem["path_world"]
    start_world = problem["start_world"]
    goal_world = problem["goal_world"]
    resolution = problem["resolution"]
    origin_world = problem["origin_world"]

    height, width = occupancy.shape

    extent = [
        origin_world[0],
        origin_world[0] + width * resolution,
        origin_world[1],
        origin_world[1] + height * resolution,
    ]

    plt.figure(figsize=(9, 9))

    plt.imshow(
        occupancy,
        cmap="gray_r",
        extent=extent,
        origin="lower",
    )

    plt.plot(
        path_world[:, 0],
        path_world[:, 1],
        linewidth=2.0,
        label="Dijkstra path",
    )

    plt.plot(
        pp_states_np[:, 0],
        pp_states_np[:, 1],
        linewidth=2.0,
        linestyle="--",
        label="Pure Pursuit rollout",
    )

    plt.plot(
        nn_states_np[:, 0],
        nn_states_np[:, 1],
        linewidth=2.0,
        linestyle="-.",
        label="Neural policy rollout",
    )

    plot_predicted_waypoints_world(
        nn_predicted_waypoints_world=nn_predicted_waypoints_world,
        waypoint_draw_stride=waypoint_draw_stride,
        label_prefix="Predicted waypoints",
    )

    plt.scatter(
        start_world[0],
        start_world[1],
        marker="o",
        s=90,
        label="Start",
    )

    plt.scatter(
        goal_world[0],
        goal_world[1],
        marker="x",
        s=120,
        label="Goal",
    )

    plt.scatter(
        pp_states_np[-1, 0],
        pp_states_np[-1, 1],
        marker="s",
        s=70,
        label="PP final",
    )

    plt.scatter(
        nn_states_np[-1, 0],
        nn_states_np[-1, 1],
        marker="D",
        s=70,
        label="NN final",
    )

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_expert_vs_nn_rollout_on_cost_to_go(
    problem,
    pp_states_np,
    nn_states_np,
    title="Pure Pursuit vs Neural Policy on Cost-to-Go",
    out_path=None,
    nn_predicted_waypoints_world=None,
    waypoint_draw_stride=25,
):
    cost_to_go = problem["cost_to_go"].copy()
    cost_to_go[~np.isfinite(cost_to_go)] = np.nan

    path_world = problem["path_world"]
    start_world = problem["start_world"]
    goal_world = problem["goal_world"]
    resolution = problem["resolution"]
    origin_world = problem["origin_world"]

    height, width = problem["occupancy_grid"].shape

    extent = [
        origin_world[0],
        origin_world[0] + width * resolution,
        origin_world[1],
        origin_world[1] + height * resolution,
    ]

    plt.figure(figsize=(9, 9))

    plt.imshow(
        cost_to_go,
        cmap="viridis",
        extent=extent,
        origin="lower",
    )

    plt.colorbar(label="Cost-to-go")

    plt.plot(
        path_world[:, 0],
        path_world[:, 1],
        linewidth=2.0,
        label="Dijkstra path",
    )

    plt.plot(
        pp_states_np[:, 0],
        pp_states_np[:, 1],
        linewidth=2.0,
        linestyle="--",
        label="Pure Pursuit rollout",
    )

    plt.plot(
        nn_states_np[:, 0],
        nn_states_np[:, 1],
        linewidth=2.0,
        linestyle="-.",
        label="Neural policy rollout",
    )

    plot_predicted_waypoints_world(
        nn_predicted_waypoints_world=nn_predicted_waypoints_world,
        waypoint_draw_stride=waypoint_draw_stride,
        label_prefix="Predicted waypoints",
    )

    plt.scatter(
        start_world[0],
        start_world[1],
        marker="o",
        s=90,
        label="Start",
    )

    plt.scatter(
        goal_world[0],
        goal_world[1],
        marker="x",
        s=120,
        label="Goal",
    )

    plt.scatter(
        pp_states_np[-1, 0],
        pp_states_np[-1, 1],
        marker="s",
        s=70,
        label="PP final",
    )

    plt.scatter(
        nn_states_np[-1, 0],
        nn_states_np[-1, 1],
        marker="D",
        s=70,
        label="NN final",
    )

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()

def compute_greedy_path_on_local_cost_to_go(
    cost_to_go_crop,
    start_row=None,
    start_col=None,
    max_steps=80,
    min_improvement=1e-4,
):
    """
    Greedy descent on a local cost-to-go crop.

    Starts from the robot center and repeatedly moves to the 8-connected
    neighbor with the lowest cost-to-go value.

    This is not a controller. It is only a diagnostic for whether the local
    cost-to-go patch suggests a simple greedy direction.
    """
    crop_size = cost_to_go_crop.shape[0]

    if start_row is None:
        start_row = crop_size // 2

    if start_col is None:
        start_col = crop_size // 2

    row = int(start_row)
    col = int(start_col)

    path = [(row, col)]

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for _ in range(max_steps):
        current_value = cost_to_go_crop[row, col]

        best_row = row
        best_col = col
        best_value = current_value

        for dr, dc in neighbor_offsets:
            nr = row + dr
            nc = col + dc

            if nr < 0 or nr >= crop_size or nc < 0 or nc >= crop_size:
                continue

            value = cost_to_go_crop[nr, nc]

            if value < best_value:
                best_value = value
                best_row = nr
                best_col = nc

        improvement = current_value - best_value

        if improvement <= min_improvement:
            break

        row = best_row
        col = best_col
        path.append((row, col))

    return np.array(path, dtype=np.int32)


def plot_local_greedy_diagnostic(
    obs,
    title="Local greedy cost-to-go diagnostic",
    out_path=None,
):
    """
    Plot local observation channels and overlay a greedy descent path on
    the local cost-to-go channel.

    Assumed channels:
      0 = traversal cost
      1 = cost-to-go
      2 = clearance, optional
      3 = goal mask, optional
    """
    num_channels = obs.shape[0]
    crop_size = obs.shape[1]

    robot_row = crop_size // 2
    robot_col = crop_size // 2

    cost_to_go_crop = obs[1]

    greedy_path = compute_greedy_path_on_local_cost_to_go(
        cost_to_go_crop=cost_to_go_crop,
        start_row=robot_row,
        start_col=robot_col,
        max_steps=80,
    )

    channel_titles = [
        "Traversal cost",
        "Cost-to-go + greedy descent",
        "Clearance",
        "Goal mask",
        "Path mask",
    ]

    plt.figure(figsize=(5.5 * num_channels, 5.0))

    for i in range(num_channels):
        channel = obs[i]
        channel_title = channel_titles[i] if i < len(channel_titles) else f"Channel {i}"

        plt.subplot(1, num_channels, i + 1)

        plt.imshow(channel, cmap="viridis", origin="upper")

        plt.scatter(
            robot_col,
            robot_row,
            marker="x",
            s=90,
            label="Robot",
        )

        if i == 1:
            plt.plot(
                greedy_path[:, 1],
                greedy_path[:, 0],
                linewidth=2.5,
                label="Greedy descent",
            )

            plt.scatter(
                greedy_path[-1, 1],
                greedy_path[-1, 0],
                marker="o",
                s=70,
                label="Greedy end",
            )

        arrow_len = crop_size * 0.18

        plt.arrow(
            robot_col,
            robot_row,
            0,
            -arrow_len,
            head_width=2.5,
            head_length=3.0,
            length_includes_head=True,
        )

        plt.arrow(
            robot_col,
            robot_row,
            -arrow_len,
            0,
            head_width=2.5,
            head_length=3.0,
            length_includes_head=True,
            linestyle="--",
        )

        plt.text(robot_col + 2, robot_row - arrow_len, "F", fontsize=12)
        plt.text(robot_col - arrow_len - 5, robot_row + 4, "L", fontsize=12)

        plt.title(channel_title)
        plt.xlabel("local image column")
        plt.ylabel("local image row")
        plt.colorbar()
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()

def compute_greedy_path_on_local_cost_to_go(
    cost_to_go_crop,
    start_row=None,
    start_col=None,
    max_steps=80,
    min_improvement=1e-4,
):
    crop_size = cost_to_go_crop.shape[0]

    if start_row is None:
        start_row = crop_size // 2

    if start_col is None:
        start_col = crop_size // 2

    row = int(start_row)
    col = int(start_col)

    path = [(row, col)]

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for _ in range(max_steps):
        current_value = cost_to_go_crop[row, col]

        best_row = row
        best_col = col
        best_value = current_value

        for dr, dc in neighbor_offsets:
            nr = row + dr
            nc = col + dc

            if nr < 0 or nr >= crop_size or nc < 0 or nc >= crop_size:
                continue

            value = cost_to_go_crop[nr, nc]

            if value < best_value:
                best_value = value
                best_row = nr
                best_col = nc

        improvement = current_value - best_value

        if improvement <= min_improvement:
            break

        row = best_row
        col = best_col
        path.append((row, col))

    return np.array(path, dtype=np.int32)


def local_crop_pixels_to_world(
    greedy_path_pixels,
    robot_state,
    resolution,
    crop_size,
):
    """
    Convert greedy path pixels from robot-frame crop coordinates to world coordinates.

    Pixel convention:
      row up    -> robot forward
      col left  -> robot left

    Body convention:
      x_body = forward
      y_body = left
    """
    half = crop_size // 2

    rows = greedy_path_pixels[:, 0].astype(np.float32)
    cols = greedy_path_pixels[:, 1].astype(np.float32)

    x_body = (half - rows) * resolution
    y_body = (half - cols) * resolution

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

    return np.stack([x_world, y_world], axis=1)


def plot_global_rollout_with_local_greedy_paths(
    problem,
    pp_states_np,
    nn_states_np,
    obs_config,
    timesteps,
    title="Global rollout with local greedy cost-to-go paths",
    out_path=None,
):
    from gridnav_il.geometry import RobotState
    from gridnav_il.controllers import make_nav_context
    from gridnav_il.observations import extract_local_grid_observation

    occupancy = problem["occupancy_grid"]
    path_world = problem["path_world"]
    start_world = problem["start_world"]
    goal_world = problem["goal_world"]
    resolution = problem["resolution"]
    origin_world = problem["origin_world"]

    nav_context = make_nav_context(problem)

    height, width = occupancy.shape

    extent = [
        origin_world[0],
        origin_world[0] + width * resolution,
        origin_world[1],
        origin_world[1] + height * resolution,
    ]

    plt.figure(figsize=(9, 9))

    plt.imshow(
        occupancy,
        cmap="gray_r",
        extent=extent,
        origin="lower",
    )

    plt.plot(
        path_world[:, 0],
        path_world[:, 1],
        linewidth=2.0,
        label="Dijkstra path",
    )

    plt.plot(
        pp_states_np[:, 0],
        pp_states_np[:, 1],
        linewidth=2.0,
        linestyle="--",
        label="Pure Pursuit rollout",
    )

    plt.plot(
        nn_states_np[:, 0],
        nn_states_np[:, 1],
        linewidth=2.0,
        linestyle="-.",
        label="Neural policy rollout",
    )

    first_greedy = True

    for t in timesteps:
        if t < 0 or t >= len(nn_states_np):
            continue

        state_np = nn_states_np[t]

        robot_state = RobotState(
            x=float(state_np[0]),
            y=float(state_np[1]),
            theta=float(state_np[2]),
        )

        obs = extract_local_grid_observation(
            robot_state=robot_state,
            nav_context=nav_context,
            obs_config=obs_config,
        )

        cost_to_go_crop = obs[1]

        greedy_path_pixels = compute_greedy_path_on_local_cost_to_go(
            cost_to_go_crop=cost_to_go_crop,
            max_steps=80,
        )

        greedy_path_world = local_crop_pixels_to_world(
            greedy_path_pixels=greedy_path_pixels,
            robot_state=robot_state,
            resolution=resolution,
            crop_size=obs_config.crop_size_cells,
        )

        label = "Local greedy descent" if first_greedy else None

        plt.plot(
            greedy_path_world[:, 0],
            greedy_path_world[:, 1],
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )

        plt.scatter(
            robot_state.x,
            robot_state.y,
            marker=".",
            s=30,
            alpha=0.8,
        )

        first_greedy = False

    plt.scatter(
        start_world[0],
        start_world[1],
        marker="o",
        s=90,
        label="Start",
    )

    plt.scatter(
        goal_world[0],
        goal_world[1],
        marker="x",
        s=120,
        label="Goal",
    )

    plt.scatter(
        pp_states_np[-1, 0],
        pp_states_np[-1, 1],
        marker="s",
        s=70,
        label="PP final",
    )

    plt.scatter(
        nn_states_np[-1, 0],
        nn_states_np[-1, 1],
        marker="D",
        s=70,
        label="NN final",
    )

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()