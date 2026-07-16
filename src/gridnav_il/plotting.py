import numpy as np
import matplotlib.pyplot as plt


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
    traversal_crop = obs[0]
    cost_to_go_crop = obs[1]

    crop_size = traversal_crop.shape[0]
    robot_row = crop_size // 2
    robot_col = crop_size // 2

    arrow_len = crop_size * 0.18

    plt.figure(figsize=(11, 4.5))

    for i, (channel, title) in enumerate([
        (traversal_crop, "Robot-frame traversal cost"),
        (cost_to_go_crop, "Robot-frame cost-to-go"),
    ]):
        plt.subplot(1, 2, i + 1)

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
