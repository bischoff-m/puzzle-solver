from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, to_rgba

from .cube_solver import _enumerate_cube_placements
from .grids import grid_to_cells
from .types import Board, Cell, Grid, Piece, Voxel


def plot_grid(
    grid: Grid,
    *,
    ax: Axes,
    title: str,
    true_color: str = "#222222",
    false_color: str = "#ffffff",
) -> None:
    """Plot a boolean grid with square voxels using seaborn."""
    data = np.array(grid, dtype=int)
    cmap = ListedColormap([false_color, true_color])
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        square=True,
        linewidths=0.8,
        linecolor="#cccccc",
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")


def print_flat_solution(board: Board, solution: dict[str, set[Cell]]) -> None:
    """Plot the flat-board solution with seaborn."""
    sns.set_theme(style="white")

    board_filled = grid_to_cells(board.grid)
    w, h = 10, 7

    piece_items = sorted(solution.items(), key=lambda kv: kv[0])
    piece_index: dict[str, int] = {
        name: i + 1 for i, (name, _) in enumerate(piece_items)
    }
    frame_value = len(piece_items) + 1

    grid_int = np.zeros((h, w), dtype=int)
    for x, y in board_filled:
        grid_int[y, x] = frame_value

    for name, occ in piece_items:
        v = piece_index[name]
        for x, y in occ:
            grid_int[y, x] = v

    piece_colors = sns.color_palette("tab10", n_colors=max(6, len(piece_items)))
    cmap_list: list[tuple[float, float, float] | str] = ["#ffffff"]
    cmap_list.extend([piece_colors[i] for i in range(len(piece_items))])
    cmap_list.append("#c0c0c0")
    cmap = ListedColormap(cmap_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        grid_int,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=frame_value,
        cbar=False,
        square=True,
        linewidths=0.8,
        linecolor="#cccccc",
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_aspect("equal")
    ax.set_title("Flat solution")
    fig.tight_layout()
    plt.show()


def draw_flat_solution(
    ax: Axes, board: Board, solution: dict[str, set[Cell]]
) -> None:
    """Draw flat solution onto an existing Matplotlib Axes."""
    board_filled = grid_to_cells(board.grid)
    w, h = 10, 7

    piece_items = sorted(solution.items(), key=lambda kv: kv[0])
    piece_index: dict[str, int] = {
        name: i + 1 for i, (name, _) in enumerate(piece_items)
    }
    frame_value = len(piece_items) + 1

    grid_int = np.zeros((h, w), dtype=int)
    for x, y in board_filled:
        grid_int[y, x] = frame_value
    for name, occ in piece_items:
        v = piece_index[name]
        for x, y in occ:
            grid_int[y, x] = v

    piece_colors = sns.color_palette("tab10", n_colors=max(6, len(piece_items)))
    cmap_list: list[tuple[float, float, float] | str] = ["#ffffff"]
    cmap_list.extend([piece_colors[i] for i in range(len(piece_items))])
    cmap_list.append("#c0c0c0")
    cmap = ListedColormap(cmap_list)

    ax.clear()
    sns.heatmap(
        grid_int,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=frame_value,
        cbar=False,
        square=True,
        linewidths=0.8,
        linecolor="#cccccc",
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_aspect("equal")
    ax.set_title("Flat")


def draw_cube_solution(
    ax, pieces: list[Piece], solution: dict[str, tuple[str, int]]
) -> None:
    """Draw cube solution onto an existing 3D Axes."""
    piece_by_name: dict[str, Piece] = {p.name: p for p in pieces}

    filled = np.zeros((4, 4, 4), dtype=bool)
    facecolors = np.empty((4, 4, 4), dtype=object)

    names_sorted = sorted(solution.keys())
    palette = sns.color_palette("tab10", n_colors=max(6, len(names_sorted)))
    name_to_rgba = {
        name: to_rgba(palette[i], alpha=1.0)
        for i, name in enumerate(names_sorted)
    }

    for name in names_sorted:
        face, rot = solution[name]
        occ: set[Voxel] | None = None
        for f, r, voxels in _enumerate_cube_placements(
            piece_by_name[name].grid
        ):
            if f == face and r == rot:
                occ = voxels
                break
        if occ is None:
            raise RuntimeError(
                f"Could not reconstruct placement for piece {name}"
            )
        rgba = name_to_rgba[name]
        for x, y, z in occ:
            filled[x, y, z] = True
            facecolors[x, y, z] = rgba

    ax.clear()
    ax.voxels(filled, facecolors=facecolors, edgecolor="#222222", linewidth=0.6)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_title("Cube")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
