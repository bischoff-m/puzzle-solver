from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .plotting import plot_grid
from .types import Board, Piece


def demo_validate_patterns(board: Board, pieces: list[Piece]) -> None:
    sns.set_theme(style="white")
    board_grid = board.grid
    piece_grids = [(p.name, p.grid) for p in pieces]

    n_plots = 1 + len(piece_grids)
    ncols = 4
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 3.2 * nrows)
    )
    axes_list = list(np.ravel(axes))

    plot_grid(board_grid, ax=axes_list[0], title="Board (frame filled)")

    for i, (name, g) in enumerate(piece_grids, start=1):
        plot_grid(g, ax=axes_list[i], title=f"Piece {name}")

    for j in range(n_plots, len(axes_list)):
        axes_list[j].axis("off")

    fig.tight_layout()
    plt.show()
