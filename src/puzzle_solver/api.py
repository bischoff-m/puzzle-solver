from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from .cube_solver import solve_cube_pool
from .flat_solver import solve_flat_pool
from .grids import board_grid_from_border30, piece_grid_from_border12
from .plotting import plot_cube_solution, plot_flat_solution
from .types import Board, Piece
from .yaml_io import load_puzzle_yaml


def load_puzzle(path: str | Path = "puzzle.yaml") -> tuple[Board, list[Piece]]:
    """Load a puzzle YAML file and construct the concrete Board/Piece objects."""
    board_input, piece_inputs = load_puzzle_yaml(path)
    pieces = [
        Piece(p.name, piece_grid_from_border12(p.border12))
        for p in piece_inputs
    ]
    board = Board(board_grid_from_border30(board_input.border30))
    return board, pieces


def solve_and_plot_flat(
    *,
    path: str | Path = "puzzle.yaml",
    max_solutions: int = 1,
    output_flag: int = 0,
    solution_index: int = 0,
) -> tuple[go.Figure, dict[str, set[tuple[int, int]]]]:
    """Solve the flat puzzle and return (figure, raw_solution)."""
    board, pieces = load_puzzle(path)
    sols = solve_flat_pool(
        board, pieces, max_solutions=max_solutions, output_flag=output_flag
    )
    if not sols:
        raise RuntimeError("No flat solutions returned")
    i = max(0, min(solution_index, len(sols) - 1))
    sol = sols[i]
    fig = plot_flat_solution(board, sol)
    return fig, sol


def solve_and_plot_cube(
    *,
    path: str | Path = "puzzle.yaml",
    max_solutions: int = 1,
    output_flag: int = 0,
    solution_index: int = 0,
) -> tuple[go.Figure, dict[str, tuple[str, int]]]:
    """Solve the cube puzzle and return (figure, raw_solution)."""
    _board, pieces = load_puzzle(path)
    sols = solve_cube_pool(
        pieces, max_solutions=max_solutions, output_flag=output_flag
    )
    if not sols:
        raise RuntimeError("No cube solutions returned")
    i = max(0, min(solution_index, len(sols) - 1))
    sol = sols[i]
    fig = plot_cube_solution(pieces, sol)
    # Keep the solution return type JSON-friendly for frontend use.
    sol_out: dict[str, tuple[str, int]] = {
        k: (v[0], v[1]) for k, v in sol.items()
    }
    return fig, sol_out
