from __future__ import annotations

import argparse
from pathlib import Path

from .cube_solver import solve_cube_pool
from .flat_solver import solve_flat, solve_flat_pool
from .grids import (
    board_grid_from_border30,
    piece_grid_from_border12,
)
from .interactive import interactive_solution_viewer
from .plotting import print_flat_solution
from .types import Board, BoardInput, Piece, PieceInput
from .yaml_io import (
    flip_puzzle_yaml_pieces_x_inplace,
    load_puzzle_yaml,
    write_puzzle_yaml_from_arrays,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="puzzle.yaml",
        help="Path to YAML puzzle definition (board.border30 + pieces[].border12)",
    )
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a template YAML (using example arrays) and exit",
    )
    parser.add_argument(
        "--flip-pieces-x-inplace",
        action="store_true",
        help="Flip all pieces across x-axis and overwrite the YAML config",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive viewer to browse flat/cube solutions",
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=20,
        help="Max solutions to collect per scenario (via Gurobi pool)",
    )
    args = parser.parse_args()

    example_piece_inputs = [
        PieceInput("A", tuple([True] * 12)),
        PieceInput("B", tuple([True] * 12)),
        PieceInput("C", tuple([True] * 12)),
        PieceInput("D", tuple([True] * 12)),
        PieceInput("E", tuple([True] * 12)),
        PieceInput("F", tuple([True] * 12)),
    ]
    example_board_input = BoardInput(tuple([True] * 30))

    if args.write_template:
        write_puzzle_yaml_from_arrays(
            args.config,
            board_border30=example_board_input.border30,
            pieces_border12={p.name: p.border12 for p in example_piece_inputs},
            overwrite=True,
        )
        print(f"Wrote template config to {args.config}")
        return

    if args.flip_pieces_x_inplace:
        flip_puzzle_yaml_pieces_x_inplace(args.config)
        print(f"Flipped pieces across x-axis in {args.config}")
        return

    config_path = Path(args.config)
    if config_path.exists():
        board_input, piece_inputs = load_puzzle_yaml(config_path)
    else:
        print(
            f"Config not found: {config_path}. Using built-in example arrays."
        )
        board_input, piece_inputs = example_board_input, example_piece_inputs

    pieces = [
        Piece(p.name, piece_grid_from_border12(p.border12))
        for p in piece_inputs
    ]
    board = Board(board_grid_from_border30(board_input.border30))

    if args.interactive:
        flat_solutions = solve_flat_pool(
            board, pieces, max_solutions=args.max_solutions, output_flag=0
        )
        cube_solutions = solve_cube_pool(
            pieces, max_solutions=args.max_solutions, output_flag=0
        )
        interactive_solution_viewer(
            board=board,
            pieces=pieces,
            flat_solutions=flat_solutions,
            cube_solutions=cube_solutions,
        )
        return

    flat_sol = solve_flat(board, pieces)
    print_flat_solution(board, flat_sol)


if __name__ == "__main__":
    main()
