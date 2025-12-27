from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from gurobipy import GRB
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

Cell = tuple[int, int]
Voxel = tuple[int, int, int]
Face = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
Grid = list[list[bool]]


@dataclass(frozen=True)
class PieceInput:
    name: str
    border12: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.border12) != 12:
            raise ValueError(f"Piece {self.name}: border12 must have length 12")


@dataclass(frozen=True)
class BoardInput:
    border30: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.border30) != 30:
            raise ValueError("Board border30 must have length 30")


@dataclass(frozen=True)
class Piece:
    name: str
    grid: Grid

    def __post_init__(self) -> None:
        if len(self.grid) != 4 or any(len(row) != 4 for row in self.grid):
            raise ValueError(f"Piece {self.name}: grid must be 4x4")


@dataclass(frozen=True)
class Board:
    grid: Grid

    def __post_init__(self) -> None:
        if len(self.grid) != 7 or any(len(row) != 10 for row in self.grid):
            raise ValueError("Board grid must be 7x10")


def _coerce_bool_list(
    values: object, *, expected_len: int, label: str
) -> tuple[bool, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a list")
    if len(values) != expected_len:
        raise ValueError(f"{label} must have length {expected_len}")

    out: list[bool] = []
    for i, v in enumerate(values):
        if isinstance(v, bool):
            out.append(v)
        elif isinstance(v, int):
            out.append(v != 0)
        elif isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "t", "1", "yes", "y"}:
                out.append(True)
            elif s in {"false", "f", "0", "no", "n"}:
                out.append(False)
            else:
                raise ValueError(f"{label}[{i}] invalid boolean string: {v!r}")
        else:
            raise TypeError(
                f"{label}[{i}] must be bool/int/str, got {type(v).__name__}"
            )
    return tuple(out)


def load_puzzle_yaml(path: str | Path) -> tuple[BoardInput, list[PieceInput]]:
    """Load puzzle input arrays from a YAML file.

    Expected schema (recommended):

    ```yaml
    version: 1
    board:
      border30: [true, false, ...]   # length 30
    pieces:
      - name: A
        border12: [true, ...]        # length 12
      - name: B
        border12: [...]
    ```

    Also accepted for `pieces`:
      - mapping form: {A: [..12..], B: [..12..], ...}
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")

    board_node = raw.get("board")
    if not isinstance(board_node, dict):
        raise ValueError("YAML must contain mapping key 'board'")
    border30 = _coerce_bool_list(
        board_node.get("border30"), expected_len=30, label="board.border30"
    )
    board = BoardInput(border30=border30)

    pieces_node = raw.get("pieces")
    if pieces_node is None:
        raise ValueError("YAML must contain key 'pieces'")

    pieces: list[PieceInput] = []
    if isinstance(pieces_node, dict):
        for name, arr in pieces_node.items():
            if not isinstance(name, str):
                raise ValueError("pieces mapping keys must be strings")
            border12 = _coerce_bool_list(
                arr, expected_len=12, label=f"pieces.{name}"
            )
            pieces.append(PieceInput(name=name, border12=border12))
    elif isinstance(pieces_node, list):
        for idx, item in enumerate(pieces_node):
            if not isinstance(item, dict):
                raise ValueError(f"pieces[{idx}] must be a mapping")
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"pieces[{idx}].name must be a non-empty string"
                )
            border12 = _coerce_bool_list(
                item.get("border12"),
                expected_len=12,
                label=f"pieces[{idx}].border12",
            )
            pieces.append(PieceInput(name=name, border12=border12))
    else:
        raise ValueError("pieces must be a list or mapping")

    if len(pieces) != 6:
        raise ValueError(f"Expected 6 pieces, got {len(pieces)}")
    names = [p.name for p in pieces]
    if len(set(names)) != len(names):
        raise ValueError("Piece names must be unique")

    return board, pieces


def dump_puzzle_yaml(board: BoardInput, pieces: list[PieceInput]) -> str:
    """Construct a YAML document (as string) from the input border arrays."""
    doc = {
        "version": 1,
        "board": {"border30": list(board.border30)},
        "pieces": [
            {"name": p.name, "border12": list(p.border12)}
            for p in sorted(pieces, key=lambda x: x.name)
        ],
    }
    return yaml.safe_dump(doc, sort_keys=False)


def write_puzzle_yaml_from_arrays(
    path: str | Path,
    *,
    board_border30: tuple[bool, ...],
    pieces_border12: dict[str, tuple[bool, ...]],
    overwrite: bool = False,
) -> None:
    """Write a YAML config file from border arrays.

    This is the requested helper to construct the YAML file from boolean arrays.
    """
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")
    board = BoardInput(border30=board_border30)
    pieces = [
        PieceInput(name=k, border12=v) for k, v in pieces_border12.items()
    ]
    p.write_text(dump_puzzle_yaml(board, pieces), encoding="utf-8")


def flip_puzzle_yaml_pieces_x_inplace(path: str | Path) -> None:
    """Load puzzle YAML, flip each piece grid across the x-axis, write back.

    This fixes the case where `border12` was specified with the top/bottom swapped.
    Board is left unchanged.
    """
    path = Path(path)
    board_input, piece_inputs = load_puzzle_yaml(path)

    flipped_inputs: list[PieceInput] = []
    for p in piece_inputs:
        grid = piece_grid_from_border12(p.border12)
        flipped_grid = flip_grid_x(grid)
        flipped_border12 = piece_border12_from_grid(flipped_grid)
        flipped_inputs.append(
            PieceInput(name=p.name, border12=flipped_border12)
        )

    path.write_text(
        dump_puzzle_yaml(board_input, flipped_inputs), encoding="utf-8"
    )


def _border_coords_clockwise(width: int, height: int) -> list[Cell]:
    """Clockwise border coordinates starting at top-left corner.

    Ordering:
      - Top row: (0,0) .. (w-1,0)
      - Right col: (w-1,1) .. (w-1,h-2)
      - Bottom row: (w-1,h-1) .. (0,h-1)
      - Left col: (0,h-2) .. (0,1)
    """
    if width < 2 or height < 2:
        raise ValueError("width and height must be >= 2")
    coords: list[Cell] = []
    coords.extend((x, 0) for x in range(width))
    coords.extend((width - 1, y) for y in range(1, height - 1))
    coords.extend((x, height - 1) for x in range(width - 1, -1, -1))
    coords.extend((0, y) for y in range(height - 2, 0, -1))
    return coords


_PIECE_BORDER_COORDS_4X4: list[Cell] = _border_coords_clockwise(4, 4)
assert len(_PIECE_BORDER_COORDS_4X4) == 12

_BOARD_BORDER_COORDS_10X7: list[Cell] = _border_coords_clockwise(10, 7)
assert len(_BOARD_BORDER_COORDS_10X7) == 30


def piece_grid_from_border12(border12: tuple[bool, ...]) -> Grid:
    """Return a 4x4 boolean grid.

    Convention: True = voxel present.
    Inner 2x2 is always present, border is controlled by border12.
    """
    if len(border12) != 12:
        raise ValueError("border12 must have length 12")

    grid = [[False for _ in range(4)] for _ in range(4)]

    # Inner 2x2 always filled.
    for y in (1, 2):
        for x in (1, 2):
            grid[y][x] = True

    for idx, (x, y) in enumerate(_PIECE_BORDER_COORDS_4X4):
        grid[y][x] = bool(border12[idx])

    return grid


def piece_border12_from_grid(grid: Grid) -> tuple[bool, ...]:
    """Convert a 4x4 piece grid back to its border12 representation.

    Validates that the inner 2x2 is filled (True).
    """
    if len(grid) != 4 or any(len(row) != 4 for row in grid):
        raise ValueError("piece grid must be 4x4")
    for y in (1, 2):
        for x in (1, 2):
            if not grid[y][x]:
                raise ValueError("piece grid must have inner 2x2 filled")
    border = [False] * 12
    for idx, (x, y) in enumerate(_PIECE_BORDER_COORDS_4X4):
        border[idx] = bool(grid[y][x])
    return tuple(border)


def board_grid_from_border30(border30: tuple[bool, ...]) -> Grid:
    """Return a 10x7 boolean grid describing *pre-filled* board voxels.

    Convention: True = already filled by the board frame.
    Inner 8x5 (x=1..8,y=1..5) is always missing (False).
    Border cells are controlled by border30.
    """
    if len(border30) != 30:
        raise ValueError("border30 must have length 30")

    grid = [[False for _ in range(10)] for _ in range(7)]

    for idx, (x, y) in enumerate(_BOARD_BORDER_COORDS_10X7):
        grid[y][x] = bool(border30[idx])

    # Ensure inner is missing.
    for y in range(1, 6):
        for x in range(1, 9):
            grid[y][x] = False

    return grid


def board_border30_from_grid(grid: Grid) -> tuple[bool, ...]:
    """Convert a 10x7 board grid back to its border30 representation.

    Validates that the inner 8x5 cavity is missing (False).
    """
    if len(grid) != 7 or any(len(row) != 10 for row in grid):
        raise ValueError("board grid must be 7x10")
    for y in range(1, 6):
        for x in range(1, 9):
            if grid[y][x]:
                raise ValueError(
                    "board grid must have inner 8x5 missing (False)"
                )

    border = [False] * 30
    for idx, (x, y) in enumerate(_BOARD_BORDER_COORDS_10X7):
        border[idx] = bool(grid[y][x])
    return tuple(border)


def rotate_grid_90_cw(grid: Grid) -> Grid:
    h = len(grid)
    w = len(grid[0])
    return [[grid[h - 1 - y][x] for y in range(h)] for x in range(w)]


def rotate_grid(grid: Grid, k_cw: int) -> Grid:
    k = k_cw % 4
    out = grid
    for _ in range(k):
        out = rotate_grid_90_cw(out)
    return out


def flip_grid_x(grid: Grid) -> Grid:
    """Flip a 2D grid across the x-axis.

    With our (x,y) convention where y increases downward, this is a vertical flip
    (top row swaps with bottom row).
    """
    return list(reversed([list(row) for row in grid]))


def grid_to_cells(grid: Grid) -> set[Cell]:
    cells: set[Cell] = set()
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v:
                cells.add((x, y))
    return cells


def plot_grid(
    grid: Grid,
    *,
    ax: Axes,
    title: str,
    true_color: str = "#222222",
    false_color: str = "#ffffff",
) -> None:
    """Plot a boolean grid with square voxels using seaborn.

    Convention: True = filled voxel (dark), False = missing voxel (white).
    """
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


def _enumerate_flat_placements(
    piece_grid: Grid,
    board_w: int,
    board_h: int,
    board_filled: set[Cell],
) -> list[set[Cell]]:
    """All non-overlapping placements for one piece on the board.

    A placement is the set of board cells occupied by the piece.
    Rotations: 0,90,180,270 degrees (no mirror flips).
    """
    placements: list[set[Cell]] = []

    for rot in range(4):
        g = rotate_grid(piece_grid, rot)
        cells = grid_to_cells(g)
        # Size is still 4x4 after rotation.
        for oy in range(0, board_h - 4 + 1):
            for ox in range(0, board_w - 4 + 1):
                placed = {(ox + x, oy + y) for (x, y) in cells}
                if placed & board_filled:
                    continue
                placements.append(placed)

    # De-duplicate placements that happen to be identical (symmetry in the pattern).
    uniq: list[set[Cell]] = []
    seen: set[frozenset[Cell]] = set()
    for p in placements:
        fp = frozenset(p)
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append(p)
    return uniq


def solve_flat(board: Board, pieces: list[Piece]) -> dict[str, set[Cell]]:
    board_grid = board.grid
    board_w, board_h = 10, 7
    board_filled = grid_to_cells(board_grid)
    board_all = {(x, y) for y in range(board_h) for x in range(board_w)}
    target_cells = board_all - board_filled

    piece_grids = {p.name: p.grid for p in pieces}
    placements_by_piece: dict[str, list[set[Cell]]] = {
        name: _enumerate_flat_placements(g, board_w, board_h, board_filled)
        for name, g in piece_grids.items()
    }

    m = gp.Model("puzzle_flat")
    m.Params.OutputFlag = 1

    xvar: dict[tuple[str, int], gp.Var] = {}
    for name, placements in placements_by_piece.items():
        for pi in range(len(placements)):
            xvar[(name, pi)] = m.addVar(
                vtype=GRB.BINARY, name=f"x[{name},{pi}]"
            )

    # Each piece placed exactly once.
    for name, placements in placements_by_piece.items():
        m.addConstr(
            gp.quicksum(xvar[(name, pi)] for pi in range(len(placements))) == 1,
            name=f"place_once[{name}]",
        )

    # Exact cover of target (missing) cells.
    for cell in sorted(target_cells):
        m.addConstr(
            gp.quicksum(
                xvar[(name, pi)]
                for name, placements in placements_by_piece.items()
                for pi, occ in enumerate(placements)
                if cell in occ
            )
            == 1,
            name=f"cover[{cell[0]},{cell[1]}]",
        )

    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"No optimal solution found; Gurobi status={m.Status}"
        )

    solution: dict[str, set[Cell]] = {}
    for name, placements in placements_by_piece.items():
        for pi, occ in enumerate(placements):
            if xvar[(name, pi)].X > 0.5:
                solution[name] = occ
                break
        if name not in solution:
            raise RuntimeError(f"Piece {name} not assigned in solution")

    return solution


def print_flat_solution(board: Board, solution: dict[str, set[Cell]]) -> None:
    """Plot the flat-board solution with seaborn.

    Frame voxels are shown in a neutral gray; each piece gets its own color.
    """
    sns.set_theme(style="white")

    board_grid = board.grid
    board_filled = grid_to_cells(board_grid)
    w, h = 10, 7

    # Encode as integers for plotting:
    # 0 = empty, 1..N = piece id, N+1 = frame
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

    # Build a colormap: [empty] + piece colors + [frame]
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


def _face_map(face: Face, u: int, v: int) -> Voxel:
    """Map (u,v) in 0..3x0..3 on a face to cube coordinates (x,y,z)."""
    if face == "+X":
        return (3, u, v)
    if face == "-X":
        return (0, u, v)
    if face == "+Y":
        return (u, 3, v)
    if face == "-Y":
        return (u, 0, v)
    if face == "+Z":
        return (u, v, 3)
    if face == "-Z":
        return (u, v, 0)
    raise ValueError(f"Unknown face: {face}")


def _enumerate_cube_placements(
    piece_grid: Grid,
) -> list[tuple[Face, int, set[Voxel]]]:
    """All (face, rotation) placements of a 4x4 piece on the 4x4x4 cube boundary."""
    placements: list[tuple[Face, int, set[Voxel]]] = []
    faces: list[Face] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    for face in faces:
        for rot in range(4):
            g = rotate_grid(piece_grid, rot)
            occ: set[Voxel] = set()
            for v in range(4):
                for u in range(4):
                    if g[v][u]:
                        occ.add(_face_map(face, u, v))
            placements.append((face, rot, occ))
    return placements


def solve_cube(pieces: list[Piece]) -> dict[str, tuple[Face, int]]:
    # Boundary voxels of a 4x4x4 cube.
    boundary: set[Voxel] = set()
    for z in range(4):
        for y in range(4):
            for x in range(4):
                if x in (0, 3) or y in (0, 3) or z in (0, 3):
                    boundary.add((x, y, z))

    piece_grids = {p.name: p.grid for p in pieces}
    placements_by_piece: dict[str, list[tuple[Face, int, set[Voxel]]]] = {
        name: _enumerate_cube_placements(g) for name, g in piece_grids.items()
    }

    m = gp.Model("puzzle_cube")
    m.Params.OutputFlag = 1

    xvar: dict[tuple[str, int], gp.Var] = {}
    for name, placements in placements_by_piece.items():
        for pi in range(len(placements)):
            xvar[(name, pi)] = m.addVar(
                vtype=GRB.BINARY, name=f"x[{name},{pi}]"
            )

    # Each piece selects exactly one (face, rotation).
    for name, placements in placements_by_piece.items():
        m.addConstr(
            gp.quicksum(xvar[(name, pi)] for pi in range(len(placements))) == 1,
            name=f"place_once[{name}]",
        )

    # Exact cover on the boundary.
    for voxel in sorted(boundary):
        m.addConstr(
            gp.quicksum(
                xvar[(name, pi)]
                for name, placements in placements_by_piece.items()
                for pi, (_face, _rot, occ) in enumerate(placements)
                if voxel in occ
            )
            == 1,
            name=f"cover[{voxel[0]},{voxel[1]},{voxel[2]}]",
        )

    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"No optimal solution found; Gurobi status={m.Status}"
        )

    sol: dict[str, tuple[Face, int]] = {}
    for name, placements in placements_by_piece.items():
        for pi, (face, rot, _occ) in enumerate(placements):
            if xvar[(name, pi)].X > 0.5:
                sol[name] = (face, rot)
                break
        if name not in sol:
            raise RuntimeError(f"Piece {name} not assigned in solution")
    return sol


def demo_validate_patterns(board: Board, pieces: list[Piece]) -> None:
    sns.set_theme(style="white")
    board_grid = board.grid
    piece_grids = [(p.name, p.grid) for p in pieces]

    # 1 board + N pieces
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

    # Hide unused axes.
    for j in range(n_plots, len(axes_list)):
        axes_list[j].axis("off")

    fig.tight_layout()
    plt.show()


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
    args = parser.parse_args()

    # Example arrays used for template generation / fallback.
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

    # Convert input arrays -> matrices ONCE. Use matrices after this point.
    pieces = [
        Piece(p.name, piece_grid_from_border12(p.border12))
        for p in piece_inputs
    ]
    board = Board(board_grid_from_border30(board_input.border30))

    # demo_validate_patterns(board, pieces)

    # Uncomment to solve once you replace example data with real data.
    flat_sol = solve_flat(board, pieces)
    print_flat_solution(board, flat_sol)
    # cube_sol = solve_cube(pieces)
    # print(cube_sol)


if __name__ == "__main__":
    main()
