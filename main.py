from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gurobipy import GRB
from matplotlib.colors import ListedColormap

Cell = tuple[int, int]
Voxel = tuple[int, int, int]
Face = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


@dataclass(frozen=True)
class PieceSpec:
    name: str
    border12: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.border12) != 12:
            raise ValueError(f"Piece {self.name}: border12 must have length 12")


@dataclass(frozen=True)
class BoardSpec:
    border30: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.border30) != 30:
            raise ValueError("Board border30 must have length 30")


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


def piece_grid_from_border12(border12: tuple[bool, ...]) -> list[list[bool]]:
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


def board_grid_from_border30(border30: tuple[bool, ...]) -> list[list[bool]]:
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


def rotate_grid_90_cw(grid: list[list[bool]]) -> list[list[bool]]:
    h = len(grid)
    w = len(grid[0])
    return [[grid[h - 1 - y][x] for y in range(h)] for x in range(w)]


def rotate_grid(grid: list[list[bool]], k_cw: int) -> list[list[bool]]:
    k = k_cw % 4
    out = grid
    for _ in range(k):
        out = rotate_grid_90_cw(out)
    return out


def grid_to_cells(grid: list[list[bool]]) -> set[Cell]:
    cells: set[Cell] = set()
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v:
                cells.add((x, y))
    return cells


def print_grid(
    grid: list[list[bool]], *, true_char: str = "#", false_char: str = "."
) -> None:
    for row in grid:
        print("".join(true_char if v else false_char for v in row))


def print_piece(piece: PieceSpec) -> None:
    print(f"Piece {piece.name}")
    grid = piece_grid_from_border12(piece.border12)
    print_grid(grid)


def print_board(board: BoardSpec) -> None:
    print("Board (True=pre-filled frame, False=missing)")
    grid = board_grid_from_border30(board.border30)
    print_grid(grid)


def plot_grid(
    grid: list[list[bool]],
    *,
    ax: plt.Axes,
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
    piece_grid: list[list[bool]],
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


def solve_flat(
    board: BoardSpec, pieces: list[PieceSpec]
) -> dict[str, set[Cell]]:
    board_grid = board_grid_from_border30(board.border30)
    board_w, board_h = 10, 7
    board_filled = grid_to_cells(board_grid)
    board_all = {(x, y) for y in range(board_h) for x in range(board_w)}
    target_cells = board_all - board_filled

    piece_grids = {p.name: piece_grid_from_border12(p.border12) for p in pieces}
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


def print_flat_solution(
    board: BoardSpec, solution: dict[str, set[Cell]]
) -> None:
    board_grid = board_grid_from_border30(board.border30)
    board_filled = grid_to_cells(board_grid)
    w, h = 10, 7
    # Start with frame.
    view = [
        ["#" if (x, y) in board_filled else "." for x in range(w)]
        for y in range(h)
    ]

    # Overlay pieces with distinct letters.
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, (name, occ) in enumerate(
        sorted(solution.items(), key=lambda kv: kv[0])
    ):
        ch = letters[idx % len(letters)]
        for x, y in occ:
            view[y][x] = ch

    for row in view:
        print("".join(row))


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
    piece_grid: list[list[bool]],
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


def solve_cube(pieces: list[PieceSpec]) -> dict[str, tuple[Face, int]]:
    # Boundary voxels of a 4x4x4 cube.
    boundary: set[Voxel] = set()
    for z in range(4):
        for y in range(4):
            for x in range(4):
                if x in (0, 3) or y in (0, 3) or z in (0, 3):
                    boundary.add((x, y, z))

    piece_grids = {p.name: piece_grid_from_border12(p.border12) for p in pieces}
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


def demo_validate_patterns(board: BoardSpec, pieces: list[PieceSpec]) -> None:
    sns.set_theme(style="white")
    board_grid = board_grid_from_border30(board.border30)
    piece_grids = [
        (p.name, piece_grid_from_border12(p.border12)) for p in pieces
    ]

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
    # TODO: Replace this example data with your real puzzle definition.
    # Piece border12 uses the following order (clockwise starting at (0,0)):
    # (0,0)(1,0)(2,0)(3,0)(3,1)(3,2)(3,3)(2,3)(1,3)(0,3)(0,2)(0,1)
    example_pieces = [
        PieceSpec("A", tuple([True] * 12)),
        PieceSpec("B", tuple([True] * 12)),
        PieceSpec("C", tuple([True] * 12)),
        PieceSpec("D", tuple([True] * 12)),
        PieceSpec("E", tuple([True] * 12)),
        PieceSpec("F", tuple([True] * 12)),
    ]

    # Board border30 order is clockwise starting at (0,0) along the 10x7 frame.
    example_board = BoardSpec(tuple([True] * 30))

    demo_validate_patterns(example_board, example_pieces)

    # Uncomment to solve once you replace example data with real data.
    # flat_sol = solve_flat(example_board, example_pieces)
    # print_flat_solution(example_board, flat_sol)
    # cube_sol = solve_cube(example_pieces)
    # print(cube_sol)


if __name__ == "__main__":
    main()
