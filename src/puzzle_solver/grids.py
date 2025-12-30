from typing import TypeVar

from .types import Cell, Grid

T = TypeVar("T")


def _border_coords_clockwise(width: int, height: int) -> list[Cell]:
    """Clockwise border coordinates starting at top-left corner."""
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

    for y in (1, 2):
        for x in (1, 2):
            grid[y][x] = True

    for idx, (x, y) in enumerate(_PIECE_BORDER_COORDS_4X4):
        grid[y][x] = bool(border12[idx])

    return grid


def piece_dots_grid_from_dots12(dots12: tuple[int, ...]) -> list[list[int]]:
    """Return a 4x4 integer grid of dots.

    Inner 2x2 is always 0.
    """
    if len(dots12) != 12:
        raise ValueError("dots12 must have length 12")

    grid = [[0 for _ in range(4)] for _ in range(4)]

    for idx, (x, y) in enumerate(_PIECE_BORDER_COORDS_4X4):
        grid[y][x] = int(dots12[idx])

    return grid


def piece_border12_from_grid(grid: Grid) -> tuple[bool, ...]:
    """Convert a 4x4 piece grid back to its border12 representation."""
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
    """Return a 10x7 boolean grid describing pre-filled board voxels."""
    if len(border30) != 30:
        raise ValueError("border30 must have length 30")

    grid = [[False for _ in range(10)] for _ in range(7)]

    for idx, (x, y) in enumerate(_BOARD_BORDER_COORDS_10X7):
        grid[y][x] = bool(border30[idx])

    for y in range(1, 6):
        for x in range(1, 9):
            grid[y][x] = False

    return grid


def board_border30_from_grid(grid: Grid) -> tuple[bool, ...]:
    """Convert a 10x7 board grid back to its border30 representation."""
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


def rotate_grid_90_cw(grid: list[list[T]]) -> list[list[T]]:
    h = len(grid)
    w = len(grid[0])
    return [[grid[h - 1 - y][x] for y in range(h)] for x in range(w)]


def rotate_grid(grid: list[list[T]], k_cw: int) -> list[list[T]]:
    k = k_cw % 4
    out = grid
    for _ in range(k):
        out = rotate_grid_90_cw(out)
    return out


def flip_grid_x(grid: list[list[T]]) -> list[list[T]]:
    """Flip across x-axis (vertical flip in screen coordinates)."""
    return list(reversed([list(row) for row in grid]))


def flip_grid_y(grid: list[list[T]]) -> list[list[T]]:
    """Flip across y-axis (horizontal flip in screen coordinates)."""
    return [list(reversed(row)) for row in grid]


def grid_to_cells(grid: Grid) -> set[Cell]:
    cells: set[Cell] = set()
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v:
                cells.add((x, y))
    return cells
