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


def piece_dots12_from_dots_grid(dots_grid: list[list[int]]) -> tuple[int, ...]:
    """Convert a 4x4 dots grid back to its border dots12 encoding."""
    if len(dots_grid) != 4 or any(len(row) != 4 for row in dots_grid):
        raise ValueError("dots_grid must be 4x4")
    out = [0] * 12
    for idx, (x, y) in enumerate(_PIECE_BORDER_COORDS_4X4):
        out[idx] = int(dots_grid[y][x])
    return tuple(out)


def piece_dots_side_grid_from_dots16(
    dots16: tuple[int, ...],
) -> list[list[tuple[int, int, int, int]]]:
    """Return a 4x4 grid of side dots (North, East, South, West).

    Mapping of dots16 (16 values) to the 4 edges:
    - dots16[0:4]   -> North faces of (0,0), (1,0), (2,0), (3,0)
    - dots16[4:8]   -> East faces of (3,0), (3,1), (3,2), (3,3)
    - dots16[8:12]  -> South faces of (3,3), (2,3), (1,3), (0,3)
    - dots16[12:16] -> West faces of (0,3), (0,2), (0,1), (0,0)
    """
    if len(dots16) != 16:
        raise ValueError("dots16 must have length 16")

    # Initialize 4x4 grid of (0,0,0,0)
    grid = [[(0, 0, 0, 0) for _ in range(4)] for _ in range(4)]

    def _set_side(x: int, y: int, side_idx: int, val: int):
        n, e, s, w = grid[y][x]
        if side_idx == 0:
            n = val
        elif side_idx == 1:
            e = val
        elif side_idx == 2:
            s = val
        elif side_idx == 3:
            w = val
        grid[y][x] = (n, e, s, w)

    # North: y=0, x=0..3
    for i in range(4):
        _set_side(i, 0, 0, int(dots16[i]))
    # East: x=3, y=0..3
    for i in range(4):
        _set_side(3, i, 1, int(dots16[4 + i]))
    # South: y=3, x=3..0
    for i in range(4):
        _set_side(3 - i, 3, 2, int(dots16[8 + i]))
    # West: x=0, y=3..0
    for i in range(4):
        _set_side(0, 3 - i, 3, int(dots16[12 + i]))

    return grid


def piece_dots16_from_side_grid(
    dots_side_grid: list[list[tuple[int, int, int, int]]],
) -> tuple[int, ...]:
    """Convert a 4x4 side dots grid back to row-major dots16 encoding."""
    if len(dots_side_grid) != 4 or any(len(row) != 4 for row in dots_side_grid):
        raise ValueError("dots_side_grid must be 4x4")

    out = [0] * 16
    # North: y=0, x=0..3
    for i in range(4):
        out[i] = dots_side_grid[0][i][0]
    # East: x=3, y=0..3
    for i in range(4):
        out[4 + i] = dots_side_grid[i][3][1]
    # South: y=3, x=3..0
    for i in range(4):
        out[8 + i] = dots_side_grid[3][3 - i][2]
    # West: x=0, y=3..0
    for i in range(4):
        out[12 + i] = dots_side_grid[3 - i][0][3]

    return tuple(out)


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


def rotate_side_grid(
    grid: list[list[tuple[int, int, int, int]]], k_cw: int
) -> list[list[tuple[int, int, int, int]]]:
    k = k_cw % 4
    out = grid
    for _ in range(k):
        # Rotate elements in the grid
        out = rotate_grid_90_cw(out)
        # Rotate the (N, E, S, W) tuples: New N = Old W, New E = Old N, etc.
        # (N, E, S, W) -> (W, N, E, S)
        new_out = []
        for row in out:
            new_row = []
            for n, e, s, w in row:
                new_row.append((w, n, e, s))
            new_out.append(new_row)
        out = new_out
    return out


def flip_grid_x(grid: list[list[T]]) -> list[list[T]]:
    """Flip across x-axis (vertical flip in screen coordinates)."""
    return list(reversed([list(row) for row in grid]))


def flip_grid_y(grid: list[list[T]]) -> list[list[T]]:
    """Flip across y-axis (horizontal flip in screen coordinates)."""
    return [list(reversed(row)) for row in grid]


def flip_side_grid_x(
    grid: list[list[tuple[int, int, int, int]]],
) -> list[list[tuple[int, int, int, int]]]:
    """Vertical flip of a side grid."""
    # Flip elements in the grid
    out = flip_grid_x(grid)
    # Flip the (N, E, S, W) tuples: New N = Old S, New S = Old N
    # (N, E, S, W) -> (S, E, N, W)
    new_out = []
    for row in out:
        new_row = []
        for n, e, s, w in row:
            new_row.append((s, e, n, w))
        new_out.append(new_row)
    return new_out


def flip_side_grid_y(
    grid: list[list[tuple[int, int, int, int]]],
) -> list[list[tuple[int, int, int, int]]]:
    """Horizontal flip of a side grid."""
    # Flip elements in the grid
    out = flip_grid_y(grid)
    # Flip the (N, E, S, W) tuples: New E = Old W, New W = Old E
    # (N, E, S, W) -> (N, W, S, E)
    new_out = []
    for row in out:
        new_row = []
        for n, e, s, w in row:
            new_row.append((n, w, s, e))
        new_out.append(new_row)
    return new_out


def grid_to_cells(grid: Grid) -> set[Cell]:
    cells: set[Cell] = set()
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v:
                cells.add((x, y))
    return cells
