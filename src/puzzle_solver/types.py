from dataclasses import dataclass
from typing import Literal

Cell = tuple[int, int]
Voxel = tuple[int, int, int]
Face = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
Grid = list[list[bool]]


@dataclass(frozen=True)
class PieceInput:
    name: str
    border12: tuple[bool, ...]
    dots: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.border12) != 12:
            raise ValueError(f"Piece {self.name}: border12 must have length 12")
        if self.dots is not None and len(self.dots) != 12:
            raise ValueError(f"Piece {self.name}: dots must have length 12")


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
    dots_grid: list[list[int]] | None = None

    def __post_init__(self) -> None:
        if len(self.grid) != 4 or any(len(row) != 4 for row in self.grid):
            raise ValueError(f"Piece {self.name}: grid must be 4x4")
        if self.dots_grid is not None:
            if len(self.dots_grid) != 4 or any(
                len(row) != 4 for row in self.dots_grid
            ):
                raise ValueError(f"Piece {self.name}: dots_grid must be 4x4")


@dataclass(frozen=True)
class Board:
    grid: Grid

    def __post_init__(self) -> None:
        if len(self.grid) != 7 or any(len(row) != 10 for row in self.grid):
            raise ValueError("Board grid must be 7x10")
