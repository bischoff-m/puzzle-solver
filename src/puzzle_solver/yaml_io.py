from pathlib import Path

import yaml

from .grids import (
    flip_grid_x,
    flip_side_grid_x,
    piece_border12_from_grid,
    piece_dots12_from_dots_grid,
    piece_dots16_from_side_grid,
    piece_dots_grid_from_dots12,
    piece_dots_side_grid_from_dots16,
    piece_grid_from_border12,
)
from .types import BoardInput, PieceInput


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


def _coerce_int_list(
    values: object, *, expected_len: int, label: str
) -> tuple[int, ...]:
    if values is None:
        return tuple([0] * expected_len)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a list")
    if len(values) != expected_len:
        raise ValueError(f"{label} must have length {expected_len}")

    out: list[int] = []
    for i, v in enumerate(values):
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            raise TypeError(
                f"{label}[{i}] must be convertible to int, got {type(v).__name__}"
            )
    return tuple(out)


def _validate_dots_range(values: tuple[int, ...], *, label: str) -> None:
    for i, v in enumerate(values):
        if not (0 <= int(v) <= 6):
            raise ValueError(f"{label}[{i}] must be in 0..6, got {v}")


def flip_dots16_reverse_shift(dots16: tuple[int, ...]) -> tuple[int, ...]:
    """Flip the 16-dot edge encoding by reversing and shifting -1.

    This is consistent with flip_border12_reverse_shift.
    """
    if len(dots16) != 16:
        raise ValueError("dots16 must have length 16")
    rev = list(reversed(dots16))
    # shift -1 (right by one)
    rev = [rev[-1]] + rev[:-1]
    return tuple(rev)


def flip_border12_reverse_shift(
    border12: tuple,
) -> tuple:
    """Flip the border12 encoding by reversing and shifting -1.

    The border12 array is a clockwise border starting at the top-left.
    When a piece is flipped on the table, the border orientation changes.
    The requested conversion is:

    1) reverse the array
    2) cyclic shift by -1 (left by one)
    """

    if len(border12) != 12:
        raise ValueError("border12 must have length 12")
    rev = list(reversed(border12))
    # shift -1 (right by one)
    rev = [rev[-1]] + rev[:-1]
    return tuple(rev)


def flip_piece_border12_reverse_shift(
    border12: tuple[bool, ...],
) -> tuple[bool, ...]:
    return tuple(bool(x) for x in flip_border12_reverse_shift(border12))


def load_puzzle_yaml(
    path: str | Path,
) -> tuple[BoardInput, list[PieceInput], bool]:
    """Load puzzle input arrays from a YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")

    is_flipped = bool(raw.get("isFlipped", False))

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
            dots_top = _coerce_int_list(
                item.get("dotsTop", item.get("dots")),
                expected_len=12,
                label=f"pieces[{idx}].dotsTop",
            )
            _validate_dots_range(dots_top, label=f"pieces[{idx}].dotsTop")
            dots_side16 = _coerce_int_list(
                item.get("dotsSide"),
                expected_len=16,
                label=f"pieces[{idx}].dotsSide",
            )
            _validate_dots_range(dots_side16, label=f"pieces[{idx}].dotsSide")
            pieces.append(
                PieceInput(
                    name=name,
                    border12=border12,
                    dots=dots_top,
                    dots_side16=dots_side16,
                )
            )
    else:
        raise ValueError("pieces must be a list or mapping")

    if len(pieces) != 6:
        raise ValueError(f"Expected 6 pieces, got {len(pieces)}")
    names = [p.name for p in pieces]
    if len(set(names)) != len(names):
        raise ValueError("Piece names must be unique")

    return board, pieces, is_flipped


def dump_puzzle_yaml(
    board: BoardInput,
    pieces: list[PieceInput],
    *,
    is_flipped: bool = False,
) -> str:
    """Construct a YAML document (as string) from the input border arrays."""
    doc = {
        "version": 1,
        "isFlipped": bool(is_flipped),
        "board": {"border30": list(board.border30)},
        "pieces": [
            {
                "name": p.name,
                "border12": list(p.border12),
                "dotsTop": list(p.dots) if p.dots else [0] * 12,
                "dotsSide": list(p.dots_side16) if p.dots_side16 else [0] * 16,
            }
            for p in sorted(pieces, key=lambda x: x.name)
        ],
    }
    return yaml.safe_dump(doc, sort_keys=False)


def write_puzzle_yaml_from_arrays(
    path: str | Path,
    *,
    board_border30: tuple[bool, ...],
    pieces_border12: dict[str, tuple[bool, ...]],
    is_flipped: bool = False,
    overwrite: bool = False,
) -> None:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")
    board = BoardInput(border30=board_border30)
    pieces = [
        PieceInput(name=k, border12=v) for k, v in pieces_border12.items()
    ]
    p.write_text(
        dump_puzzle_yaml(board, pieces, is_flipped=bool(is_flipped)),
        encoding="utf-8",
    )


def flip_puzzle_yaml_pieces_x_inplace(path: str | Path) -> None:
    """Load puzzle YAML, flip each piece grid across the x-axis, write back."""
    path = Path(path)
    board_input, piece_inputs, is_flipped = load_puzzle_yaml(path)

    flipped_inputs: list[PieceInput] = []
    for p in piece_inputs:
        grid = piece_grid_from_border12(p.border12)
        flipped_grid = flip_grid_x(grid)
        flipped_border12 = piece_border12_from_grid(flipped_grid)

        dots12 = p.dots if p.dots is not None else tuple([0] * 12)
        dots_grid = piece_dots_grid_from_dots12(dots12)
        flipped_dots12 = piece_dots12_from_dots_grid(flip_grid_x(dots_grid))

        dots_side16 = (
            p.dots_side16 if p.dots_side16 is not None else tuple([0] * 16)
        )
        dots_side_grid = piece_dots_side_grid_from_dots16(dots_side16)
        flipped_dots_side16 = piece_dots16_from_side_grid(
            flip_side_grid_x(dots_side_grid)
        )
        flipped_inputs.append(
            PieceInput(
                name=p.name,
                border12=flipped_border12,
                dots=flipped_dots12,
                dots_side16=flipped_dots_side16,
            )
        )

    path.write_text(
        dump_puzzle_yaml(board_input, flipped_inputs, is_flipped=is_flipped),
        encoding="utf-8",
    )


def rotate_puzzle_yaml_pieces_inplace(path: str | Path, steps: int = 1) -> None:
    """Rotate each piece by steps * 90 degrees and write back.

    steps=1: +90 (CW), steps=-1: -90 (CCW).
    """
    path = Path(path)
    board_input, piece_inputs, is_flipped = load_puzzle_yaml(path)

    rotated_inputs: list[PieceInput] = []
    for p in piece_inputs:
        # border12 and dotsTop (12 elements) -> shift by 3 * steps
        s12 = (3 * steps) % 12
        b12 = list(p.border12)
        b12 = b12[-s12:] + b12[:-s12]

        d12 = list(p.dots) if p.dots is not None else [0] * 12
        d12 = d12[-s12:] + d12[:-s12]

        # dotsSide (16 elements) -> shift by 4 * steps
        s16 = (4 * steps) % 16
        d16 = list(p.dots_side16) if p.dots_side16 is not None else [0] * 16
        d16 = d16[-s16:] + d16[:-s16]

        rotated_inputs.append(
            PieceInput(
                name=p.name,
                border12=tuple(b12),
                dots=tuple(d12),
                dots_side16=tuple(d16),
            )
        )

    path.write_text(
        dump_puzzle_yaml(board_input, rotated_inputs, is_flipped=is_flipped),
        encoding="utf-8",
    )
