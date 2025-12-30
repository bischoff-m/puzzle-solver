from pathlib import Path

import plotly.graph_objects as go
import yaml

from .cube_solver import solve_cube_pool
from .flat_solver import solve_flat_pool
from .grids import (
    board_grid_from_border30,
    piece_dots_grid_from_dots12,
    piece_dots_side_grid_from_dots16,
    piece_grid_from_border12,
)
from .plotting import plot_cube_solution, plot_flat_solution
from .types import Board, Cell, Face, Piece
from .yaml_io import (
    flip_border12_reverse_shift,
    flip_dots16_reverse_shift,
    flip_piece_border12_reverse_shift,
    load_puzzle_yaml,
)


def _repo_root() -> Path:
    # src/puzzle_solver/api.py -> src -> repo root
    return Path(__file__).resolve().parents[2]


def puzzles_assets_dir() -> Path:
    return _repo_root() / "assets" / "puzzles"


def list_puzzle_assets() -> list[str]:
    """List available puzzle YAML filenames under assets/puzzles."""

    d = puzzles_assets_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.yaml") if p.is_file())


def resolve_puzzle_asset(name: str) -> Path:
    """Resolve a puzzle YAML within assets/puzzles by filename.

    Only allows selecting files directly under assets/puzzles (no path
    separators) to avoid arbitrary file access.
    """

    n = str(name or "").strip()
    if not n:
        raise ValueError("Puzzle name is empty")
    if "/" in n or "\\" in n or n.startswith("."):
        raise ValueError(f"Invalid puzzle name: {n!r}")
    if not n.lower().endswith(".yaml"):
        n = f"{n}.yaml"

    p = puzzles_assets_dir() / n
    if not p.exists():
        raise FileNotFoundError(f"Puzzle not found: {n}")
    return p


def character_table_defaults_path() -> Path:
    return _repo_root() / "assets" / "character_table_defaults.yaml"


def load_character_table_defaults(
    path: str | Path | None = None,
) -> dict[str, object]:
    """Load default Character Table UI config from a YAML file.

    This is intended to be called from the Reflex backend when browser
    localStorage has no saved config.
    """
    if path is None:
        path = character_table_defaults_path()

    defaults: dict[str, object] = {
        "text": "",
        "table_width": 20,
        "code_word": "",
        "show_encrypted": False,
        "punch_cards": [
            {
                "x": 1,
                "y": 1,
                "flipped": False,
                "isActive": True,
                "shift": 1,
                "word": "",
            }
        ],
    }

    p = Path(path)
    if not p.exists():
        return defaults

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return defaults

    out = dict(defaults)

    text = raw.get("text")
    if isinstance(text, str):
        out["text"] = text

    def _int_ge_1(v: object, fallback: int) -> int:
        try:
            if isinstance(v, bool) or v is None:
                n = fallback
            elif isinstance(v, int):
                n = v
            elif isinstance(v, float):
                n = int(v)
            elif isinstance(v, str):
                n = int(float(v))
            else:
                n = fallback
        except (TypeError, ValueError):
            n = fallback
        return max(1, n)

    def _int(v: object, fallback: int) -> int:
        try:
            if isinstance(v, bool) or v is None:
                n = fallback
            elif isinstance(v, int):
                n = v
            elif isinstance(v, float):
                n = int(v)
            elif isinstance(v, str):
                n = int(float(v))
            else:
                n = fallback
        except (TypeError, ValueError):
            n = fallback
        return n

    if "table_width" in raw:
        out["table_width"] = _int_ge_1(raw.get("table_width"), 20)

    cw = raw.get("code_word")
    if isinstance(cw, str) and cw.strip() != "":
        out["code_word"] = cw.upper()

    if "show_encrypted" in raw:
        out["show_encrypted"] = bool(raw.get("show_encrypted"))

    # Backward compatibility: accept the old field, but ignore it.
    # (Older configs stored only a length; the UI now uses the full code word.)

    pcs = raw.get("punch_cards")
    if isinstance(pcs, list):
        cards: list[dict[str, object]] = []
        for item in pcs:
            if not isinstance(item, dict):
                continue
            cards.append(
                {
                    "x": _int_ge_1(item.get("x"), 1),
                    "y": _int_ge_1(item.get("y"), 1),
                    "flipped": bool(item.get("flipped", False)),
                    "isActive": bool(item.get("isActive", True)),
                    "shift": _int(item.get("shift"), 1),
                    "word": item.get("word")
                    if isinstance(item.get("word"), str)
                    else "",
                }
            )
        if cards:
            out["punch_cards"] = cards

    return out


def save_character_table_defaults(
    config: dict[str, object],
    path: str | Path | None = None,
) -> None:
    """Write Character Table defaults to the backend YAML file."""
    if path is None:
        path = character_table_defaults_path()

    class _Dumper(yaml.SafeDumper):
        pass

    def _str_representer(dumper: yaml.SafeDumper, data: str):
        # Prefer a literal block for multi-line text to avoid unreadable quoting
        # (e.g., doubled apostrophes) and to preserve Unicode as-is.
        if "\n" in data:
            return dumper.represent_scalar(
                "tag:yaml.org,2002:str", data, style="|"
            )
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    _Dumper.add_representer(str, _str_representer)

    p = Path(path)
    doc = dict(config)
    # Keep the file schema stable.
    doc.setdefault("version", 2)

    # Make the default file stable + readable.
    text = doc.get("text")
    if isinstance(text, str):
        # Avoid trailing whitespace/newlines turning into confusing empty lines.
        doc["text"] = text.replace("\r\n", "\n").rstrip()

    data = yaml.dump(
        doc,
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )
    p.write_text(data, encoding="utf-8")


def load_puzzle(path: str | Path = "puzzle.yaml") -> tuple[Board, list[Piece]]:
    """Load a puzzle YAML file and construct the concrete Board/Piece objects."""
    board, pieces, _is_flipped = load_puzzle_with_meta(path)
    return board, pieces


def load_puzzle_with_meta(
    path: str | Path = "puzzle.yaml",
) -> tuple[Board, list[Piece], bool]:
    """Load a puzzle and also return its YAML config metadata."""

    board_input, piece_inputs, is_flipped = load_puzzle_yaml(path)

    pieces: list[Piece] = []
    for p in piece_inputs:
        border12 = p.border12
        dots12 = p.dots if p.dots is not None else tuple([0] * 12)
        dots_side16 = (
            p.dots_side16 if p.dots_side16 is not None else tuple([0] * 16)
        )
        if bool(is_flipped):
            border12 = flip_piece_border12_reverse_shift(border12)
            dots12 = flip_border12_reverse_shift(dots12)
            dots_side16 = flip_dots16_reverse_shift(dots_side16)

        pieces.append(
            Piece(
                p.name,
                piece_grid_from_border12(border12),
                piece_dots_grid_from_dots12(dots12),
                piece_dots_side_grid_from_dots16(dots_side16),
            )
        )

    board = Board(board_grid_from_border30(board_input.border30))
    return board, pieces, bool(is_flipped)


def get_puzzle_is_flipped(name: str) -> bool:
    p = resolve_puzzle_asset(name)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return False
    return bool(raw.get("isFlipped", False))


def set_puzzle_is_flipped(name: str, *, is_flipped: bool) -> None:
    p = resolve_puzzle_asset(name)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")

    raw["isFlipped"] = bool(is_flipped)

    data = yaml.safe_dump(
        raw,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )
    p.write_text(data, encoding="utf-8")


def solve_and_plot_flat(
    *,
    path: str | Path = "puzzle.yaml",
    max_solutions: int = 1,
    output_flag: int = 0,
    solution_index: int = 0,
) -> tuple[go.Figure, dict[str, dict[Cell, int]]]:
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


def get_cube_solution_shift(
    pieces: list[Piece], solution: dict[str, tuple[Face, int]]
) -> int:
    """Count dots on the bottom face (-Z) of the cube solution."""
    from .plotting import _voxels_from_solution

    voxels_by_piece = _voxels_from_solution(pieces, solution)
    total_dots = 0
    for _name, (_face, occ) in voxels_by_piece.items():
        for (_x, _y, z), d_dict in occ.items():
            if z == 0:  # Bottom layer
                total_dots += d_dict.get("-Z", 0)
    return total_dots
