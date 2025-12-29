from pathlib import Path

import plotly.graph_objects as go
import yaml

from .cube_solver import solve_cube_pool
from .flat_solver import solve_flat_pool
from .grids import board_grid_from_border30, piece_grid_from_border12
from .plotting import plot_cube_solution, plot_flat_solution
from .types import Board, Piece
from .yaml_io import load_puzzle_yaml


def load_character_table_defaults(
    path: str | Path = "character_table_defaults.yaml",
) -> dict[str, object]:
    """Load default Character Table UI config from a YAML file.

    This is intended to be called from the Reflex backend when browser
    localStorage has no saved config.
    """

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
            n = int(float(v))
        except (TypeError, ValueError):
            n = fallback
        return max(1, n)

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
                    "shift": _int_ge_1(item.get("shift"), 1),
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
    path: str | Path = "character_table_defaults.yaml",
) -> None:
    """Write Character Table defaults to the backend YAML file."""

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
