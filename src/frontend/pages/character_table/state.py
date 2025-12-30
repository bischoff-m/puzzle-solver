import json
import re
from typing import Any

import numpy as np
import plotly.graph_objects as go
import reflex as rx
from pydantic import BaseModel

from puzzle_solver.api import (
    load_character_table_defaults,
    save_character_table_defaults,
)
from puzzle_solver.plotting import qualitative_palette

from .punch_card import mask as punch_mask


class PunchCardConfig(BaseModel):
    x: int = 1
    y: int = 1
    flipped: bool = False
    is_active: bool = True
    shift: int = 1
    word: str = ""


def _discrete_colorscale(colors: list[str]) -> list[tuple[float, str]]:
    if not colors:
        raise ValueError("colors must be non-empty")
    n = len(colors)
    if n == 1:
        return [(0.0, colors[0]), (1.0, colors[0])]

    scale: list[tuple[float, str]] = []
    for i, c in enumerate(colors):
        lo = i / n
        hi = (i + 1) / n
        scale.append((lo, c))
        scale.append((hi, c))
    scale[0] = (0.0, scale[0][1])
    scale[-1] = (1.0, scale[-1][1])
    return scale


def _hex_to_rgba(color: str, *, alpha: float) -> str:
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Unsupported color format: {color!r}")
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def preprocess_text(text: str) -> str:
    # Normalize all whitespace (including line breaks) to single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation.
    text = re.sub(r"[^\w\s]", "", text)
    return text.upper()


def compute_character_alphabet(text: str, *, code_word: str = "") -> list[str]:
    """Compute an ordered alphabet for shifting.

    - Starts with A-Z in order (so 'A' is always first).
    - Then appends any remaining unique characters from the (preprocessed) text
      and code word in deterministic (Unicode) order.
    """

    letters = [chr(ord("A") + i) for i in range(26)]
    letters_set = set(letters)

    unique = set(text)
    unique.update(code_word)

    umlauts = ["Ä", "Ö", "Ü"]
    umlauts_present = [
        u for u in umlauts if u in unique and u not in letters_set
    ]

    remaining = sorted(
        ch
        for ch in unique
        if ch not in letters_set and ch not in set(umlauts_present)
    )
    return letters + umlauts_present + remaining


def _shift_over_alphabet(
    text: str,
    *,
    alphabet: list[str],
    index: dict[str, int],
    key_indices: list[int],
) -> str:
    if not alphabet:
        return text

    if not key_indices:
        key_indices = [0]

    out: list[str] = []
    n = len(alphabet)
    for i, ch in enumerate(text):
        ci = index.get(ch)
        if ci is None:
            out.append(ch)
            continue
        shift = int(key_indices[i % len(key_indices)])
        out.append(alphabet[(ci + shift) % n])
    return "".join(out)


def vigenere_encrypt(
    text: str, *, code_word: str, backward: bool = False
) -> str:
    """Encrypt text using a Vigenere-style shift over the computed alphabet."""

    code_word = (code_word or "A").upper()
    alphabet = compute_character_alphabet(text, code_word=code_word)
    index = {ch: i for i, ch in enumerate(alphabet)}

    key_indices = [index.get(ch, 0) for ch in code_word]
    if backward:
        key_indices = [-idx for idx in key_indices]

    return _shift_over_alphabet(
        text,
        alphabet=alphabet,
        index=index,
        key_indices=key_indices,
    )


def build_character_mapping_figure(
    *,
    text: str,
    code_word: str,
    theme: str,
) -> go.Figure:
    """Build a table-style grid showing shifts for each code word position.

    Rows: text character (alphabet)
    Columns: shift character (alphabet)
    Cells: shifted character (Caesar-style)

    The first header row shows the shift character (A, B, C, ...).
    The second header row shows the shift index (0, 1, 2, ...).
    """

    code_word = str(code_word or "").upper()
    theme = str(theme or "light").lower()

    cell_px = 28
    gap_px = 1
    margin = dict(l=10, r=10, t=10, b=10)

    if not text:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["Character mapping"]),
                    cells=dict(values=[[""]]),
                )
            ]
        )
        fig.update_layout(margin=margin, autosize=False, height=120, width=320)
        return fig

    alphabet = compute_character_alphabet(text, code_word=code_word)
    idx = {ch: i for i, ch in enumerate(alphabet)}
    n = len(alphabet)

    # Columns represent all possible shifts (by index):
    # column 0: A/0, column 1: B/1, ...
    shifts = list(range(n))

    # Build a table-like grid by inserting header rows/columns.
    # grid rows = 2 (header rows) + len(alphabet)
    # grid cols = 1 (row label col) + len(alphabet)
    grid_rows = 2 + len(alphabet)
    grid_cols = 1 + len(alphabet)

    z: list[list[int]] = []
    annotations: list[dict] = []
    shapes: list[dict] = []

    if theme == "dark":
        cell_bg = "rgba(255,255,255,0.15)"
        label_bg = "rgba(255,255,255,0.25)"
        grid_bg = "rgba(0,0,0,1)"
        text_color = "white"
        separator_color = "rgba(255,255,255,0.4)"
    else:
        cell_bg = "rgba(0,0,0,0.15)"
        label_bg = "rgba(0,0,0,0.25)"
        grid_bg = "rgba(255,255,255,1)"
        text_color = "black"
        separator_color = "rgba(0,0,0,0.4)"
    # Use z to differentiate label cells vs mapping cells.
    colorscale = [(0.0, cell_bg), (1.0, label_bg)]

    for r in range(grid_rows):
        z_row: list[int] = []
        for c in range(grid_cols):
            is_label_cell = (r <= 1) or (c == 0)
            z_row.append(1 if is_label_cell else 0)

            # Determine displayed value for this cell.
            if r == 0 and c == 0:
                value = ""
            elif r == 0 and c > 0:
                # Shift character header row.
                value = alphabet[c - 1]
            elif r == 1 and c == 0:
                value = ""
            elif r == 1 and c > 0:
                # Shift index header row.
                value = str(c - 1)
            elif r >= 2 and c == 0:
                # Row label.
                value = alphabet[r - 2]
            else:
                base = alphabet[r - 2]
                shift = shifts[c - 1]
                value = alphabet[(idx[base] + shift) % n]

            if value != "":
                annotations.append(
                    dict(
                        x=c,
                        y=r,
                        text=value,
                        showarrow=False,
                        font=dict(size=14, color=text_color),
                    )
                )
        z.append(z_row)

    # Thicker borders separating labels (top rows + left col) from data cells.
    # Use filled rectangles (same style as punch-card highlighting) so they
    # remain visible regardless of zoom/gaps.
    bar_half = 0.04
    # Vertical separator between label column (c=0) and data columns.
    shapes.append(
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=0.5 - bar_half,
            x1=0.5 + bar_half,
            y0=-0.5,
            y1=grid_rows - 0.5,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor=separator_color,
            layer="above",
        )
    )
    # Horizontal separator between header rows (r=0,1) and data rows.
    shapes.append(
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=-0.5,
            x1=grid_cols - 0.5,
            y0=1.5 - bar_half,
            y1=1.5 + bar_half,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor=separator_color,
            layer="above",
        )
    )

    fig_width = (
        margin["l"]
        + margin["r"]
        + grid_cols * cell_px
        + (grid_cols - 1) * gap_px
    )
    fig_height = (
        margin["t"]
        + margin["b"]
        + grid_rows * cell_px
        + (grid_rows - 1) * gap_px
    )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                zmin=0,
                zmax=1,
                colorscale=colorscale,
                showscale=False,
                hoverinfo="skip",
                xgap=gap_px,
                ygap=gap_px,
            )
        ]
    )
    fig.update_layout(
        margin=margin,
        autosize=False,
        width=fig_width,
        height=fig_height,
        annotations=annotations,
        shapes=shapes,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=grid_bg,
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[-0.5, grid_cols - 0.5],
        constrain="domain",
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[grid_rows - 0.5, -0.5],
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def build_character_table_figure(
    *,
    text: str,
    width: int,
    code_word: str,
    punch_cards: list[PunchCardConfig],
) -> go.Figure:
    width = max(1, min(200, int(width)))
    n_colors = max(1, len(str(code_word or "")))
    palette = qualitative_palette(n_colors)

    cell_px = 28
    gap_px = 1
    margin = dict(l=10, r=10, t=10, b=10)

    chars = list(text)
    if not chars:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["Character table"]),
                    cells=dict(values=[[""]]),
                )
            ]
        )
        fig.update_layout(margin=margin, autosize=False, height=120, width=320)
        return fig

    n_rows = (len(chars) + width - 1) // width
    padded = chars + [" "] * (n_rows * width - len(chars))

    mask_rows, mask_cols = punch_mask.shape

    # Caesar shifting alphabet for punch-card words.
    # We build it from the table text, code word, and all punch-card words.
    # We explicitly include a space so that padded holes are also shifted.
    _word_pool = "".join((c.word or "") for c in punch_cards)
    _alpha_source = f"{text}{code_word}{_word_pool} ".upper()
    caesar_alphabet = compute_character_alphabet(_alpha_source)
    caesar_index = {ch: i for i, ch in enumerate(caesar_alphabet)}

    def _caesar_shift_char(ch: str, *, shift: int) -> str:
        return _shift_over_alphabet(
            ch,
            alphabet=caesar_alphabet,
            index=caesar_index,
            key_indices=[int(shift)],
        )

    # Pre-calculate hole indices for both orientations.
    punch_holes = np.full(punch_mask.shape, -1, dtype=int)
    punch_holes[punch_mask == 1] = np.arange(np.sum(punch_mask == 1))

    flipped_mask = punch_mask[::-1, ::-1]
    flipped_holes = np.full(flipped_mask.shape, -1, dtype=int)
    flipped_holes[flipped_mask == 1] = np.arange(np.sum(flipped_mask == 1))

    def _card_info(card: PunchCardConfig):
        if not bool(card.flipped):
            return punch_mask, punch_holes
        # Flip orientation (rotate 180°) for display.
        return flipped_mask, flipped_holes

    def _card_pos(card: PunchCardConfig) -> tuple[int, int]:
        # UI exposes x/y positions; interpret as col/row offsets.
        return max(1, int(card.x)), max(1, int(card.y))

    n_cols = width

    # z values encode the repeating color index; 0 is "empty/white".
    z: list[list[int]] = []
    annotations: list[dict] = []
    shapes: list[dict] = []

    colors = ["rgba(255,255,255,1)"] + [
        _hex_to_rgba(c, alpha=0.5) for c in palette
    ]
    colorscale = _discrete_colorscale(colors)

    for row in range(n_rows):
        z_row: list[int] = []
        for col in range(n_cols):
            linear = row * n_cols + col
            v = padded[linear]

            # If a punch card reveals this cell, override the displayed character
            # with the punch-card word Caesar-shifted by card.shift.
            for card in punch_cards:
                if not bool(card.is_active):
                    continue
                px, py = _card_pos(card)
                punch_col = px - 1
                punch_row = py - 1
                m, mh = _card_info(card)
                mr = row - punch_row
                mc = col - punch_col
                if (
                    0 <= mr < mask_rows
                    and 0 <= mc < mask_cols
                    and int(m[mr, mc]) == 1
                ):
                    # Fill the card's holes starting at top-left,
                    # padding with spaces if the word is shorter.
                    hole_idx = int(mh[mr, mc])
                    w = str(card.word or "").upper()
                    wc = w[hole_idx] if hole_idx < len(w) else " "
                    v = _caesar_shift_char(wc, shift=int(card.shift))
                    break

            if v == "":
                z_row.append(0)
            else:
                z_row.append((linear % n_colors) + 1)
                annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=v,
                        showarrow=False,
                        font=dict(size=14, color="black"),
                    )
                )

            # Highlight punch-mask hits (union of active punch cards) by
            # whitening the background.
            for card in punch_cards:
                if not bool(card.is_active):
                    continue
                px, py = _card_pos(card)
                # Positions are 1-based to match the user's expectation.
                punch_col = px - 1
                punch_row = py - 1
                m, _ = _card_info(card)
                mr = row - punch_row
                mc = col - punch_col
                if (
                    0 <= mr < mask_rows
                    and 0 <= mc < mask_cols
                    and int(m[mr, mc]) == 1
                ):
                    shapes.append(
                        dict(
                            type="rect",
                            xref="x",
                            yref="y",
                            x0=col - 0.5,
                            x1=col + 0.5,
                            y0=row - 0.5,
                            y1=row + 0.5,
                            line=dict(color="rgba(0,0,0,1)", width=3),
                            fillcolor="rgba(255,255,255,0.3)",
                        )
                    )
                    break

        z.append(z_row)

    fig_width = (
        margin["l"] + margin["r"] + n_cols * cell_px + (n_cols - 1) * gap_px
    )
    fig_height = (
        margin["t"] + margin["b"] + n_rows * cell_px + (n_rows - 1) * gap_px
    )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                zmin=0,
                zmax=len(colors) - 1,
                colorscale=colorscale,
                showscale=False,
                hoverinfo="skip",
                xgap=gap_px,
                ygap=gap_px,
            )
        ]
    )
    fig.update_layout(
        margin=margin,
        autosize=False,
        width=fig_width,
        height=fig_height,
        annotations=annotations,
        shapes=shapes,
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[-0.5, n_cols - 0.5],
        constrain="domain",
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[n_rows - 0.5, -0.5],
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


class CharacterTableState(rx.State):
    # Persisted in browser localStorage (simple, no backend required).
    text: str = rx.LocalStorage("", name="character_table_text")
    # Full config blob (preferred).
    config_storage: str = rx.LocalStorage("", name="character_table_config")

    # Legacy per-field storage keys.
    table_width_storage: str = rx.LocalStorage("", name="character_table_width")

    # New multi-instance punch card config (JSON list).
    punch_cards_storage: str = rx.LocalStorage(
        "", name="character_table_punch_cards"
    )

    # Legacy single-instance punch-card keys (used for migration only).
    punch_row_storage: str = rx.LocalStorage(
        "", name="character_table_punch_row"
    )
    punch_col_storage: str = rx.LocalStorage(
        "", name="character_table_punch_col"
    )

    table_width: int = 20
    table_width_slider: list[float] = [20.0]

    code_word: str = ""
    show_encrypted: bool = False
    vigenere_backward: bool = False

    punch_cards: list[PunchCardConfig] = []

    figure: go.Figure = go.Figure()
    mapping_figure_light: go.Figure = go.Figure()
    mapping_figure_dark: go.Figure = go.Figure()

    @rx.var
    def alphabet_size(self) -> int:
        processed = preprocess_text(self.text)
        _word_pool = "".join((c.word or "") for c in self.punch_cards)
        _alpha_source = f"{processed}{self.code_word}{_word_pool} ".upper()
        alpha = compute_character_alphabet(_alpha_source)
        return len(alpha)

    def _rebuild(self) -> None:
        processed = preprocess_text(self.text)
        display = (
            vigenere_encrypt(
                processed,
                code_word=self.code_word,
                backward=bool(self.vigenere_backward),
            )
            if bool(self.show_encrypted)
            else processed
        )
        self.figure = build_character_table_figure(
            text=display,
            width=self.table_width,
            code_word=self.code_word,
            punch_cards=self.punch_cards,
        )
        self.mapping_figure_light = build_character_mapping_figure(
            text=processed,
            code_word=self.code_word,
            theme="light",
        )
        self.mapping_figure_dark = build_character_mapping_figure(
            text=processed,
            code_word=self.code_word,
            theme="dark",
        )

    @staticmethod
    def _cards_to_jsonable(
        cards: list[PunchCardConfig],
    ) -> list[dict[str, object]]:
        return [
            {
                "x": int(c.x),
                "y": int(c.y),
                "flipped": bool(c.flipped),
                "isActive": bool(c.is_active),
                "shift": int(c.shift),
                "word": str(c.word),
            }
            for c in cards
        ]

    def _sync_punch_cards_storage(self) -> None:
        self.punch_cards_storage = json.dumps(
            self._cards_to_jsonable(self.punch_cards)
        )

    def _sync_config_storage(self) -> None:
        self.config_storage = json.dumps(
            {
                "text": self.text,
                "table_width": int(self.table_width),
                "code_word": str(self.code_word),
                "show_encrypted": bool(self.show_encrypted),
                "vigenere_backward": bool(self.vigenere_backward),
                "punch_cards": self._cards_to_jsonable(self.punch_cards),
            }
        )

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        if value is None or isinstance(value, bool):
            return int(default)
        if isinstance(value, int):
            return value
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_int_ge_1(value: Any, *, default: int) -> int:
        if value is None or isinstance(value, bool):
            return max(1, int(default))
        if isinstance(value, int):
            return max(1, value)
        if isinstance(value, float):
            return max(1, int(value))
        if isinstance(value, str):
            try:
                return max(1, int(float(value)))
            except (TypeError, ValueError):
                return max(1, int(default))
        return max(1, int(default))

    @staticmethod
    def _coerce_cards(value: object) -> list[PunchCardConfig]:
        if not isinstance(value, list):
            raise ValueError("punch_cards must be a list")
        out: list[PunchCardConfig] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            x = CharacterTableState._coerce_int_ge_1(item.get("x"), default=1)
            y = CharacterTableState._coerce_int_ge_1(item.get("y"), default=1)
            is_active = bool(
                item.get(
                    "isActive",
                    item.get("is_active", True),
                )
            )
            shift = CharacterTableState._coerce_int(
                item.get("shift"), default=1
            )
            word = item.get("word")
            if not isinstance(word, str):
                word = ""
            flipped = bool(item.get("flipped", False))
            out.append(
                PunchCardConfig(
                    x=x,
                    y=y,
                    flipped=flipped,
                    is_active=is_active,
                    shift=shift,
                    word=word,
                )
            )
        return out

    def _apply_defaults(self, defaults: dict[str, Any]) -> None:
        text_val = defaults.get("text")
        if isinstance(text_val, str):
            self.text = text_val
        else:
            self.text = ""

        self.table_width = self._coerce_int_ge_1(
            defaults.get("table_width"), default=20
        )
        self.table_width_storage = str(self.table_width)
        self.table_width_slider = [float(self.table_width)]

        cw = defaults.get("code_word")
        if isinstance(cw, str) and cw.strip() != "":
            self.code_word = cw.upper()
        else:
            self.code_word = ""

        se = defaults.get("show_encrypted")
        self.show_encrypted = bool(se) if se is not None else False

        vb = defaults.get("vigenere_backward")
        self.vigenere_backward = bool(vb) if vb is not None else False

        try:
            self.punch_cards = self._coerce_cards(
                defaults.get(
                    "punch_cards",
                    [
                        {
                            "x": 1,
                            "y": 1,
                            "flipped": False,
                            "isActive": True,
                            "shift": 1,
                            "word": "",
                        }
                    ],
                )
            )
        except Exception:
            self.punch_cards = [PunchCardConfig()]

        # Clear legacy single-card keys to avoid re-migration.
        self.punch_row_storage = ""
        self.punch_col_storage = ""

        self._sync_punch_cards_storage()
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def reset_to_defaults(self):
        defaults: dict[str, Any] = load_character_table_defaults()
        self._apply_defaults(defaults)

    @rx.event
    def set_default(self):
        doc: dict[str, Any] = {
            "version": 2,
            "text": self.text,
            "table_width": int(self.table_width),
            "code_word": str(self.code_word),
            "show_encrypted": bool(self.show_encrypted),
            "vigenere_backward": bool(self.vigenere_backward),
            "punch_cards": self._cards_to_jsonable(self.punch_cards),
        }
        save_character_table_defaults(doc)

    @rx.event
    def on_load(self):
        defaults: dict[str, Any] = load_character_table_defaults()

        def _defaults_int(key: str, fallback: int) -> int:
            return self._coerce_int_ge_1(defaults.get(key), default=fallback)

        # If a full config blob exists in localStorage, prefer it.
        if self.config_storage not in {None, ""}:
            try:
                cfg_any = json.loads(self.config_storage)
                if isinstance(cfg_any, dict):
                    cfg: dict[str, Any] = cfg_any
                    text_val = cfg.get("text")
                    if isinstance(text_val, str):
                        self.text = text_val
                    self.table_width = self._coerce_int_ge_1(
                        cfg.get("table_width"),
                        default=_defaults_int("table_width", 20),
                    )
                    self.table_width_storage = str(self.table_width)
                    self.table_width_slider = [float(self.table_width)]

                    cw = cfg.get("code_word")
                    if isinstance(cw, str) and cw.strip() != "":
                        self.code_word = cw.upper()
                    else:
                        d = defaults.get("code_word")
                        self.code_word = (
                            d.upper()
                            if isinstance(d, str) and d.strip() != ""
                            else ""
                        )

                    se = cfg.get("show_encrypted")
                    self.show_encrypted = bool(se) if se is not None else False

                    vb = cfg.get("vigenere_backward")
                    self.vigenere_backward = (
                        bool(vb) if vb is not None else False
                    )

                    self.punch_cards = self._coerce_cards(
                        cfg.get("punch_cards", defaults.get("punch_cards", []))
                    )
                    self._sync_punch_cards_storage()
                    self._sync_config_storage()
                    self._rebuild()
                    return
            except Exception:
                # If parsing fails, fall back to legacy keys/defaults below.
                pass

        # Ensure the slider reflects the persisted width and the figure is built
        # when the page is opened/refreshed.
        if self.table_width_storage in {None, ""}:
            width = _defaults_int("table_width", 20)
        else:
            width = self._coerce_int_ge_1(
                self.table_width_storage,
                default=_defaults_int("table_width", 20),
            )
        self.table_width = width
        self.table_width_storage = str(width)
        self.table_width_slider = [float(width)]

        d = defaults.get("code_word")
        self.code_word = (
            d.upper() if isinstance(d, str) and d.strip() != "" else ""
        )

        se = defaults.get("show_encrypted")
        self.show_encrypted = bool(se) if se is not None else False

        vb = defaults.get("vigenere_backward")
        self.vigenere_backward = bool(vb) if vb is not None else False

        # Punch cards: prefer new JSON storage; if empty, try migrating legacy row/col;
        # otherwise fall back to backend YAML defaults.
        cards_raw: object | None = None
        if self.punch_cards_storage not in {None, ""}:
            try:
                cards_raw = json.loads(self.punch_cards_storage)
            except Exception:
                cards_raw = None
        elif self.punch_row_storage not in {
            None,
            "",
        } or self.punch_col_storage not in {None, ""}:
            pr = self._coerce_int_ge_1(self.punch_row_storage, default=1)
            pc = self._coerce_int_ge_1(self.punch_col_storage, default=1)
            cards_raw = [
                {"x": pc, "y": pr, "flipped": False, "shift": 1, "word": ""}
            ]
        else:
            cards_raw = defaults.get(
                "punch_cards",
                [{"x": 1, "y": 1, "flipped": False, "shift": 1, "word": ""}],
            )

        try:
            self.punch_cards = self._coerce_cards(cards_raw)
        except Exception:
            self.punch_cards = [PunchCardConfig()]

        self._sync_punch_cards_storage()
        self._sync_config_storage()

        self._rebuild()

    @rx.event
    def set_text(self, value: str):
        self.text = value
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def set_code_word(self, value: str):
        self.code_word = str(value).upper()
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def set_show_encrypted(self, value: bool):
        self.show_encrypted = bool(value)
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def set_vigenere_backward(self, value: bool):
        self.vigenere_backward = bool(value)
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def set_table_width(self, value: list[float]):
        raw = value[0] if value else 1
        try:
            width = int(float(raw))
        except (TypeError, ValueError):
            width = 1

        width = max(1, min(80, width))
        self.table_width = width
        self.table_width_storage = str(width)
        self.table_width_slider = [float(width)]
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def set_table_width_input(self, value: str):
        width = self._coerce_int_ge_1(value, default=1)
        width = max(1, min(80, width))
        self.table_width = width
        self.table_width_storage = str(width)
        self.table_width_slider = [float(width)]
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def dec_table_width(self):
        self.table_width = max(1, int(self.table_width) - 1)
        self.table_width_storage = str(self.table_width)
        self.table_width_slider = [float(self.table_width)]
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def inc_table_width(self):
        self.table_width = min(80, int(self.table_width) + 1)
        self.table_width_storage = str(self.table_width)
        self.table_width_slider = [float(self.table_width)]
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def add_punch_card(self):
        cards = list(self.punch_cards)
        cards.append(PunchCardConfig())
        self.punch_cards = cards
        self._sync_punch_cards_storage()
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def remove_punch_card(self, index: int):
        cards = list(self.punch_cards)
        if 0 <= int(index) < len(cards):
            cards.pop(int(index))
        self.punch_cards = cards
        self._sync_punch_cards_storage()
        self._sync_config_storage()
        self._rebuild()

    def _update_card(self, index: int, *, updates: dict[str, object]) -> None:
        cards = list(self.punch_cards)
        i = int(index)
        if not (0 <= i < len(cards)):
            return
        current = cards[i]
        merged = {
            "x": current.x,
            "y": current.y,
            "flipped": current.flipped,
            "shift": current.shift,
            "word": current.word,
        }
        merged.update(updates)
        cards[i] = self._coerce_cards([merged])[0]
        self.punch_cards = cards
        self._sync_punch_cards_storage()
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def set_punch_card_x(self, index: int, value: str):
        self._update_card(
            index, updates={"x": self._coerce_int_ge_1(value, default=1)}
        )

    @rx.event
    def set_punch_card_x_slider(self, index: int, value: list[float]):
        raw = value[0] if value else 1
        try:
            x = int(float(raw))
        except (TypeError, ValueError):
            x = 1
        x = max(1, min(200, x))
        self._update_card(index, updates={"x": x})

    @rx.event
    def dec_punch_card_x(self, index: int):
        cards = list(self.punch_cards)
        i = int(index)
        if not (0 <= i < len(cards)):
            return
        x = max(1, int(cards[i].x) - 1)
        self._update_card(index, updates={"x": x})

    @rx.event
    def inc_punch_card_x(self, index: int):
        cards = list(self.punch_cards)
        i = int(index)
        if not (0 <= i < len(cards)):
            return
        x = min(200, int(cards[i].x) + 1)
        self._update_card(index, updates={"x": x})

    @rx.event
    def set_punch_card_y(self, index: int, value: str):
        self._update_card(
            index, updates={"y": self._coerce_int_ge_1(value, default=1)}
        )

    @rx.event
    def set_punch_card_y_slider(self, index: int, value: list[float]):
        raw = value[0] if value else 1
        try:
            y = int(float(raw))
        except (TypeError, ValueError):
            y = 1
        y = max(1, min(200, y))
        self._update_card(index, updates={"y": y})

    @rx.event
    def dec_punch_card_y(self, index: int):
        cards = list(self.punch_cards)
        i = int(index)
        if not (0 <= i < len(cards)):
            return
        y = max(1, int(cards[i].y) - 1)
        self._update_card(index, updates={"y": y})

    @rx.event
    def inc_punch_card_y(self, index: int):
        cards = list(self.punch_cards)
        i = int(index)
        if not (0 <= i < len(cards)):
            return
        y = min(200, int(cards[i].y) + 1)
        self._update_card(index, updates={"y": y})

    @rx.event
    def set_punch_card_shift(self, index: int, value: str):
        n = int(self.alphabet_size)
        shift = self._coerce_int(value, default=1)
        shift = max(-n, min(n, shift))
        self._update_card(index, updates={"shift": shift})

    @rx.event
    def set_punch_card_shift_slider(self, index: int, value: list[float]):
        n = int(self.alphabet_size)
        raw = value[0] if value else 1
        try:
            shift = int(float(raw))
        except (TypeError, ValueError):
            shift = 1
        shift = max(-n, min(n, shift))
        self._update_card(index, updates={"shift": shift})

    @rx.event
    def dec_punch_card_shift(self, index: int):
        cards = list(self.punch_cards)
        i = int(index)
        if not (0 <= i < len(cards)):
            return
        n = int(self.alphabet_size)
        shift = max(-n, int(cards[i].shift) - 1)
        self._update_card(index, updates={"shift": shift})

    @rx.event
    def inc_punch_card_shift(self, index: int):
        cards = list(self.punch_cards)
        i = int(index)
        if not (0 <= i < len(cards)):
            return
        n = int(self.alphabet_size)
        shift = min(n, int(cards[i].shift) + 1)
        self._update_card(index, updates={"shift": shift})

    @rx.event
    def set_punch_card_word(self, index: int, value: str):
        self._update_card(index, updates={"word": value})

    @rx.event
    def set_punch_card_flipped(self, index: int, value: bool):
        self._update_card(index, updates={"flipped": bool(value)})

    @rx.event
    def set_punch_card_active(self, index: int, value: bool):
        self._update_card(index, updates={"isActive": bool(value)})
