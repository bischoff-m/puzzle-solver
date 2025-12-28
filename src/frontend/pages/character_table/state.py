import json
import re
from typing import Any

import plotly.graph_objects as go
import reflex as rx
import yaml
from pydantic import BaseModel

from puzzle_solver.api import load_character_table_defaults
from puzzle_solver.plotting import _qualitative_palette

from .punch_card import mask as punch_mask


class PunchCardConfig(BaseModel):
    x: int = 1
    y: int = 1
    flipped: bool = False
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
    # Replace all whitespace with a single space
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Split by words and join back with single spaces
    words = re.findall(r"\w+", text)
    return " ".join(words).upper()


def build_character_table_figure(
    *,
    text: str,
    width: int,
    code_word_length: int,
    punch_cards: list[PunchCardConfig],
) -> go.Figure:
    width = max(1, min(200, int(width)))
    n_colors = max(1, int(code_word_length))
    palette = _qualitative_palette(n_colors)

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
    padded = chars + [""] * (n_rows * width - len(chars))

    mask_rows, mask_cols = punch_mask.shape

    def _card_mask(card: PunchCardConfig):
        if not bool(card.flipped):
            return punch_mask
        # Flip orientation (rotate 180°) for display.
        return punch_mask[::-1, ::-1]

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

            # Thick border for punch-mask hits (union of all punch cards).
            for card in punch_cards:
                px, py = _card_pos(card)
                # Positions are 1-based to match the user's expectation.
                punch_col = px - 1
                punch_row = py - 1
                m = _card_mask(card)
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
                            fillcolor="rgba(0,0,0,0)",
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

    code_word_length_storage: str = rx.LocalStorage(
        "", name="character_table_code_word_length"
    )

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

    code_word_length: int = 5
    code_word_length_slider: list[float] = [5.0]

    punch_cards: list[PunchCardConfig] = []

    figure: go.Figure = go.Figure()

    def _rebuild(self) -> None:
        processed = preprocess_text(self.text)
        self.figure = build_character_table_figure(
            text=processed,
            width=self.table_width,
            code_word_length=self.code_word_length,
            punch_cards=self.punch_cards,
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
                "code_word_length": int(self.code_word_length),
                "punch_cards": self._cards_to_jsonable(self.punch_cards),
            }
        )

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
            shift = CharacterTableState._coerce_int_ge_1(
                item.get("shift"), default=1
            )
            word = item.get("word")
            if not isinstance(word, str):
                word = ""
            flipped = bool(item.get("flipped", False))
            out.append(
                PunchCardConfig(
                    x=x, y=y, flipped=flipped, shift=shift, word=word
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

        self.code_word_length = self._coerce_int_ge_1(
            defaults.get("code_word_length"), default=5
        )
        self.code_word_length_storage = str(self.code_word_length)
        self.code_word_length_slider = [float(self.code_word_length)]

        try:
            self.punch_cards = self._coerce_cards(
                defaults.get(
                    "punch_cards",
                    [
                        {
                            "x": 1,
                            "y": 1,
                            "flipped": False,
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
    def export_to_yaml(self):
        doc: dict[str, Any] = {
            "version": 1,
            "text": self.text,
            "table_width": int(self.table_width),
            "code_word_length": int(self.code_word_length),
            "punch_cards": self._cards_to_jsonable(self.punch_cards),
        }
        data = yaml.safe_dump(doc, sort_keys=False)
        return rx.download(
            data=data,
            filename="character_table_config.yaml",
            mime_type="text/yaml",
        )

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

                    self.code_word_length = self._coerce_int_ge_1(
                        cfg.get("code_word_length"),
                        default=_defaults_int("code_word_length", 5),
                    )
                    self.code_word_length_storage = str(self.code_word_length)
                    self.code_word_length_slider = [
                        float(self.code_word_length)
                    ]

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

        if self.code_word_length_storage in {None, ""}:
            n = _defaults_int("code_word_length", 5)
        else:
            n = self._coerce_int_ge_1(
                self.code_word_length_storage,
                default=_defaults_int("code_word_length", 5),
            )
        self.code_word_length = n
        self.code_word_length_storage = str(n)
        self.code_word_length_slider = [float(n)]

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
    def set_table_width(self, value: list[float]):
        raw = value[0] if value else 1
        try:
            width = int(float(raw))
        except (TypeError, ValueError):
            width = 1

        self.table_width = width
        self.table_width_storage = str(width)
        self.table_width_slider = [float(width)]
        self._sync_config_storage()
        self._rebuild()

    @rx.event
    def set_code_word_length(self, value: list[float]):
        raw = value[0] if value else 1
        try:
            n = int(float(raw))
        except (TypeError, ValueError):
            n = 1
        n = max(1, n)

        self.code_word_length = n
        self.code_word_length_storage = str(n)
        self.code_word_length_slider = [float(n)]
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
    def set_punch_card_y(self, index: int, value: str):
        self._update_card(
            index, updates={"y": self._coerce_int_ge_1(value, default=1)}
        )

    @rx.event
    def set_punch_card_shift(self, index: int, value: str):
        self._update_card(
            index, updates={"shift": self._coerce_int_ge_1(value, default=1)}
        )

    @rx.event
    def set_punch_card_word(self, index: int, value: str):
        self._update_card(index, updates={"word": value})

    @rx.event
    def set_punch_card_flipped(self, index: int, value: bool):
        self._update_card(index, updates={"flipped": bool(value)})
