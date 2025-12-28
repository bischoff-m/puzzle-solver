import re

import plotly.graph_objects as go
import reflex as rx

from puzzle_solver.plotting import _qualitative_palette


def preprocess_text(text: str) -> str:
    # Replace all whitespace with a single space
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Split by words and join back with single spaces
    words = re.findall(r"\w+", text)
    return " ".join(words).upper()


def build_character_table_figure(
    *, text: str, width: int, code_word_length: int
) -> go.Figure:
    width = max(1, min(200, int(width)))
    n_colors = max(1, int(code_word_length))
    palette = _qualitative_palette(n_colors)

    cell_px = 28
    header_px = 28
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

    columns: list[list[str]] = []
    fill_colors: list[list[str]] = []
    for col in range(width):
        col_vals: list[str] = []
        col_colors: list[str] = []
        for row in range(n_rows):
            linear = row * width + col
            v = padded[linear]
            col_vals.append(v)
            if v == "":
                col_colors.append("#ffffff")
            else:
                col_colors.append(palette[linear % n_colors])
        columns.append(col_vals)
        fill_colors.append(col_colors)

    header_values = [str(i + 1) for i in range(width)]
    n_cols = width
    n_rows = len(columns[0]) if columns else 0

    fig_width = margin["l"] + margin["r"] + n_cols * cell_px
    fig_height = margin["t"] + margin["b"] + header_px + n_rows * cell_px
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[cell_px] * n_cols,
                header=dict(
                    values=header_values, align="center", height=header_px
                ),
                cells=dict(
                    values=columns,
                    align="center",
                    height=cell_px,
                    fill_color=fill_colors,
                ),
            )
        ]
    )
    fig.update_layout(
        margin=margin,
        autosize=False,
        width=fig_width,
        height=fig_height,
    )
    return fig


class CharacterTableState(rx.State):
    # Persisted in browser localStorage (simple, no backend required).
    text: str = rx.LocalStorage("", name="character_table_text")
    table_width_storage: str = rx.LocalStorage(
        "20", name="character_table_width"
    )

    code_word_length_storage: str = rx.LocalStorage(
        "5", name="character_table_code_word_length"
    )

    table_width: int = 20
    table_width_slider: list[float] = [20.0]

    code_word_length: int = 5
    code_word_length_slider: list[float] = [5.0]

    figure: go.Figure = go.Figure()

    def _rebuild(self) -> None:
        processed = preprocess_text(self.text)
        self.figure = build_character_table_figure(
            text=processed,
            width=self.table_width,
            code_word_length=self.code_word_length,
        )

    @rx.event
    def on_load(self):
        # Ensure the slider reflects the persisted width and the figure is built
        # when the page is opened/refreshed.
        try:
            width = int(float(self.table_width_storage))
        except (TypeError, ValueError):
            width = 20

        width = max(1, width)
        self.table_width = width
        self.table_width_storage = str(width)
        self.table_width_slider = [float(width)]

        try:
            n = int(float(self.code_word_length_storage))
        except (TypeError, ValueError):
            n = 5
        n = max(1, n)
        self.code_word_length = n
        self.code_word_length_storage = str(n)
        self.code_word_length_slider = [float(n)]

        self._rebuild()

    @rx.event
    def set_text(self, value: str):
        self.text = value
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
        self._rebuild()
