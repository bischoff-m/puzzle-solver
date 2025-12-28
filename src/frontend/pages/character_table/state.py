import re

import plotly.graph_objects as go
import reflex as rx

from puzzle_solver.plotting import _qualitative_palette

from .punch_card import mask as punch_mask


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
    punch_row: int,
    punch_col: int,
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
    punch_row = max(0, int(punch_row))
    punch_col = max(0, int(punch_col))

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

            # Thick border for punch-mask hits.
            mr = row - punch_row
            mc = col - punch_col
            if (
                0 <= mr < mask_rows
                and 0 <= mc < mask_cols
                and int(punch_mask[mr, mc]) == 1
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
    table_width_storage: str = rx.LocalStorage(
        "20", name="character_table_width"
    )

    code_word_length_storage: str = rx.LocalStorage(
        "5", name="character_table_code_word_length"
    )

    punch_row_storage: str = rx.LocalStorage(
        "0", name="character_table_punch_row"
    )
    punch_col_storage: str = rx.LocalStorage(
        "0", name="character_table_punch_col"
    )

    table_width: int = 20
    table_width_slider: list[float] = [20.0]

    code_word_length: int = 5
    code_word_length_slider: list[float] = [5.0]

    punch_row: int = 0
    punch_row_slider: list[float] = [0.0]
    punch_col: int = 0
    punch_col_slider: list[float] = [0.0]

    figure: go.Figure = go.Figure()

    def _rebuild(self) -> None:
        processed = preprocess_text(self.text)
        self.figure = build_character_table_figure(
            text=processed,
            width=self.table_width,
            code_word_length=self.code_word_length,
            punch_row=self.punch_row,
            punch_col=self.punch_col,
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

        try:
            pr = int(float(self.punch_row_storage))
        except (TypeError, ValueError):
            pr = 0
        pr = max(0, pr)
        self.punch_row = pr
        self.punch_row_storage = str(pr)
        self.punch_row_slider = [float(pr)]

        try:
            pc = int(float(self.punch_col_storage))
        except (TypeError, ValueError):
            pc = 0
        pc = max(0, pc)
        self.punch_col = pc
        self.punch_col_storage = str(pc)
        self.punch_col_slider = [float(pc)]

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

    @rx.event
    def set_punch_row(self, value: list[float]):
        raw = value[0] if value else 0
        try:
            pr = int(float(raw))
        except (TypeError, ValueError):
            pr = 0
        pr = max(0, pr)

        self.punch_row = pr
        self.punch_row_storage = str(pr)
        self.punch_row_slider = [float(pr)]
        self._rebuild()

    @rx.event
    def set_punch_col(self, value: list[float]):
        raw = value[0] if value else 0
        try:
            pc = int(float(raw))
        except (TypeError, ValueError):
            pc = 0
        pc = max(0, pc)

        self.punch_col = pc
        self.punch_col_storage = str(pc)
        self.punch_col_slider = [float(pc)]
        self._rebuild()
