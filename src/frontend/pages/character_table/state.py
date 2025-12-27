from __future__ import annotations

import plotly.graph_objects as go
import reflex as rx


def preprocess_text(text: str) -> str:
    return text.replace("\r", "").replace("\n", "")


def build_character_table_figure(*, text: str, width: int) -> go.Figure:
    width = max(1, min(200, int(width)))

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
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig

    n_rows = (len(chars) + width - 1) // width
    padded = chars + [""] * (n_rows * width - len(chars))

    columns: list[list[str]] = []
    for col in range(width):
        columns.append([padded[row * width + col] for row in range(n_rows)])

    header_values = [str(i + 1) for i in range(width)]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header_values, align="center"),
                cells=dict(values=columns, align="center"),
            )
        ]
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


class CharacterTableState(rx.State):
    text: str = ""
    table_width: int = 20
    table_width_slider: list[float] = [20.0]
    figure: go.Figure = go.Figure()

    def _rebuild(self) -> None:
        processed = preprocess_text(self.text)
        self.figure = build_character_table_figure(
            text=processed,
            width=self.table_width,
        )

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
        self.table_width_slider = [float(width)]
        self._rebuild()
