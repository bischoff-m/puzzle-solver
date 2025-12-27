from __future__ import annotations

import reflex as rx

from .config import config_section
from .table_plot import table_plot
from .text_and_preprocessing import text_and_preprocessing_section


def page() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Character Table", size="7"),
            text_and_preprocessing_section(),
            rx.divider(),
            config_section(),
            rx.divider(),
            table_plot(),
            spacing="4",
            width="100%",
        ),
        width="100%",
        max_width="1100px",
        padding_y="2em",
    )
