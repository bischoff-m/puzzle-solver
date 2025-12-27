from __future__ import annotations

import reflex as rx

from .config import config_section
from .table_plot import table_plot
from .text_and_preprocessing import text_and_preprocessing_section


def page() -> rx.Component:
    return rx.center(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Character Table", size="7"),
            rx.hstack(
                rx.vstack(
                    text_and_preprocessing_section(),
                    rx.divider(),
                    config_section(),
                    spacing="4",
                    align="start",
                ),
                rx.vstack(
                    table_plot(),
                    spacing="2",
                    align="start",
                ),
                spacing="6",
                align="start",
            ),
            spacing="4",
            align="center",
        ),
        padding="2em",
    )
