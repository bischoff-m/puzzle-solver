import reflex as rx

from .config import config_section
from .table_plot import mapping_plot, table_plot
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
                    rx.box(
                        table_plot(),
                        width="100%",
                        overflow_x="auto",
                    ),
                    spacing="2",
                    align="start",
                ),
                spacing="6",
                align="start",
            ),
            rx.divider(),
            rx.box(
                mapping_plot(),
                width="100%",
                overflow_x="auto",
            ),
            spacing="4",
            align="center",
        ),
        padding="2em",
    )
