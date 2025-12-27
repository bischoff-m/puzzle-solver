from __future__ import annotations

import reflex as rx

from .state import CharacterTableState


def config_section() -> rx.Component:
    return rx.vstack(
        rx.heading("Configuration", size="5"),
        rx.hstack(
            rx.text("Table width"),
            rx.text(CharacterTableState.table_width),
            rx.slider(
                value=CharacterTableState.table_width_slider,
                min=1,
                max=80,
                step=1,
                on_change=CharacterTableState.set_table_width,
                width="20em",
            ),
            spacing="3",
            width="100%",
            align="center",
        ),
        width="100%",
        spacing="2",
    )
