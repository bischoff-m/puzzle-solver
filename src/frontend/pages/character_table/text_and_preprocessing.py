from __future__ import annotations

import reflex as rx

from .state import CharacterTableState


def text_and_preprocessing_section() -> rx.Component:
    return rx.vstack(
        rx.heading("Text", size="5"),
        rx.text_area(
            value=CharacterTableState.text,
            on_change=CharacterTableState.set_text,
            placeholder="Enter text (line breaks will be removed)",
            width="100%",
            height="12em",
        ),
        rx.text("Line breaks are removed before table generation."),
        width="100%",
        spacing="2",
    )
