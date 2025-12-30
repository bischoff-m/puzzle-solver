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
        width="100%",
        spacing="2",
    )
