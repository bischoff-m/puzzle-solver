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
        rx.hstack(
            rx.text("code word length"),
            rx.text(CharacterTableState.code_word_length),
            rx.slider(
                value=CharacterTableState.code_word_length_slider,
                min=1,
                max=40,
                step=1,
                on_change=CharacterTableState.set_code_word_length,
                width="20em",
            ),
            spacing="3",
            width="100%",
            align="center",
        ),
        rx.hstack(
            rx.text("punch row"),
            rx.text(CharacterTableState.punch_row),
            rx.slider(
                value=CharacterTableState.punch_row_slider,
                min=0,
                max=200,
                step=1,
                on_change=CharacterTableState.set_punch_row,
                width="20em",
            ),
            spacing="3",
            width="100%",
            align="center",
        ),
        rx.hstack(
            rx.text("punch col"),
            rx.text(CharacterTableState.punch_col),
            rx.slider(
                value=CharacterTableState.punch_col_slider,
                min=0,
                max=200,
                step=1,
                on_change=CharacterTableState.set_punch_col,
                width="20em",
            ),
            spacing="3",
            width="100%",
            align="center",
        ),
        width="100%",
        spacing="2",
    )
