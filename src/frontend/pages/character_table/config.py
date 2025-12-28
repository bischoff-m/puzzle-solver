import reflex as rx

from .state import CharacterTableState


def _punch_card_row(card, index) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.heading(rx.text("Punch card "), rx.text(index + 1), size="4"),
            rx.spacer(),
            rx.button(
                "Remove",
                on_click=CharacterTableState.remove_punch_card(index),
                size="2",
                variant="outline",
            ),
            width="100%",
            align="center",
        ),
        rx.hstack(
            rx.text("X"),
            rx.input(
                type="number",
                min=1,
                step=1,
                value=card.x,
                on_change=CharacterTableState.set_punch_card_x(index),
                width="8em",
            ),
            rx.text("Y"),
            rx.input(
                type="number",
                min=1,
                step=1,
                value=card.y,
                on_change=CharacterTableState.set_punch_card_y(index),
                width="8em",
            ),
            rx.text("Shift"),
            rx.input(
                type="number",
                min=1,
                step=1,
                value=card.shift,
                on_change=CharacterTableState.set_punch_card_shift(index),
                width="8em",
            ),
            spacing="3",
            width="100%",
            align="center",
            wrap="wrap",
        ),
        rx.hstack(
            rx.text("Word"),
            rx.input(
                value=card.word,
                on_change=CharacterTableState.set_punch_card_word(index),
                placeholder="",
                width="20em",
            ),
            rx.checkbox(
                "Flipped",
                is_checked=card.flipped,
                on_change=CharacterTableState.set_punch_card_flipped(index),
            ),
            spacing="3",
            width="100%",
            align="center",
            wrap="wrap",
        ),
        rx.divider(),
        width="100%",
        spacing="2",
    )


def config_section() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.heading("Configuration", size="5"),
            rx.spacer(),
            rx.button(
                "Reset to defaults",
                on_click=CharacterTableState.reset_to_defaults,
                variant="outline",
            ),
            rx.button(
                "Export to YAML",
                on_click=CharacterTableState.export_to_yaml,
            ),
            width="100%",
            align="center",
        ),
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
        rx.divider(),
        rx.hstack(
            rx.heading("Punch cards", size="5"),
            rx.spacer(),
            rx.button(
                "Add punch card", on_click=CharacterTableState.add_punch_card
            ),
            width="100%",
            align="center",
        ),
        rx.foreach(CharacterTableState.punch_cards, _punch_card_row),
        width="100%",
        spacing="2",
    )
