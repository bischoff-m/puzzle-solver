import reflex as rx

from frontend.components.int_control import int_control

from .state import CharacterTableState

_LABEL_W = "10em"


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
        int_control(
            label="X",
            value=card.x,
            min_=1,
            max_=200,
            label_width=_LABEL_W,
            on_dec=CharacterTableState.dec_punch_card_x(index),
            on_inc=CharacterTableState.inc_punch_card_x(index),
            on_change_slider=CharacterTableState.set_punch_card_x_slider(index),
            on_change_text=CharacterTableState.set_punch_card_x(index),
        ),
        int_control(
            label="Y",
            value=card.y,
            min_=1,
            max_=200,
            label_width=_LABEL_W,
            on_dec=CharacterTableState.dec_punch_card_y(index),
            on_inc=CharacterTableState.inc_punch_card_y(index),
            on_change_slider=CharacterTableState.set_punch_card_y_slider(index),
            on_change_text=CharacterTableState.set_punch_card_y(index),
        ),
        int_control(
            label="Shift",
            value=card.shift,
            min_=1,
            max_=200,
            label_width=_LABEL_W,
            on_dec=CharacterTableState.dec_punch_card_shift(index),
            on_inc=CharacterTableState.inc_punch_card_shift(index),
            on_change_slider=CharacterTableState.set_punch_card_shift_slider(
                index
            ),
            on_change_text=CharacterTableState.set_punch_card_shift(index),
        ),
        rx.hstack(
            rx.text("Word", width=_LABEL_W, text_align="right"),
            rx.input(
                value=card.word,
                on_change=CharacterTableState.set_punch_card_word(index),
                placeholder="",
            ),
            rx.checkbox(
                "Active",
                checked=card.is_active,
                on_change=CharacterTableState.set_punch_card_active(index),
            ),
            rx.checkbox(
                "Flipped",
                checked=card.flipped,
                on_change=CharacterTableState.set_punch_card_flipped(index),
            ),
            spacing="3",
            width="100%",
            align="center",
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
        int_control(
            label="Table width",
            value=CharacterTableState.table_width,
            min_=1,
            max_=80,
            slider_value=CharacterTableState.table_width_slider,
            label_width=_LABEL_W,
            on_dec=CharacterTableState.dec_table_width,
            on_inc=CharacterTableState.inc_table_width,
            on_change_slider=CharacterTableState.set_table_width,
            on_change_text=CharacterTableState.set_table_width_input,
        ),
        rx.hstack(
            rx.text("Code word", width=_LABEL_W, text_align="right"),
            rx.input(
                value=CharacterTableState.code_word,
                on_change=CharacterTableState.set_code_word,
                placeholder="",
            ),
            rx.checkbox(
                "Encrypted",
                checked=CharacterTableState.show_encrypted,
                on_change=CharacterTableState.set_show_encrypted,
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
