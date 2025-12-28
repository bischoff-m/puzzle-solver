import reflex as rx


def page() -> rx.Component:
    return rx.center(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Puzzle Solver", size="7"),
            rx.text("Choose a page:"),
            rx.hstack(
                rx.link(rx.button("Solver"), href="/solver"),
                rx.link(rx.button("Character Table"), href="/character-table"),
                spacing="3",
            ),
            spacing="4",
            align="center",
        ),
        padding="2em",
    )
