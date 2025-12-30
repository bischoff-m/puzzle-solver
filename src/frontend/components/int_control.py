import reflex as rx


def int_control(
    *,
    label: str,
    value: rx.Var | int,
    min_: rx.Var | int,
    max_: rx.Var | int,
    slider_value: rx.Var | list | None = None,
    on_change_slider=None,
    on_change_text=None,
    on_dec=None,
    on_inc=None,
    label_width: str = "auto",
    slider_width: str = "auto",
    input_width: str = "auto",
) -> rx.Component:
    """Integer control with +/- buttons, slider, and number input.

    Handlers are intentionally untyped here to allow passing Reflex EventSpecs.
    """

    if slider_value is None:
        slider_value = [value]

    return rx.hstack(
        rx.text(label, width=label_width, text_align="right"),
        rx.button("-", on_click=on_dec, size="2", variant="outline"),
        rx.slider(
            value=slider_value,
            min=min_,
            max=max_,
            step=1,
            on_change=on_change_slider,
            width=slider_width,
        ),
        rx.button("+", on_click=on_inc, size="2", variant="outline"),
        rx.input(
            type="number",
            min=min_,
            max=max_,
            step=1,
            value=value,
            on_change=on_change_text,
            width=input_width,
        ),
        spacing="3",
        align="center",
        width="100%",
        wrap="wrap",
    )
