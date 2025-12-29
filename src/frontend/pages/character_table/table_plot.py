import reflex as rx

from .state import CharacterTableState


def table_plot() -> rx.Component:
    return rx.plotly(data=CharacterTableState.figure, use_resize_handler=False)


def mapping_plot() -> rx.Component:
    return rx.color_mode_cond(
        rx.plotly(
            data=CharacterTableState.mapping_figure_light,
            use_resize_handler=False,
        ),
        rx.plotly(
            data=CharacterTableState.mapping_figure_dark,
            use_resize_handler=False,
        ),
    )
