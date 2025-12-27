from __future__ import annotations

import reflex as rx

from .state import CharacterTableState


def table_plot() -> rx.Component:
    return rx.plotly(data=CharacterTableState.figure, use_resize_handler=True)
