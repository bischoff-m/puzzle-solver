from __future__ import annotations

import reflex as rx

from .state import SolverState


def page() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Puzzle Solver", size="7"),
            rx.text("Solve and visualize the flat and cube solutions."),
            rx.hstack(
                rx.input(
                    value=SolverState.config_path,
                    on_change=SolverState.set_config_path,
                    placeholder="puzzle.yaml",
                    width="20em",
                ),
                rx.button(
                    "Solve flat",
                    on_click=SolverState.solve_flat,
                    loading=SolverState.solving_flat,
                ),
                rx.button(
                    "Solve cube",
                    on_click=SolverState.solve_cube,
                    loading=SolverState.solving_cube,
                ),
                spacing="3",
                width="100%",
            ),
            rx.cond(SolverState.error != "", rx.text(SolverState.error)),
            rx.divider(),
            rx.heading("Flat", size="5"),
            rx.plotly(data=SolverState.flat_figure, use_resize_handler=True),
            rx.divider(),
            rx.heading("Cube", size="5"),
            rx.plotly(data=SolverState.cube_figure, use_resize_handler=True),
            spacing="4",
            width="100%",
        ),
        width="100%",
        max_width="1100px",
        padding_y="2em",
    )
