from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import reflex as rx

from puzzle_solver.api import solve_and_plot_cube, solve_and_plot_flat


class State(rx.State):
    config_path: str = "puzzle.yaml"

    flat_figure: go.Figure = go.Figure()
    cube_figure: go.Figure = go.Figure()

    solving_flat: bool = False
    solving_cube: bool = False
    error: str = ""

    @rx.event
    def set_config_path(self, value: str):
        self.config_path = value

    @rx.event
    def solve_flat(self):
        self.error = ""
        self.solving_flat = True
        yield

        try:
            fig, _ = solve_and_plot_flat(path=Path(self.config_path))
            self.flat_figure = fig
        except Exception as e:
            self.error = str(e)
        finally:
            self.solving_flat = False

    @rx.event
    def solve_cube(self):
        self.error = ""
        self.solving_cube = True
        yield

        try:
            fig, _ = solve_and_plot_cube(path=Path(self.config_path))
            self.cube_figure = fig
        except Exception as e:
            self.error = str(e)
        finally:
            self.solving_cube = False


def index() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Puzzle Solver", size="7"),
            rx.text("Solve and visualize the flat and cube solutions."),
            rx.hstack(
                rx.input(
                    value=State.config_path,
                    on_change=State.set_config_path,
                    placeholder="puzzle.yaml",
                    width="20em",
                ),
                rx.button(
                    "Solve flat",
                    on_click=State.solve_flat,
                    loading=State.solving_flat,
                ),
                rx.button(
                    "Solve cube",
                    on_click=State.solve_cube,
                    loading=State.solving_cube,
                ),
                spacing="3",
                width="100%",
            ),
            rx.cond(State.error != "", rx.text(State.error)),
            rx.divider(),
            rx.heading("Flat", size="5"),
            rx.plotly(data=State.flat_figure, use_resize_handler=True),
            rx.divider(),
            rx.heading("Cube", size="5"),
            rx.plotly(data=State.cube_figure, use_resize_handler=True),
            spacing="4",
            width="100%",
        ),
        width="100%",
        max_width="1100px",
        padding_y="2em",
    )


app = rx.App()
app.add_page(index, title="Puzzle Solver")
