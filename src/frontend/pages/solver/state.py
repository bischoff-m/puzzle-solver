from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import reflex as rx

from puzzle_solver.api import solve_and_plot_cube, solve_and_plot_flat


class SolverState(rx.State):
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
