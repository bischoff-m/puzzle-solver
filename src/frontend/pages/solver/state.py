import plotly.graph_objects as go
import reflex as rx

from puzzle_solver.api import (
    get_puzzle_is_flipped,
    list_puzzle_assets,
    load_puzzle,
    load_puzzle_with_meta,
    resolve_puzzle_asset,
)
from puzzle_solver.api import (
    set_puzzle_is_flipped as api_set_puzzle_is_flipped,
)
from puzzle_solver.cube_solver import solve_cube_pool
from puzzle_solver.flat_solver import solve_flat_pool
from puzzle_solver.plotting import (
    plot_cube_solution,
    plot_flat_board,
    plot_flat_solution,
    plot_pieces_row,
)
from puzzle_solver.types import Face


class SolverState(rx.State):
    puzzles: list[str] = []
    selected_puzzle: str = ""
    max_solutions: int = 50

    puzzle_is_flipped: bool = False

    flat_figure: go.Figure = go.Figure()
    cube_figure: go.Figure = go.Figure()

    flat_board_figure_light: go.Figure = go.Figure()
    flat_board_figure_dark: go.Figure = go.Figure()
    flat_pieces_figure_light: go.Figure = go.Figure()
    flat_pieces_figure_dark: go.Figure = go.Figure()

    flat_solutions: list[dict[str, list[list[int]]]] = []
    cube_solutions: list[dict[str, list[object]]] = []

    flat_solution_index: int = 0
    cube_solution_index: int = 0

    flat_solution_count: int = 0
    cube_solution_count: int = 0

    solving_flat: bool = False
    solving_cube: bool = False
    flat_error: str = ""
    cube_error: str = ""

    @rx.event
    def on_load(self):
        self.puzzles = list_puzzle_assets()
        if not self.selected_puzzle and self.puzzles:
            self.selected_puzzle = self.puzzles[0]
        if self.selected_puzzle:
            self.puzzle_is_flipped = bool(
                get_puzzle_is_flipped(self.selected_puzzle)
            )
        if self.selected_puzzle:
            yield SolverState.select_puzzle(self.selected_puzzle)

    @rx.event
    def set_puzzle_is_flipped(self, value: bool):
        try:
            api_set_puzzle_is_flipped(
                self.selected_puzzle,
                is_flipped=bool(value),
            )
        except Exception as e:
            msg = str(e)
            self.flat_error = msg
            self.cube_error = msg
            return
        yield SolverState.select_puzzle(self.selected_puzzle)

    def _flat_sol_from_json(
        self, sol: dict[str, list[list[int]]]
    ) -> dict[str, set[tuple[int, int]]]:
        out: dict[str, set[tuple[int, int]]] = {}
        for name, cells in sol.items():
            occ: set[tuple[int, int]] = set()
            if isinstance(cells, list):
                for item in cells:
                    if (
                        isinstance(item, list)
                        and len(item) == 2
                        and isinstance(item[0], int)
                        and isinstance(item[1], int)
                    ):
                        occ.add((item[0], item[1]))
            out[str(name)] = occ
        return out

    def _cube_sol_from_json(
        self, sol: dict[str, list[object]]
    ) -> dict[str, tuple[Face, int]]:
        out: dict[str, tuple[Face, int]] = {}
        for name, placement in sol.items():
            face: Face = "+Z"
            rot: int = 0
            if (
                isinstance(placement, list)
                and len(placement) == 2
                and isinstance(placement[0], str)
            ):
                cand = placement[0]
                if cand in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
                    face = cand  # type: ignore[assignment]
                try:
                    v = placement[1]
                    if isinstance(v, bool) or v is None:
                        rot = 0
                    elif isinstance(v, int):
                        rot = v
                    elif isinstance(v, float):
                        rot = int(v)
                    elif isinstance(v, str):
                        rot = int(float(v))
                    else:
                        rot = 0
                except (TypeError, ValueError):
                    rot = 0
            out[str(name)] = (face, rot)
        return out

    def _replot_flat(self) -> None:
        if self.flat_solution_count <= 0:
            self.flat_figure = go.Figure()
            return
        i = max(
            0, min(int(self.flat_solution_index), self.flat_solution_count - 1)
        )
        self.flat_solution_index = i
        path = resolve_puzzle_asset(self.selected_puzzle)
        board, pieces = load_puzzle(path)
        sol = self._flat_sol_from_json(self.flat_solutions[i])
        self.flat_figure = plot_flat_solution(board, sol)

    def _replot_cube(self) -> None:
        if self.cube_solution_count <= 0:
            self.cube_figure = go.Figure()
            return
        i = max(
            0, min(int(self.cube_solution_index), self.cube_solution_count - 1)
        )
        self.cube_solution_index = i
        path = resolve_puzzle_asset(self.selected_puzzle)
        _board, pieces = load_puzzle(path)
        sol = self._cube_sol_from_json(self.cube_solutions[i])
        self.cube_figure = plot_cube_solution(pieces, sol)

    @rx.event
    def select_puzzle(self, puzzle: str):
        self.selected_puzzle = str(puzzle)
        self.flat_error = ""
        self.cube_error = ""
        self.solving_flat = True
        self.solving_cube = True
        yield

        try:
            path = resolve_puzzle_asset(self.selected_puzzle)
        except Exception as e:
            msg = str(e)
            self.flat_error = msg
            self.cube_error = msg
            self.flat_solutions = []
            self.cube_solutions = []
            self.flat_solution_count = 0
            self.cube_solution_count = 0
            self.flat_solution_index = 0
            self.cube_solution_index = 0
            self.flat_figure = go.Figure()
            self.cube_figure = go.Figure()
            self.solving_flat = False
            self.solving_cube = False
            return

        # Load once for solving/plotting.
        try:
            board, pieces, is_flipped = load_puzzle_with_meta(path)
            self.puzzle_is_flipped = bool(is_flipped)
        except Exception as e:
            msg = str(e)
            self.flat_error = msg
            self.cube_error = msg
            self.flat_solutions = []
            self.cube_solutions = []
            self.flat_solution_count = 0
            self.cube_solution_count = 0
            self.flat_solution_index = 0
            self.cube_solution_index = 0
            self.flat_figure = go.Figure()
            self.cube_figure = go.Figure()
            self.flat_board_figure_light = go.Figure()
            self.flat_board_figure_dark = go.Figure()
            self.flat_pieces_figure_light = go.Figure()
            self.flat_pieces_figure_dark = go.Figure()
            self.solving_flat = False
            self.solving_cube = False
            return

        # Flat fallback figures (used when flat has no solutions).
        self.flat_board_figure_light = plot_flat_board(board, theme="light")
        self.flat_board_figure_dark = plot_flat_board(board, theme="dark")
        self.flat_pieces_figure_light = plot_pieces_row(
            pieces, margin=1, theme="light"
        )
        self.flat_pieces_figure_dark = plot_pieces_row(
            pieces, margin=1, theme="dark"
        )

        # Solve flat.
        try:
            flat_sols = solve_flat_pool(
                board,
                pieces,
                max_solutions=int(self.max_solutions),
                output_flag=0,
            )
            self.flat_solutions = [
                {
                    name: [[int(x), int(y)] for (x, y) in sorted(list(occ))]
                    for name, occ in sol.items()
                }
                for sol in flat_sols
            ]
            self.flat_solution_count = len(self.flat_solutions)
            self.flat_solution_index = 0
            if self.flat_solution_count > 0:
                self.flat_figure = plot_flat_solution(
                    board, self._flat_sol_from_json(self.flat_solutions[0])
                )
            else:
                self.flat_error = "No flat solutions found"
                self.flat_figure = go.Figure()
        except Exception as e:
            self.flat_error = str(e)
            self.flat_solutions = []
            self.flat_solution_count = 0
            self.flat_solution_index = 0
            self.flat_figure = go.Figure()

        # Solve cube.
        try:
            cube_sols = solve_cube_pool(
                pieces,
                max_solutions=int(self.max_solutions),
                output_flag=0,
            )
            self.cube_solutions = [
                {
                    name: [str(face), int(rot)]
                    for name, (face, rot) in sol.items()
                }
                for sol in cube_sols
            ]
            self.cube_solution_count = len(self.cube_solutions)
            self.cube_solution_index = 0
            if self.cube_solution_count > 0:
                self.cube_figure = plot_cube_solution(
                    pieces, self._cube_sol_from_json(self.cube_solutions[0])
                )
            else:
                self.cube_error = "No cube solutions found"
                self.cube_figure = go.Figure()
        except Exception as e:
            self.cube_error = str(e)
            self.cube_solutions = []
            self.cube_solution_count = 0
            self.cube_solution_index = 0
            self.cube_figure = go.Figure()
        finally:
            self.solving_flat = False
            self.solving_cube = False

    @rx.event
    def solve_flat(self):
        yield SolverState.select_puzzle(self.selected_puzzle)

    @rx.event
    def solve_cube(self):
        yield SolverState.select_puzzle(self.selected_puzzle)

    @rx.event
    def prev_flat_solution(self):
        if self.flat_solution_count <= 1:
            return
        self.flat_solution_index = max(0, int(self.flat_solution_index) - 1)
        self._replot_flat()

    @rx.event
    def next_flat_solution(self):
        if self.flat_solution_count <= 1:
            return
        self.flat_solution_index = min(
            int(self.flat_solution_count) - 1,
            int(self.flat_solution_index) + 1,
        )
        self._replot_flat()

    @rx.event
    def prev_cube_solution(self):
        if self.cube_solution_count <= 1:
            return
        self.cube_solution_index = max(0, int(self.cube_solution_index) - 1)
        self._replot_cube()

    @rx.event
    def next_cube_solution(self):
        if self.cube_solution_count <= 1:
            return
        self.cube_solution_index = min(
            int(self.cube_solution_count) - 1,
            int(self.cube_solution_index) + 1,
        )
        self._replot_cube()
