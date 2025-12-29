import reflex as rx

from .state import SolverState


def page() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Puzzle Solver", size="7"),
            rx.text("Solve and visualize the flat and cube solutions."),
            rx.text("Puzzles:"),
            rx.flex(
                rx.foreach(
                    SolverState.puzzles,
                    lambda p: rx.button(
                        p,
                        on_click=SolverState.select_puzzle(p),
                        variant=rx.cond(
                            SolverState.selected_puzzle == p,
                            "solid",
                            "outline",
                        ),
                    ),
                ),
                wrap="wrap",
                gap="0.5em",
                width="100%",
            ),
            rx.hstack(
                rx.button(
                    "Solve",
                    on_click=SolverState.select_puzzle(
                        SolverState.selected_puzzle
                    ),
                    loading=SolverState.solving_flat | SolverState.solving_cube,
                ),
                rx.text("Selected:"),
                rx.text(SolverState.selected_puzzle),
                spacing="3",
                width="100%",
            ),
            rx.divider(),
            rx.heading("Flat", size="5"),
            rx.hstack(
                rx.text("Solutions:"),
                rx.text(SolverState.flat_solution_count),
                rx.spacer(),
                rx.button(
                    "Prev",
                    on_click=SolverState.prev_flat_solution,
                    is_disabled=(SolverState.flat_solution_count <= 1)
                    | (SolverState.flat_solution_index <= 0),
                ),
                rx.text("#"),
                rx.text(SolverState.flat_solution_index + 1),
                rx.button(
                    "Next",
                    on_click=SolverState.next_flat_solution,
                    is_disabled=(SolverState.flat_solution_count <= 1)
                    | (
                        SolverState.flat_solution_index
                        >= (SolverState.flat_solution_count - 1)
                    ),
                ),
                width="100%",
            ),
            rx.cond(
                SolverState.flat_solution_count > 0,
                rx.plotly(
                    data=SolverState.flat_figure, use_resize_handler=True
                ),
                rx.cond(
                    SolverState.flat_error != "",
                    rx.text(SolverState.flat_error),
                ),
            ),
            rx.divider(),
            rx.heading("Cube", size="5"),
            rx.hstack(
                rx.text("Solutions:"),
                rx.text(SolverState.cube_solution_count),
                rx.spacer(),
                rx.button(
                    "Prev",
                    on_click=SolverState.prev_cube_solution,
                    is_disabled=(SolverState.cube_solution_count <= 1)
                    | (SolverState.cube_solution_index <= 0),
                ),
                rx.text("#"),
                rx.text(SolverState.cube_solution_index + 1),
                rx.button(
                    "Next",
                    on_click=SolverState.next_cube_solution,
                    is_disabled=(SolverState.cube_solution_count <= 1)
                    | (
                        SolverState.cube_solution_index
                        >= (SolverState.cube_solution_count - 1)
                    ),
                ),
                width="100%",
            ),
            rx.cond(
                SolverState.cube_solution_count > 0,
                rx.plotly(
                    data=SolverState.cube_figure, use_resize_handler=True
                ),
                rx.cond(
                    SolverState.cube_error != "",
                    rx.text(SolverState.cube_error),
                ),
            ),
            spacing="4",
            width="100%",
        ),
        width="100%",
        max_width="1100px",
        padding_y="2em",
    )
