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
                rx.spacer(),
                rx.checkbox(
                    "Flip pieces",
                    checked=SolverState.puzzle_is_flipped,
                    on_change=SolverState.set_puzzle_is_flipped,
                ),
                spacing="3",
                width="100%",
                align="center",
            ),
            rx.vstack(
                rx.hstack(
                    rx.text("Randomize Top:", weight="bold"),
                    rx.text("Mean:"),
                    rx.input(
                        value=SolverState.randomize_mean_top.to(str),
                        on_change=SolverState.set_randomize_mean_top,
                        type="number",
                        width="4em",
                    ),
                    rx.text("Var:"),
                    rx.input(
                        value=SolverState.randomize_var_top.to(str),
                        on_change=SolverState.set_randomize_var_top,
                        type="number",
                        width="4em",
                    ),
                    rx.button(
                        "Randomize Top",
                        on_click=SolverState.randomize_top_dots,
                        variant="outline",
                    ),
                    spacing="3",
                    align="center",
                ),
                rx.hstack(
                    rx.text("Randomize Side:", weight="bold"),
                    rx.text("Mean:"),
                    rx.input(
                        value=SolverState.randomize_mean_side.to(str),
                        on_change=SolverState.set_randomize_mean_side,
                        type="number",
                        width="4em",
                    ),
                    rx.text("Var:"),
                    rx.input(
                        value=SolverState.randomize_var_side.to(str),
                        on_change=SolverState.set_randomize_var_side,
                        type="number",
                        width="4em",
                    ),
                    rx.button(
                        "Randomize Side",
                        on_click=SolverState.randomize_side_dots,
                        variant="outline",
                    ),
                    spacing="3",
                    align="center",
                ),
                width="100%",
                align="start",
                spacing="2",
            ),
            rx.hstack(
                rx.text("Legend:", weight="bold"),
                rx.foreach(
                    SolverState.piece_legend,
                    lambda item: rx.hstack(
                        rx.box(
                            width="1em",
                            height="1em",
                            background_color=item["color"],
                            border_radius="2px",
                        ),
                        rx.text(item["name"]),
                        align="center",
                        spacing="1",
                    ),
                ),
                spacing="4",
                wrap="wrap",
                width="100%",
            ),
            rx.divider(),
            rx.heading("Flat", size="5"),
            rx.hstack(
                rx.text("Solutions:"),
                rx.text(SolverState.flat_solution_count),
                rx.spacer(),
                rx.text("X:"),
                rx.text(SolverState.flat_x),
                rx.text("Y:"),
                rx.text(SolverState.flat_y),
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
                rx.vstack(
                    rx.cond(
                        SolverState.flat_error != "",
                        rx.text(SolverState.flat_error),
                    ),
                    rx.color_mode_cond(
                        rx.plotly(
                            data=SolverState.flat_board_figure_light,
                            use_resize_handler=True,
                        ),
                        rx.plotly(
                            data=SolverState.flat_board_figure_dark,
                            use_resize_handler=True,
                        ),
                    ),
                    rx.color_mode_cond(
                        rx.plotly(
                            data=SolverState.flat_pieces_figure_light,
                            use_resize_handler=True,
                        ),
                        rx.plotly(
                            data=SolverState.flat_pieces_figure_dark,
                            use_resize_handler=True,
                        ),
                    ),
                    spacing="3",
                    width="100%",
                ),
            ),
            rx.divider(),
            rx.heading("Cube", size="5"),
            rx.hstack(
                rx.text("Solutions:"),
                rx.text(SolverState.cube_solution_count),
                rx.spacer(),
                rx.text("Shift:"),
                rx.text(SolverState.cube_shift),
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
        margin_x="auto",
        padding_y="2em",
    )
