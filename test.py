from main import BoardSpec, PieceSpec, demo_validate_patterns


def main() -> None:
    # TODO: Replace this example data with your real puzzle definition.
    # Piece border12 uses the following order (clockwise starting at (0,0)):
    # (0,0)(1,0)(2,0)(3,0)(3,1)(3,2)(3,3)(2,3)(1,3)(0,3)(0,2)(0,1)
    example_pieces = [
        PieceSpec(
            "A",
            (
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
            ),
        ),
        PieceSpec(
            "B",
            (
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                True,
            ),
        ),
        PieceSpec(
            "C",
            (
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
            ),
        ),
        PieceSpec(
            "D",
            (
                True,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ),
        ),
        PieceSpec(
            "E",
            (
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
            ),
        ),
        PieceSpec(
            "F",
            (
                False,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                True,
            ),
        ),
    ]

    # Board border30 order is clockwise starting at (0,0) along the 10x7 frame.
    example_board = BoardSpec(
        (
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            #
            True,
            True,
            False,
            True,
            True,
            True,
            #
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            #
            False,
            False,
            True,
            False,
            True,
            False,
        )
    )

    demo_validate_patterns(example_board, example_pieces)

    # Uncomment to solve once you replace example data with real data.
    # flat_sol = solve_flat(example_board, example_pieces)
    # print_flat_solution(example_board, flat_sol)
    # cube_sol = solve_cube(example_pieces)
    # print(cube_sol)


if __name__ == "__main__":
    main()
