import gurobipy as gp
from gurobipy import GRB

from .grids import grid_to_cells, rotate_grid
from .types import Board, Cell, Piece


def _enumerate_flat_placements(
    piece: Piece,
    board_w: int,
    board_h: int,
    board_filled: set[Cell],
) -> list[dict[Cell, int]]:
    """All non-overlapping placements for one piece on the board."""
    placements: list[dict[Cell, int]] = []

    for rot in range(4):
        g = rotate_grid(piece.grid, rot)
        dg = (
            rotate_grid(piece.dots_grid, rot)
            if piece.dots_grid
            else [[0] * 4 for _ in range(4)]
        )
        cells = grid_to_cells(g)
        for oy in range(0, board_h - 4 + 1):
            for ox in range(0, board_w - 4 + 1):
                placed_cells = {(ox + x, oy + y) for (x, y) in cells}
                if placed_cells & board_filled:
                    continue
                # Map each cell to its dot count
                cell_dots = {(ox + x, oy + y): dg[y][x] for (x, y) in cells}
                placements.append(cell_dots)

    uniq: list[dict[Cell, int]] = []
    seen: set[frozenset[Cell]] = set()
    for p in placements:
        fp = frozenset(p.keys())
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append(p)
    return uniq


def solve_flat(board: Board, pieces: list[Piece]) -> dict[str, dict[Cell, int]]:
    board_w, board_h = 10, 7
    board_filled = grid_to_cells(board.grid)
    board_all = {(x, y) for y in range(board_h) for x in range(board_w)}
    target_cells = board_all - board_filled

    placements_by_piece: dict[str, list[dict[Cell, int]]] = {
        p.name: _enumerate_flat_placements(p, board_w, board_h, board_filled)
        for p in pieces
    }

    m = gp.Model("puzzle_flat")
    m.Params.OutputFlag = 1

    xvar: dict[tuple[str, int], gp.Var] = {}
    for name, placements in placements_by_piece.items():
        for pi in range(len(placements)):
            xvar[(name, pi)] = m.addVar(
                vtype=GRB.BINARY, name=f"x[{name},{pi}]"
            )

    for name, placements in placements_by_piece.items():
        m.addConstr(
            gp.quicksum(xvar[(name, pi)] for pi in range(len(placements))) == 1,
            name=f"place_once[{name}]",
        )

    for cell in sorted(target_cells):
        m.addConstr(
            gp.quicksum(
                xvar[(name, pi)]
                for name, placements in placements_by_piece.items()
                for pi, occ in enumerate(placements)
                if cell in occ
            )
            == 1,
            name=f"cover[{cell[0]},{cell[1]}]",
        )

    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"No optimal solution found; Gurobi status={m.Status}"
        )

    solution: dict[str, dict[Cell, int]] = {}
    for name, placements in placements_by_piece.items():
        for pi, occ in enumerate(placements):
            if xvar[(name, pi)].X > 0.5:
                solution[name] = occ
                break
        if name not in solution:
            raise RuntimeError(f"Piece {name} not assigned in solution")

    return solution


def solve_flat_pool(
    board: Board,
    pieces: list[Piece],
    *,
    max_solutions: int = 20,
    output_flag: int = 0,
) -> list[dict[str, dict[Cell, int]]]:
    """Return up to `max_solutions` distinct flat solutions via Gurobi solution pool."""
    board_w, board_h = 10, 7
    board_filled = grid_to_cells(board.grid)
    board_all = {(x, y) for y in range(board_h) for x in range(board_w)}
    target_cells = board_all - board_filled

    placements_by_piece: dict[str, list[dict[Cell, int]]] = {
        p.name: _enumerate_flat_placements(p, board_w, board_h, board_filled)
        for p in pieces
    }

    m = gp.Model("puzzle_flat")
    m.Params.OutputFlag = output_flag
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = max_solutions
    m.Params.PoolGap = 0.0

    xvar: dict[tuple[str, int], gp.Var] = {}
    for name, placements in placements_by_piece.items():
        for pi in range(len(placements)):
            xvar[(name, pi)] = m.addVar(
                vtype=GRB.BINARY, name=f"x[{name},{pi}]"
            )

    for name, placements in placements_by_piece.items():
        m.addConstr(
            gp.quicksum(xvar[(name, pi)] for pi in range(len(placements))) == 1,
            name=f"place_once[{name}]",
        )

    for cell in sorted(target_cells):
        m.addConstr(
            gp.quicksum(
                xvar[(name, pi)]
                for name, placements in placements_by_piece.items()
                for pi, occ in enumerate(placements)
                if cell in occ
            )
            == 1,
            name=f"cover[{cell[0]},{cell[1]}]",
        )

    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"No solution found; Gurobi status={m.Status}")
    if m.SolCount <= 0:
        raise RuntimeError("No solutions in pool")

    sols: list[dict[str, dict[Cell, int]]] = []
    for sn in range(min(m.SolCount, max_solutions)):
        m.Params.SolutionNumber = sn
        sol: dict[str, dict[Cell, int]] = {}
        for name, placements in placements_by_piece.items():
            chosen: int | None = None
            for pi in range(len(placements)):
                if xvar[(name, pi)].Xn > 0.5:
                    chosen = pi
                    break
            if chosen is None:
                raise RuntimeError(
                    f"Incomplete pool solution {sn} for piece {name}"
                )
            sol[name] = placements[chosen]
        sols.append(sol)
    return sols
