from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from .grids import grid_to_cells, rotate_grid
from .types import Board, Cell, Grid, Piece


def _enumerate_flat_placements(
    piece_grid: Grid,
    board_w: int,
    board_h: int,
    board_filled: set[Cell],
) -> list[set[Cell]]:
    """All non-overlapping placements for one piece on the board."""
    placements: list[set[Cell]] = []

    for rot in range(4):
        g = rotate_grid(piece_grid, rot)
        cells = grid_to_cells(g)
        for oy in range(0, board_h - 4 + 1):
            for ox in range(0, board_w - 4 + 1):
                placed = {(ox + x, oy + y) for (x, y) in cells}
                if placed & board_filled:
                    continue
                placements.append(placed)

    uniq: list[set[Cell]] = []
    seen: set[frozenset[Cell]] = set()
    for p in placements:
        fp = frozenset(p)
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append(p)
    return uniq


def solve_flat(board: Board, pieces: list[Piece]) -> dict[str, set[Cell]]:
    board_w, board_h = 10, 7
    board_filled = grid_to_cells(board.grid)
    board_all = {(x, y) for y in range(board_h) for x in range(board_w)}
    target_cells = board_all - board_filled

    piece_grids = {p.name: p.grid for p in pieces}
    placements_by_piece: dict[str, list[set[Cell]]] = {
        name: _enumerate_flat_placements(g, board_w, board_h, board_filled)
        for name, g in piece_grids.items()
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

    solution: dict[str, set[Cell]] = {}
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
) -> list[dict[str, set[Cell]]]:
    """Return up to `max_solutions` distinct flat solutions via Gurobi solution pool."""
    board_w, board_h = 10, 7
    board_filled = grid_to_cells(board.grid)
    board_all = {(x, y) for y in range(board_h) for x in range(board_w)}
    target_cells = board_all - board_filled

    piece_grids = {p.name: p.grid for p in pieces}
    placements_by_piece: dict[str, list[set[Cell]]] = {
        name: _enumerate_flat_placements(g, board_w, board_h, board_filled)
        for name, g in piece_grids.items()
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

    sols: list[dict[str, set[Cell]]] = []
    for sn in range(min(m.SolCount, max_solutions)):
        m.Params.SolutionNumber = sn
        sol: dict[str, set[Cell]] = {}
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
