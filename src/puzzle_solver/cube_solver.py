from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from .grids import flip_grid_y, rotate_grid
from .types import Face, Grid, Piece, Voxel


def _face_map(face: Face, u: int, v: int) -> Voxel:
    if face == "+Z":
        return (u, v, 3)
    if face == "-Z":
        return (u, 3 - v, 0)
    if face == "+Y":
        return (u, 3, 3 - v)
    if face == "-Y":
        return (u, 0, v)
    if face == "+X":
        return (3, u, v)
    if face == "-X":
        return (0, u, 3 - v)
    raise ValueError(f"Unknown face: {face}")


def _enumerate_cube_placements(
    piece_grid: Grid,
) -> list[tuple[Face, int, set[Voxel]]]:
    """All (face, rotation) placements of a 4x4 piece on the 4x4x4 cube boundary."""
    placements: list[tuple[Face, int, set[Voxel]]] = []
    faces: list[Face] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

    base_grid = flip_grid_y(piece_grid)

    for face in faces:
        for rot in range(4):
            g = rotate_grid(base_grid, rot)
            occ: set[Voxel] = set()
            for v in range(4):
                for u in range(4):
                    if g[v][u]:
                        occ.add(_face_map(face, u, v))
            placements.append((face, rot, occ))
    return placements


def solve_cube_pool(
    pieces: list[Piece],
    *,
    max_solutions: int = 20,
    output_flag: int = 0,
) -> list[dict[str, tuple[Face, int]]]:
    """Return up to `max_solutions` distinct cube solutions via Gurobi solution pool."""
    boundary: set[Voxel] = set()
    for z in range(4):
        for y in range(4):
            for x in range(4):
                if x in (0, 3) or y in (0, 3) or z in (0, 3):
                    boundary.add((x, y, z))

    piece_grids = {p.name: p.grid for p in pieces}
    placements_by_piece: dict[str, list[tuple[Face, int, set[Voxel]]]] = {
        name: _enumerate_cube_placements(g) for name, g in piece_grids.items()
    }

    # Symmetry breaking: fix the first piece to the top face with a fixed
    # in-plane rotation to avoid counting globally rotated solutions.
    if not pieces:
        raise ValueError("pieces is empty")
    first_piece_name = pieces[0].name
    fixed_face: Face = "+Z"
    fixed_rot = 0
    fixed = [
        p
        for p in placements_by_piece[first_piece_name]
        if p[0] == fixed_face and p[1] == fixed_rot
    ]
    if not fixed:
        raise RuntimeError(
            f"No placements for first piece {first_piece_name} on {fixed_face} with rot={fixed_rot}"
        )
    placements_by_piece[first_piece_name] = fixed

    m = gp.Model("puzzle_cube")
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

    for voxel in sorted(boundary):
        m.addConstr(
            gp.quicksum(
                xvar[(name, pi)]
                for name, placements in placements_by_piece.items()
                for pi, (_face, _rot, occ) in enumerate(placements)
                if voxel in occ
            )
            == 1,
            name=f"cover[{voxel[0]},{voxel[1]},{voxel[2]}]",
        )

    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"No solution found; Gurobi status={m.Status}")
    if m.SolCount <= 0:
        raise RuntimeError("No solutions in pool")

    sols: list[dict[str, tuple[Face, int]]] = []
    for sn in range(min(m.SolCount, max_solutions)):
        m.Params.SolutionNumber = sn
        sol: dict[str, tuple[Face, int]] = {}
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
            face, rot, _ = placements[chosen]
            sol[name] = (face, rot)
        sols.append(sol)
    return sols
