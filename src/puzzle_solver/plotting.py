from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go

from .cube_solver import _enumerate_cube_placements
from .grids import grid_to_cells
from .types import Board, Cell, Face, Piece, Voxel


def _discrete_colorscale(colors: list[str]) -> list[tuple[float, str]]:
    """Build a Plotly colorscale with hard steps for integer categories."""
    if not colors:
        raise ValueError("colors must be non-empty")

    n = len(colors)
    if n == 1:
        return [(0.0, colors[0]), (1.0, colors[0])]

    scale: list[tuple[float, str]] = []
    for i, c in enumerate(colors):
        lo = i / n
        hi = (i + 1) / n
        scale.append((lo, c))
        scale.append((hi, c))
    scale[0] = (0.0, scale[0][1])
    scale[-1] = (1.0, scale[-1][1])
    return scale


def _qualitative_palette(n: int) -> list[str]:
    # Avoid reaching into Plotly's template typing (varies across versions and
    # is not always modeled in type stubs). Keep a stable default palette.
    base = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def plot_flat_solution(
    board: Board, solution: dict[str, set[Cell]]
) -> go.Figure:
    """Return a Plotly figure for the flat-board solution."""
    board_filled = grid_to_cells(board.grid)
    board_w, board_h = 10, 7

    piece_items = sorted(solution.items(), key=lambda kv: kv[0])
    piece_index: dict[str, int] = {
        name: i + 1 for i, (name, _) in enumerate(piece_items)
    }
    frame_value = len(piece_items) + 1

    grid_int = np.zeros((board_h, board_w), dtype=int)
    for x, y in board_filled:
        grid_int[y, x] = frame_value
    for name, occ in piece_items:
        v = piece_index[name]
        for x, y in occ:
            grid_int[y, x] = v

    piece_colors = _qualitative_palette(max(6, len(piece_items)))[
        : len(piece_items)
    ]
    colors = ["#ffffff", *piece_colors, "#c0c0c0"]
    colorscale = _discrete_colorscale(colors)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=grid_int,
                zmin=0,
                zmax=frame_value,
                colorscale=colorscale,
                showscale=False,
                hoverinfo="skip",
                xgap=1,
                ygap=1,
            )
        ]
    )

    fig.update_layout(
        title="Flat solution",
        margin=dict(l=10, r=10, t=50, b=10),
        height=450,
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        constrain="domain",
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        autorange="reversed",
    )
    return fig


def _voxels_from_solution(
    pieces: list[Piece], solution: dict[str, tuple[Face, int]]
) -> dict[str, set[Voxel]]:
    piece_by_name: dict[str, Piece] = {p.name: p for p in pieces}

    out: dict[str, set[Voxel]] = {}
    for name, (face, rot) in solution.items():
        piece = piece_by_name.get(name)
        if piece is None:
            raise KeyError(f"Piece {name!r} not found in pieces")

        occ: set[Voxel] | None = None
        for f, r, voxels in _enumerate_cube_placements(piece.grid):
            if f == face and r == rot:
                occ = voxels
                break
        if occ is None:
            raise RuntimeError(
                f"Could not reconstruct placement for piece {name}"
            )
        out[name] = occ
    return out


def _add_voxel_cube_to_mesh(
    *,
    x0: float,
    y0: float,
    z0: float,
    size: float,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    ii: list[int],
    jj: list[int],
    kk: list[int],
) -> None:
    """Append a unit cube (12 triangles) to a growing Mesh3d buffer."""
    base = len(xs)
    x1, y1, z1 = x0 + size, y0 + size, z0 + size

    # 8 vertices of the cube.
    verts = [
        (x0, y0, z0),
        (x0, y1, z0),
        (x1, y1, z0),
        (x1, y0, z0),
        (x0, y0, z1),
        (x0, y1, z1),
        (x1, y1, z1),
        (x1, y0, z1),
    ]
    for x, y, z in verts:
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # 12 triangles forming cube faces (same topology as Plotly docs “Mesh Cube”).
    i_loc = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j_loc = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k_loc = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    for a, b, c in zip(i_loc, j_loc, k_loc, strict=True):
        ii.append(base + a)
        jj.append(base + b)
        kk.append(base + c)


def plot_cube_solution(
    pieces: list[Piece], solution: dict[str, tuple[Face, int]]
) -> go.Figure:
    """Return a Plotly 3D figure for the cube solution.

    Rendered as voxel cubes (triangulated mesh), colored by piece.
    """
    voxels_by_piece = _voxels_from_solution(pieces, solution)

    names_sorted = sorted(voxels_by_piece.keys())
    palette = _qualitative_palette(max(6, len(names_sorted)))[
        : len(names_sorted)
    ]
    name_to_color = {name: palette[i] for i, name in enumerate(names_sorted)}

    # Build one mesh per piece (keeps coloring simple and hover meaningful).
    data: list[object] = []
    voxel_size = 1.0
    inset = 0.02  # tiny inset to reduce z-fighting between adjacent cubes

    for name in names_sorted:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        ii: list[int] = []
        jj: list[int] = []
        kk: list[int] = []

        for x, y, z in sorted(voxels_by_piece[name]):
            _add_voxel_cube_to_mesh(
                x0=float(x) + inset,
                y0=float(y) + inset,
                z0=float(z) + inset,
                size=voxel_size - 2 * inset,
                xs=xs,
                ys=ys,
                zs=zs,
                ii=ii,
                jj=jj,
                kk=kk,
            )

        data.append(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=ii,
                j=jj,
                k=kk,
                color=name_to_color[name],
                opacity=1.0,
                flatshading=True,
                name=name,
                hovertemplate=f"{name}<extra></extra>",
                showscale=False,
            )
        )

    fig = go.Figure(data=data)

    fig.update_layout(
        title="Cube solution",
        margin=dict(l=10, r=10, t=50, b=10),
        height=600,
        scene=dict(
            xaxis=dict(range=[-0.25, 4.25], dtick=1, title="x"),
            yaxis=dict(range=[-0.25, 4.25], dtick=1, title="y"),
            zaxis=dict(range=[-0.25, 4.25], dtick=1, title="z"),
            aspectmode="cube",
        ),
        showlegend=True,
    )
    return fig


def plot_grids(grids: Iterable[tuple[str, list[list[bool]]]]) -> go.Figure:
    """Minimal helper: plot a boolean 2D grid (first item) as a heatmap."""
    items = list(grids)
    if not items:
        return go.Figure()

    title, grid = items[0]
    data = np.array(grid, dtype=int)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=data,
                zmin=0,
                zmax=1,
                colorscale=_discrete_colorscale(["#ffffff", "#222222"]),
                showscale=False,
                hoverinfo="skip",
                xgap=1,
                ygap=1,
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        autorange="reversed",
    )
    return fig
