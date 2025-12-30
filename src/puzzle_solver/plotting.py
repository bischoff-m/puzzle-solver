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


def _get_dot_offsets(n: int) -> list[tuple[float, float]]:
    """Relative offsets for 1-6 dots in a unit square [-0.5, 0.5]."""
    s = 0.25
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(-s, -s), (s, s)]
    if n == 3:
        return [(-s, -s), (0.0, 0.0), (s, s)]
    if n == 4:
        return [(-s, -s), (-s, s), (s, -s), (s, s)]
    if n == 5:
        return [(-s, -s), (-s, s), (0.0, 0.0), (s, -s), (s, s)]
    if n == 6:
        return [(-s, -s), (-s, 0.0), (-s, s), (s, -s), (s, 0.0), (s, s)]
    return []


def qualitative_palette(n: int) -> list[str]:
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


def _character_mapping_theme(theme: str) -> dict[str, str]:
    theme = str(theme or "light").lower()
    if theme == "dark":
        return {
            "cell_bg": "rgba(255,255,255,0.15)",
            "label_bg": "rgba(255,255,255,0.25)",
            "grid_bg": "rgba(0,0,0,1)",
            "text_color": "white",
        }
    return {
        "cell_bg": "rgba(0,0,0,0.15)",
        "label_bg": "rgba(0,0,0,0.25)",
        "grid_bg": "rgba(255,255,255,1)",
        "text_color": "black",
    }


def plot_flat_solution(
    board: Board, solution: dict[str, dict[Cell, int]]
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

    dot_xs, dot_ys = [], []
    for name, occ in piece_items:
        v = piece_index[name]
        for (x, y), dots in occ.items():
            grid_int[y, x] = v
            if 1 <= dots <= 6:
                for dx, dy in _get_dot_offsets(dots):
                    dot_xs.append(x + dx)
                    dot_ys.append(y + dy)

    piece_colors = qualitative_palette(max(6, len(piece_items)))[
        : len(piece_items)
    ]
    colors = ["#ffffff", *piece_colors, "#c0c0c0"]
    colorscale = _discrete_colorscale(colors)

    data: list[object] = [
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

    if dot_xs:
        data.append(
            go.Scatter(
                x=dot_xs,
                y=dot_ys,
                mode="markers",
                marker=dict(size=8, color="black", line=dict(width=0)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=data)

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


def plot_flat_board(board: Board, *, theme: str = "light") -> go.Figure:
    """Plot just the board (frame cells) without placing any pieces."""

    board_filled = grid_to_cells(board.grid)
    board_w, board_h = 10, 7

    grid_int = np.zeros((board_h, board_w), dtype=int)
    for x, y in board_filled:
        grid_int[y, x] = 1

    t = _character_mapping_theme(theme)
    colorscale = _discrete_colorscale([t["cell_bg"], t["label_bg"]])

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=grid_int,
                zmin=0,
                zmax=1,
                colorscale=colorscale,
                showscale=False,
                hoverinfo="skip",
                xgap=1,
                ygap=1,
            )
        ]
    )
    fig.update_layout(
        title="Board",
        margin=dict(l=10, r=10, t=50, b=10),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["grid_bg"],
        font=dict(color=t["text_color"]),
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


def plot_pieces_row(
    pieces: list[Piece], *, margin: int = 1, theme: str = "light"
) -> go.Figure:
    """Plot all pieces next to each other in one row.

    Each piece is rendered as a 4x4 grid. Pieces are laid out left-to-right
    with `margin` empty columns between them.
    """

    margin = max(0, int(margin))
    if not pieces:
        return go.Figure()

    pieces_sorted = sorted(pieces, key=lambda p: p.name)
    n = len(pieces_sorted)
    h, w_piece = 4, 4
    w = n * w_piece + (n - 1) * margin

    grid_int = np.zeros((h, w), dtype=int)
    dot_xs, dot_ys = [], []

    for i, p in enumerate(pieces_sorted):
        x0 = i * (w_piece + margin)
        for y in range(4):
            for x in range(4):
                if p.grid[y][x]:
                    grid_int[y, x0 + x] = i + 1
                    dots = p.dots_grid[y][x] if p.dots_grid else 0
                    if 1 <= dots <= 6:
                        for dx, dy in _get_dot_offsets(dots):
                            dot_xs.append(x0 + x + dx)
                            dot_ys.append(y + dy)

    t = _character_mapping_theme(theme)
    palette = qualitative_palette(max(6, n))[:n]
    colors = [t["cell_bg"], *palette]
    colorscale = _discrete_colorscale(colors)

    data: list[object] = [
        go.Heatmap(
            z=grid_int,
            zmin=0,
            zmax=n,
            colorscale=colorscale,
            showscale=False,
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        )
    ]

    if dot_xs:
        data.append(
            go.Scatter(
                x=dot_xs,
                y=dot_ys,
                mode="markers",
                marker=dict(size=8, color="black", line=dict(width=0)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Pieces",
        margin=dict(l=10, r=10, t=50, b=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["grid_bg"],
        font=dict(color=t["text_color"]),
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


def _voxels_from_solution(
    pieces: list[Piece], solution: dict[str, tuple[Face, int]]
) -> dict[str, tuple[Face, dict[Voxel, int]]]:
    piece_by_name: dict[str, Piece] = {p.name: p for p in pieces}

    out: dict[str, tuple[Face, dict[Voxel, int]]] = {}
    for name, (face, rot) in solution.items():
        piece = piece_by_name.get(name)
        if piece is None:
            raise KeyError(f"Piece {name!r} not found in pieces")

        occ: dict[Voxel, int] | None = None
        for f, r, voxels in _enumerate_cube_placements(piece):
            if f == face and r == rot:
                occ = voxels
                break
        if occ is None:
            raise RuntimeError(
                f"Could not reconstruct placement for piece {name}"
            )
        out[name] = (face, occ)
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


def _add_dots_3d(
    dot_xs: list[float],
    dot_ys: list[float],
    dot_zs: list[float],
    n: int,
    face: Face,
    x: int,
    y: int,
    z: int,
) -> None:
    offsets_2d = _get_dot_offsets(n)
    for du, dv in offsets_2d:
        # du, dv are in [-0.25, 0.25]
        if face == "+Z":
            dot_xs.append(x + 0.5 + du)
            dot_ys.append(y + 0.5 + dv)
            dot_zs.append(z + 1.01)
        elif face == "-Z":
            dot_xs.append(x + 0.5 + du)
            dot_ys.append(y + 0.5 + dv)
            dot_zs.append(z - 0.01)
        elif face == "+Y":
            dot_xs.append(x + 0.5 + du)
            dot_ys.append(y + 1.01)
            dot_zs.append(z + 0.5 + dv)
        elif face == "-Y":
            dot_xs.append(x + 0.5 + du)
            dot_ys.append(y - 0.01)
            dot_zs.append(z + 0.5 + dv)
        elif face == "+X":
            dot_xs.append(x + 1.01)
            dot_ys.append(y + 0.5 + du)
            dot_zs.append(z + 0.5 + dv)
        elif face == "-X":
            dot_xs.append(x - 0.01)
            dot_ys.append(y + 0.5 + du)
            dot_zs.append(z + 0.5 + dv)


def plot_cube_solution(
    pieces: list[Piece], solution: dict[str, tuple[Face, int]]
) -> go.Figure:
    """Return a Plotly 3D figure for the cube solution.

    Rendered as voxel cubes (triangulated mesh), colored by piece.
    """
    voxels_by_piece = _voxels_from_solution(pieces, solution)

    names_sorted = sorted(voxels_by_piece.keys())
    palette = qualitative_palette(max(6, len(names_sorted)))[
        : len(names_sorted)
    ]
    name_to_color = {name: palette[i] for i, name in enumerate(names_sorted)}

    # Build one mesh per piece (keeps coloring simple and hover meaningful).
    data: list[object] = []
    voxel_size = 1.0
    inset = 0.02  # tiny inset to reduce z-fighting between adjacent cubes

    dot_xs, dot_ys, dot_zs = [], [], []

    for name in names_sorted:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        ii: list[int] = []
        jj: list[int] = []
        kk: list[int] = []

        face, occ = voxels_by_piece[name]
        for (x, y, z), dots in sorted(occ.items()):
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
            if dots > 0:
                _add_dots_3d(dot_xs, dot_ys, dot_zs, dots, face, x, y, z)

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

    if dot_xs:
        data.append(
            go.Scatter3d(
                x=dot_xs,
                y=dot_ys,
                z=dot_zs,
                mode="markers",
                marker=dict(size=6, color="black", line=dict(width=0)),
                name="Dots",
                showlegend=False,
                hoverinfo="skip",
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
