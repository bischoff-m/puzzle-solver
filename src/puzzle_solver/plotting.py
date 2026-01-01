from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go

from .cube_solver import _enumerate_cube_placements
from .grids import (
    _PIECE_BORDER_COORDS_4X4,
    grid_to_cells,
    rotate_grid,
    rotate_side_grid,
)
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
    board: Board,
    pieces: list[Piece],
    solution: dict[str, dict[Cell, int]],
    *,
    is_flipped: bool = False,
    theme: str = "light",
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

    # Prepare hover text for border voxels (1-12)
    hover_text = np.full((board_h, board_w), "", dtype=object)

    dot_xs, dot_ys = [], []
    for name, occ in piece_items:
        v = piece_index[name]
        # Find the Piece object to account for rotation
        p_obj = next((p for p in pieces if p.name == name), None)
        ox, oy, rot = 0, 0, 0
        found = False
        if p_obj:
            cells_set = set(occ.keys())
            min_x = min(c[0] for c in cells_set)
            max_x = max(c[0] for c in cells_set)
            min_y = min(c[1] for c in cells_set)
            max_y = max(c[1] for c in cells_set)

            for r in range(4):
                g = rotate_grid(p_obj.grid, r)
                rel_cells = grid_to_cells(g)
                # Try to find (ox, oy)
                for cand_ox in range(max_x - 3, min_x + 1):
                    for cand_oy in range(max_y - 3, min_y + 1):
                        if {
                            (cand_ox + rx, cand_oy + ry)
                            for (rx, ry) in rel_cells
                        } == cells_set:
                            ox, oy, rot = cand_ox, cand_oy, r
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        for (x, y), dots in occ.items():
            grid_int[y, x] = v
            if p_obj and found:
                # Find the index in the ORIGINAL piece (unrotated)
                # (x, y) = (ox + rx, oy + ry) where (rx, ry) is in rotated grid
                rx, ry = x - ox, y - oy
                # To find the original (ux, uy), we need to "unrotate" (rx, ry)
                # rotate_grid(grid, r) means:
                # r=1: (x,y) -> (3-y, x)
                # r=2: (x,y) -> (3-x, 3-y)
                # r=3: (x,y) -> (y, 3-x)
                # So unrotate is:
                if rot == 0:
                    ux, uy = rx, ry
                elif rot == 1:
                    ux, uy = ry, 3 - rx
                elif rot == 2:
                    ux, uy = 3 - rx, 3 - ry
                elif rot == 3:
                    ux, uy = 3 - ry, rx
                else:
                    ux, uy = rx, ry

                if (ux, uy) in _PIECE_BORDER_COORDS_4X4:
                    p = _PIECE_BORDER_COORDS_4X4.index((ux, uy))
                    if is_flipped:
                        # Convert position p in flipped border back to YAML index k (0-11)
                        k = 0 if p == 0 else 12 - p
                    else:
                        k = p
                    hover_text[y, x] = f"Index: {k}"

            if 1 <= dots <= 6:
                for dx, dy in _get_dot_offsets(dots):
                    dot_xs.append(x + dx)
                    dot_ys.append(y + dy)

    piece_colors = qualitative_palette(max(6, len(piece_items)))[
        : len(piece_items)
    ]
    t = _character_mapping_theme(theme)
    colors = ["#ffffff", *piece_colors, t["cell_bg"]]
    colorscale = _discrete_colorscale(colors)

    data: list[object] = [
        go.Heatmap(
            z=grid_int,
            zmin=0,
            zmax=frame_value,
            colorscale=colorscale,
            showscale=False,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
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
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        margin=dict(l=0, r=0, t=0, b=0),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
                marker=dict(size=4, color="black", line=dict(width=0)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=140,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text_color"]),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        autorange="reversed",
        constrain="domain",
    )
    return fig


def plot_piece_rotations_sides(
    pieces: list[Piece], *, margin: int = 1, theme: str = "light"
) -> go.Figure:
    """Plot all pieces and their 4 rotations, showing only the first side row.

    The "first row" is the North side strip (4 cells). One row is shown per
    rotation (k=0..3). Pieces are laid out horizontally.
    """
    margin = max(0, int(margin))
    if not pieces:
        return go.Figure()

    pieces_sorted = sorted(pieces, key=lambda p: p.name)
    n_pieces = len(pieces_sorted)

    # Keep only the first row (North strip), one row per rotation,
    # with blank separator rows between rotations.
    rot_gap = 1
    h = 4 + 3 * rot_gap
    w_block = 4
    w = n_pieces * w_block + (n_pieces - 1) * margin

    grid_int = np.zeros((h, w), dtype=int)
    dot_xs, dot_ys = [], []

    for i, p in enumerate(pieces_sorted):
        x0 = i * (w_block + margin)
        for k in range(4):  # 4 rotations -> 4 output rows
            y0 = k * (1 + rot_gap)

            # Rotate grid and side dots
            rotated_grid = rotate_grid(p.grid, k)
            if p.dots_side_grid:
                rotated_side_grid = rotate_side_grid(p.dots_side_grid, k)
            else:
                rotated_side_grid = [[(0, 0, 0, 0)] * 4 for _ in range(4)]

            # North strip only: y=0, x=0..3, side_idx=0
            for cell_idx in range(4):
                present = rotated_grid[0][cell_idx]
                dots = rotated_side_grid[0][cell_idx][0]
                if not present:
                    continue
                x = x0 + (3 - cell_idx)
                grid_int[y0, x] = i + 1
                if 1 <= dots <= 6:
                    for dx, dy in _get_dot_offsets(dots):
                        dot_xs.append(x + dx)
                        dot_ys.append(y0 + dy)

    t = _character_mapping_theme(theme)
    palette = qualitative_palette(max(6, n_pieces))[:n_pieces]
    colors = [t["cell_bg"], *palette]
    colorscale = _discrete_colorscale(colors)

    data: list[object] = [
        go.Heatmap(
            z=grid_int,
            zmin=0,
            zmax=n_pieces,
            colorscale=colorscale,
            showscale=False,
            hoverinfo="skip",
            xgap=0,
            ygap=0,
        )
    ]

    # Draw grid lines for each 4x1 strip cell (including z=0), but do not draw
    # any lines in separator rows or the margin columns between pieces.
    line_xs: list[float | None] = []
    line_ys: list[float | None] = []
    for k in range(4):
        yy = k * (1 + rot_gap)
        for i in range(n_pieces):
            x0_piece = i * (w_block + margin)
            for xx in range(x0_piece, x0_piece + w_block):
                x0 = xx - 0.5
                x1 = xx + 0.5
                y0 = yy - 0.5
                y1 = yy + 0.5
                line_xs.extend([x0, x1, x1, x0, x0, None])
                line_ys.extend([y0, y0, y1, y1, y0, None])

    if line_xs:
        line_color = t["grid_bg"]
        data.append(
            go.Scatter(
                x=line_xs,
                y=line_ys,
                mode="lines",
                line=dict(color=line_color, width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if dot_xs:
        data.append(
            go.Scatter(
                x=dot_xs,
                y=dot_ys,
                mode="markers",
                marker=dict(size=5, color="black", line=dict(width=0)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text_color"]),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        showticklabels=True,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        autorange="reversed",
        constrain="domain",
        tickmode="array",
        tickvals=[k * (1 + rot_gap) for k in range(4)],
        ticktext=["↑", "→", "↓", "←"],
        tickfont=dict(size=20),
        ticklabelstandoff=0,
        ticks="",
    )
    return fig


def _voxels_from_solution(
    pieces: list[Piece], solution: dict[str, tuple[Face, int]]
) -> dict[str, tuple[Face, dict[Voxel, dict[Face, int]]]]:
    piece_by_name: dict[str, Piece] = {p.name: p for p in pieces}

    out: dict[str, tuple[Face, dict[Voxel, dict[Face, int]]]] = {}
    for name, (face, rot) in solution.items():
        piece = piece_by_name.get(name)
        if piece is None:
            raise KeyError(f"Piece {name!r} not found in pieces")

        occ: dict[Voxel, dict[Face, int]] | None = None
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
        for (x, y, z), d_dict in sorted(occ.items()):
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
            for d_face, dots in d_dict.items():
                if dots > 0:
                    _add_dots_3d(dot_xs, dot_ys, dot_zs, dots, d_face, x, y, z)

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
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
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
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
