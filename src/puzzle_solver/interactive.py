from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button, RadioButtons, Slider

from .plotting import draw_cube_solution, draw_flat_solution
from .types import Board, Cell, Face, Piece


def interactive_solution_viewer(
    *,
    board: Board,
    pieces: list[Piece],
    flat_solutions: list[dict[str, set[Cell]]],
    cube_solutions: list[dict[str, tuple[Face, int]]],
) -> None:
    """Interactive viewer to switch between flat/cube and among solutions."""
    sns.set_theme(style="white")

    if not flat_solutions:
        raise ValueError("flat_solutions is empty")
    if not cube_solutions:
        raise ValueError("cube_solutions is empty")

    fig = plt.figure(figsize=(10, 7))

    main_rect: tuple[float, float, float, float] = (0.05, 0.22, 0.9, 0.73)
    ax2d = fig.add_axes(main_rect)
    ax3d = fig.add_axes(main_rect, projection="3d")
    ax3d.set_visible(False)

    ax_radio = fig.add_axes((0.05, 0.05, 0.15, 0.12))
    radio = RadioButtons(ax_radio, ("flat", "cube"), active=0)

    ax_slider = fig.add_axes((0.27, 0.10, 0.55, 0.03))
    slider_max_init = max(1, len(flat_solutions) - 1)
    slider = Slider(
        ax_slider, "solution", 0, slider_max_init, valinit=0, valstep=1
    )

    ax_prev = fig.add_axes((0.27, 0.05, 0.08, 0.04))
    btn_prev = Button(ax_prev, "Prev")
    ax_next = fig.add_axes((0.37, 0.05, 0.08, 0.04))
    btn_next = Button(ax_next, "Next")

    ax_text = fig.add_axes((0.50, 0.04, 0.45, 0.06))
    ax_text.axis("off")
    status_text = ax_text.text(0.0, 0.5, "", va="center")

    state = {"mode": "flat"}

    def _current_solutions() -> list:
        return flat_solutions if state["mode"] == "flat" else cube_solutions

    def _sync_slider_limits() -> None:
        sols = _current_solutions()
        slider.valmin = 0
        slider.valmax = max(0, len(sols) - 1)
        if slider.valmin == slider.valmax:
            slider.ax.set_xlim(slider.valmin - 0.5, slider.valmax + 0.5)
        else:
            slider.ax.set_xlim(slider.valmin, slider.valmax)
        if slider.val > slider.valmax:
            slider.set_val(slider.valmax)

    def _render() -> None:
        sols = _current_solutions()
        idx = int(slider.val)
        idx = max(0, min(idx, len(sols) - 1))
        if state["mode"] == "flat":
            ax3d.set_visible(False)
            ax2d.set_visible(True)
            draw_flat_solution(ax2d, board, sols[idx])
        else:
            ax2d.set_visible(False)
            ax3d.set_visible(True)
            draw_cube_solution(ax3d, pieces, sols[idx])
        status_text.set_text(f"{state['mode']} solution {idx + 1}/{len(sols)}")
        fig.canvas.draw_idle()

    def _on_mode(label: str | None) -> None:
        if label is None:
            return
        state["mode"] = label
        _sync_slider_limits()
        _render()

    def _on_slider(_val: float) -> None:
        _render()

    def _on_prev(_event) -> None:
        slider.set_val(max(slider.valmin, slider.val - 1))

    def _on_next(_event) -> None:
        slider.set_val(min(slider.valmax, slider.val + 1))

    radio.on_clicked(_on_mode)
    slider.on_changed(_on_slider)
    btn_prev.on_clicked(_on_prev)
    btn_next.on_clicked(_on_next)

    _sync_slider_limits()
    _render()
    plt.show()
