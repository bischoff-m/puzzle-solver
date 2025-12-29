import reflex as rx

from .pages.character_table.page import page as character_table_page
from .pages.character_table.state import CharacterTableState
from .pages.frontpage.page import page as frontpage
from .pages.solver.page import page as solver_page
from .pages.solver.state import SolverState

app = rx.App()
app.add_page(frontpage, route="/", title="Puzzle Solver")
app.add_page(
    solver_page,
    route="/solver",
    title="Solver",
    on_load=SolverState.on_load,
)
app.add_page(
    character_table_page,
    route="/character-table",
    title="Character Table",
    on_load=CharacterTableState.on_load,
)
