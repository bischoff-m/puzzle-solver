import reflex as rx

from .pages.character_table.page import page as character_table_page
from .pages.solver.page import page as solver_page

app = rx.App()
app.add_page(solver_page, route="/", title="Puzzle Solver")
app.add_page(
    character_table_page, route="/character-table", title="Character Table"
)
