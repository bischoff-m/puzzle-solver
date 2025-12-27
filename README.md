# Puzzle Solver (Gurobi IP)

This repo models a 3D-printed voxel puzzle as an **integer program** and solves it with **gurobipy**.

You specify:

- **6 pieces**, each a 4×4 voxel pattern.

  - The **inner 2×2** voxels are **always filled**.
  - The **outer border** voxels (12 cells) are configurable as a boolean array `border12`.

- A **10×7 board**.

  - The **inner 8×5** region is **always missing** (the cavity).
  - The **outer border** voxels (30 cells) are configurable as a boolean array `border30`.

The goal (flat scenario) is to place the 6 pieces onto the board (translation + rotation) so that:

- The pieces do not overlap.
- They do not overlap pre-filled board voxels.
- Every missing board cell is filled by exactly one piece.

The goal (cube scenario) is to place the 6 pieces on the **surface** of a 4×4×4 cube (each piece covers one face, rotated in-plane) so that every boundary voxel is filled exactly once.

The “bottom marked” requirement is interpreted as **no mirroring/flipping**: only rotations by 0/90/180/270 degrees in the plane are allowed.

---

## Border Array Conventions

### Piece `border12` (4×4)

Length 12, enumerating border coordinates **clockwise starting at top-left** `(0,0)`:

Top row:

`(0,0) (1,0) (2,0) (3,0)`

Right column (excluding corners):

`(3,1) (3,2)`

Bottom row (right-to-left):

`(3,3) (2,3) (1,3) (0,3)`

Left column (excluding corners, bottom-to-top):

`(0,2) (0,1)`

The inner 2×2 cells `(1,1),(2,1),(1,2),(2,2)` are always `True`.

### Board `border30` (10×7)

Length 30, enumerating border coordinates **clockwise starting at top-left** `(0,0)` in the same way:

- Top row: 10 cells
- Right column (excluding corners): 5 cells
- Bottom row: 10 cells
- Left column (excluding corners): 5 cells

The inner cavity cells `(x,y)` with `x=1..8` and `y=1..5` are always missing (`False`).

---

## Validation Printing

Pattern parsing lives in the `puzzle_solver` library (see `puzzle_solver.grids` and `puzzle_solver.yaml_io`).

---

## Mathematical Formulation (Integer Programming)

Both the flat-board and cube-shell versions can be written as a **0–1 exact cover** model.

### Common notation

- Pieces: $P = \{1,\dots,6\}$.
- For each piece $p\in P$, let $K_p$ be the set of all **allowed placements** (pre-enumerated).
- For each placement $k\in K_p$, define a binary incidence constant:

  - Flat: $a_{p,k,c} \in \{0,1\}$ indicates whether placement $k$ of piece $p$ covers board cell $c$.
  - Cube: $a_{p,k,v} \in \{0,1\}$ indicates whether placement $k$ of piece $p$ covers boundary voxel $v$.

Decision variables:

$$
x_{p,k} \in \{0,1\}\quad \forall p\in P,\; k\in K_p
$$

where $x_{p,k}=1$ means “choose placement $k$ for piece $p$”.

Objective:

- Pure feasibility is typical. In Gurobi we set `min 0`.

$$
\min 0
$$

---

### Flat board scenario

Board dimensions are 10×7.

- Let $B$ be the set of all board cells.
- Let $F\subseteq B$ be the set of board cells already filled by the board frame (from `border30`).
- Let $T = B\setminus F$ be the set of target (missing) cells that must be filled by pieces.

Placements $K_p$:

- For each piece $p$, generate all translations $(o_x,o_y)$ such that the 4×4 pattern fits within the board.
- Allow in-plane rotations $r\in\{0,1,2,3\}$ representing 0/90/180/270 degrees.
- Exclude any placement that overlaps $F$.

Constraints:

1. Each piece is placed exactly once:

$$
\sum_{k\in K_p} x_{p,k} = 1\quad \forall p\in P
$$

1. Every target cell is covered exactly once:

$$
\sum_{p\in P}\sum_{k\in K_p} a_{p,k,c}\,x_{p,k} = 1\quad \forall c\in T
$$

Notes:

- Non-overlap is implied by the exact-cover constraints on $T$.
- Overlap with the pre-filled frame $F$ is prevented by construction (placements overlapping $F$ are not included in $K_p$).

---

### 4×4×4 cube “shell” scenario

The cube has coordinates $(x,y,z)$ with each dimension in $\{0,1,2,3\}$.

- Let $V$ be the set of boundary voxels (the “shell”):

$$
V = \{(x,y,z): x\in\{0,3\}\;\text{or}\;y\in\{0,3\}\;\text{or}\;z\in\{0,3\}\}
$$

Placements $K_p$:

- Each piece is placed on exactly one of 6 faces: $\{\pm X,\pm Y,\pm Z\}$.
- For a chosen face, allow in-plane rotations $r\in\{0,1,2,3\}$.
- No translations: the piece always spans the full 4×4 face.
- The incidence constants $a_{p,k,v}$ come from mapping the 4×4 face coordinates to cube coordinates.

Constraints:

1. Each piece chooses exactly one face/rotation:

$$
\sum_{k\in K_p} x_{p,k} = 1\quad \forall p\in P
$$

1. Every boundary voxel is covered exactly once:

$$
\sum_{p\in P}\sum_{k\in K_p} a_{p,k,v}\,x_{p,k} = 1\quad \forall v\in V
$$

Interpretation:

- This is an exact cover of the cube shell by 6 patterned faces.
- If your physical “interlock” rules require additional constraints beyond voxel occupancy (e.g., keyed connectors, forbidden face adjacencies, or edge compatibility), those would be additional constraints layered on top of this base model.

---

## Running

- Edit [puzzle.yaml](puzzle.yaml) to change the board/pieces.
- Run the Reflex UI: `reflex run`
