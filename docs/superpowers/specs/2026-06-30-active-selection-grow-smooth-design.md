# Active Face Selection + Grow + Smooth — Design Spec

- **Date:** 2026-06-30
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)
- **Part of:** a larger "shared selection subsystem". This spec is chunk 1 of 2
  (selection core + grow + smooth). Chunk 2 (populate selection from mesh-check
  defects: open/boundary/non-manifold edges) is a separate later spec.

## 1. Goal

Add a reusable **active face selection** to the GUI and three things that use it:
a lasso **Select** mode, a **Grow** button that dilates the selection by face
rings, and a **Smooth** button that Laplacian-smooths the selected patch
("moving average"). The selection is a shared concept future tools can also use.

## 1.5 Workflow & data model

There is one evolving **active base mesh** (`STLClipperEngine.original_mesh`).
Cleanup operations — **trim** and **smooth** — mutate it in place and **stack** on
each other; each acts on the result of the previous, and `Ctrl+Z` walks back through
them via one shared history. **Clips are a separate parametric layer**, not baked into
the base: `recompute_all()` re-applies the clip list on top of the *current* base to
produce the export geometry (`_wall_mesh`). Intended flow:

```
load STL → trim / smooth / grow-select (repeat to tidy the raw geometry)
        → clip (define inlet/outlet patches) → export for CFD
```

This is why v1 edits target the unclipped base and the edit buttons are disabled once
clips exist: **cleanup happens first; clipping is the final, parametric step.**

## 2. Requirements (decided in brainstorming)

| # | Decision |
|---|----------|
| Selection unit | **Faces (cell ids)** of the base mesh |
| Persistence | Selection stays active across mode toggles; cleared by Escape, empty-space click, a new lasso, or any topology change (load/trim/undo). Smoothing keeps it (topology unchanged). |
| Select interaction | Lasso (reuses trim lasso). Each lasso **replaces** the selection. |
| Selected faces | Front-facing only (cell-normal · view-direction < 0) and inside the screen polygon |
| Grow | One face-ring per press, repeatable |
| Smooth | Constrained Laplacian, fixed **5 iterations per press**, repeatable; unselected points stay fixed |
| Undo | Reuses the existing trim history + `Ctrl+Z` |
| v1 target mesh | Operates on the **unclipped base** `original_mesh`; Select/Smooth/Grow are **disabled when clips exist** (mapping a selection through clips is out of v1) |

## 3. Architecture — shared active selection

```
ACTIVE SELECTION  = set[int] of original_mesh cell ids   (held by STLClipperApp)
  sources   : lasso (Select mode)                         [chunk 1]
              mesh-check defects -> adjacent faces         [chunk 2, later]
  modifier  : Grow (face-neighbor dilation)                [chunk 1]
  shown as  : highlighted overlay actor                    [chunk 1]
  consumed  : Smooth (Laplacian on the selection's points) [chunk 1]
```

The **engine** stays selection-agnostic: it exposes pure operations that take
cell ids + viewing params and return ids or a new mesh. The **app** owns the
selection set and the highlight actor. This keeps engine ops headless-testable
and lets any future tool read the same `self._selection`.

## 4. Engine interfaces (headless, unit-tested)

All operate on `self.original_mesh` (a triangulated `pv.PolyData`).

- `select_cells_in_polygon(polygon_xy, view_matrix, viewport, view_direction, front_only=True) -> list[int]`
  - Project each cell **centroid** (`cell_centers().points`) through `view_matrix`
    to display space (reusing the trim projection math), test inside `polygon_xy`
    (the existing `_points_in_polygon` helper). If `front_only`, also require
    `cell_normal · view_direction < 0` (normal points back toward the camera),
    using `original_mesh.cell_normals`. Returns the matching cell ids.
- `grow_cells(cell_ids, rings=1) -> list[int]`
  - Union of `cell_ids` with their face neighbors via
    `cell_neighbors(cid, connections='points')`, repeated `rings` times.
- `smooth_cells(cell_ids, iterations=5, relaxation=0.5) -> pv.PolyData | None`
  - **Movable** = every point belonging to a selected cell; **fixed** = all other
    points. For `iterations` rounds, each movable point steps toward the mean of its
    `point_neighbors`: `p ← (1-relaxation)·p + relaxation·mean(neighbors)`. Because
    fixed points never move, a movable point on the selection rim averages partly
    against fixed neighbors, so the patch **blends** into the untouched surface
    instead of stepping; topology is unchanged. Push `self._trim_history` (shared
    undo), set `self.original_mesh`
    to the smoothed copy, return `self.recompute_all()`. No-op (`None`) on empty
    selection.

Reuse: factor the trim projection (homogeneous transform → NDC → display, bottom-left
origin) into a shared module helper `_project_to_display(points, view_matrix, viewport)`
used by both `trim_by_screen_polygon` and `select_cells_in_polygon`.

Undo: `smooth_cells` uses the existing `_trim_history` (`stl_clipper.py:249`) and
`undo_trim` (`:489`) — they snapshot/restore `original_mesh`, which is exactly what
smoothing needs. (Names stay as-is to avoid touching working trim code; they are now
the shared geometry-edit history.)

## 5. App (GUI) layer

- **State:** `self._selection: set[int]` (cell ids); `self._selection_actor` (highlight).
- **Select mode:** "🖈 Select" checkable button. Reuses the trim lasso capture
  (`vtkInteractorStyleUser` observers + yellow overlay). On release: compute
  `view_matrix` + viewport (as `_apply_trim` does) and `view_direction` from
  `self.plotter.camera.direction`; call `select_cells_in_polygon(...)`; **replace**
  `self._selection`; refresh highlight. Selection persists when the mode is toggled off.
- **Highlight:** `_refresh_selection_highlight()` adds/updates an actor showing
  `original_mesh.extract_cells(sorted(selection))` in a bright color (e.g. orange),
  `name="selection"`. Cleared when selection empties.
- **Grow button:** "➕ Grow" → `self._selection = set(engine.grow_cells(self._selection, 1))`; refresh highlight.
- **Smooth button:** "✨ Smooth (×5)" → `engine.smooth_cells(self._selection, iterations=5)`;
  on success `_refresh_display()` + re-highlight (ids still valid — topology unchanged); status message.
- **Clear:** Escape, click on empty background, or a new lasso. Topology changes
  (load/trim/undo) clear `self._selection` and remove the highlight.
- **Clips guard:** Select/Grow/Smooth buttons disabled (with a status hint
  "clear clips to edit the base mesh") whenever `engine.clips` is non-empty.
- `Ctrl+Z` already calls `undo_trim` → also undoes a smooth.

## 6. Refactor (shared lasso)

Extract the trim lasso plumbing (`_toggle_trim_mode` body, `_on_trim_press/move/release`,
overlay, `_trim_safe`) into a small reusable helper parameterized by an **on-release
callback** (display points → action). Trim passes its `_apply_trim`; Select passes a
new `_apply_select`. Keeps one copy of the `vtkInteractorStyleUser` + overlay code
(the part that was hard to get right). Trim behavior must remain unchanged.

## 7. Data flow

```
Select mode lasso → release → view_matrix+viewport+view_direction
   → engine.select_cells_in_polygon → self._selection (replace) → highlight
Grow press   → engine.grow_cells(self._selection,1) → self._selection → highlight
Smooth press → engine.smooth_cells(self._selection,5) → original_mesh updated
   → recompute_all → _refresh_display → re-highlight   (Ctrl+Z → undo_trim)
```

## 8. Edge cases / error handling

- Empty / off-mesh lasso → selection unchanged or cleared; status note.
- Smooth with empty selection → no-op, no history push.
- Grow at full mesh → selection saturates (no error).
- Mesh has no normals → compute via `cell_normals` (pyvista computes on access).
- Clips present → edit buttons disabled (guard above).
- Selection ids are validated against `original_mesh.n_cells` before use (drop stale).

## 9. Testing (headless, pytest)

- `grow_cells`: on a known small mesh, growing a single cevll by 1 ring returns
  exactly that cell + its point-connected neighbors; 2 rings ⊇ 1 ring.
- `smooth_cells`: build a mesh with one displaced (noisy) interior point in the
  selected set; after smoothing, that point moved toward its neighbors' mean and
  **every unselected point is byte-identical**; history grew by 1; `undo_trim` restores.
- `select_cells_in_polygon`: with a known orthographic `view_matrix`, viewport, and
  `view_direction`, a polygon over a known region returns the expected front cells
  and excludes back-facing ones.

## 10. Scope

**v1 (this spec):** active cell selection; lasso Select mode (replace) + highlight;
Grow (1 ring/press); Smooth (Laplacian ×5/press, rim-anchored); `Ctrl+Z`; shared
lasso refactor; clips-guard; headless engine tests.

**Explicitly out of v1:** add/subtract selection modifiers; click-individual-face
select; Taubin / volume-preserving smoothing; strength slider; selection through
active clips; mesh-check → selection (chunk 2); redo.

## 11. Integration anchors

- Engine: `_trim_history` `stl_clipper.py:249`; `recompute_all` `:424`;
  `trim_by_screen_polygon` `:452` (projection to reuse); `undo_trim` `:489`;
  `geometry_quality` `:501` (chunk 2).
- App: `self.plotter` (QtInteractor); `_refresh_display` (clear→enable_lightkit→add wall);
  trim lasso handlers + overlay (to refactor); `Ctrl+Z` shortcut in `__init__`;
  the existing `_btn_opaque_wall`/`_btn_show_mesh_edges` sidebar group (place new buttons nearby).
- Verified APIs (VTK 9.2.6 / pyvista 0.46.5): `Camera.direction`, `PolyData.cell_normals`,
  `cell_centers`, `cell_neighbors(ind, connections='points')`, `point_neighbors`,
  `extract_cells`, `_points_in_polygon` (existing module helper).
