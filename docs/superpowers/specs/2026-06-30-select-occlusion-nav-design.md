# Select Mode: Visible-Only Selection + Rotate-While-Selecting — Design Spec

- **Date:** 2026-06-30
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperApp` — GUI only)
- **Builds on:** the active-selection feature (`select_cells_in_polygon`, `_apply_select`, `_begin_lasso`/`_end_lasso`, selection highlight) already merged on dev.

## 1. Goal

Two upgrades to the **Select faces** mode (only — Trim mode is untouched):
1. **Visible-only selection** — the lasso must not pick faces that are hidden behind nearer geometry, only the ones the user can actually see.
2. **Rotate while selecting + accumulate** — in Select mode, dragging from empty background rotates the view (normal trackball), dragging on the model lassoes; each lasso **adds** to the selection, so the user can orbit and build a selection across multiple views.

## 2. Requirements (decided in brainstorming)

| # | Decision |
|---|----------|
| Visibility | Selection excludes occluded faces (behind nearer geometry), via the live depth buffer, on top of the existing front-facing normal test |
| Gesture | **Left-press on the mesh → lasso**; **left-press on empty background → rotate** (trackball). Scroll zoom / right-drag unchanged. Decided per-drag by a pick at press. |
| Accumulate | Each lasso **unions** its visible faces into the current selection (was: replace). `Escape` still clears. |
| Scope | Select mode only. **Trim mode interaction is unchanged** (plain drag = lasso, one-shot). v1 has no lasso-*subtract*. |
| Layer | Both changes are **GUI-side**; the engine `select_cells_in_polygon` is unchanged (still the unit-tested polygon + front-facing core). |

## 3. Change 1 — visible-only selection (occlusion culling)

In `_apply_select` (`stl_clipper.py:3330`), after the engine returns candidate cell ids (in-polygon **and** front-facing), intersect them with the set of cells whose centroid is actually **visible** from the current camera:

- Build a `pv.PolyData` of the candidate cells' centroids (`original_mesh.cell_centers().points[candidate_ids]`).
- Run `vtkSelectVisiblePoints` with `SetRenderer(self.plotter.renderer)` and a small tolerance (~1e-3); it tests each point against the rendered depth buffer and outputs the visible subset.
- Map the surviving points back to candidate cell ids (preserve order; the i-th centroid ↔ candidate_ids[i]) and keep only those.

Verified (offscreen, sphere): `vtkSelectVisiblePoints` keeps front cells and drops occluded back cells (443 front / 14 silhouette of 528 back; the 14 are excluded anyway by the existing front-facing test). Needs a live render (z-buffer populated), so it lives in the GUI, not the engine.

New GUI helper `_visible_cells(candidate_ids) -> set[int]` encapsulates this; `_apply_select` calls it.

## 4. Change 2 — rotate-while-selecting + accumulate

### 4.1 Interaction
Select mode installs `vtkInteractorStyleTrackballCamera` (so rotate/zoom/pan all work normally) and registers **high-priority** observers (priority > 0, above the style's default 0) for `LeftButtonPressEvent`, `MouseMoveEvent`, `LeftButtonReleaseEvent` on the interactor:

- **Press:** `vtkPropPicker.Pick(x, y, 0, self.plotter.renderer)`. If it hits **any pickable prop** (i.e. the cursor is over geometry — `picker.GetActor()` is not None) → set `self._lasso_active = True`, start the lasso capture (reuse the overlay), and **abort the event** so the trackball does not start rotating. If it hits nothing (background) → leave `_lasso_active = False` and do **not** abort, so the trackball rotates. (The selection-highlight actor is created `pickable=False`, so it never blocks the wall underneath it.)
- **Move:** if `_lasso_active`, append the point + update overlay + abort; else do nothing (trackball rotates).
- **Release:** if `_lasso_active`, finish the stroke + abort, clear `_lasso_active`, and call `_apply_select(points)`; else do nothing.

The exact VTK event-suppression call (observer `AbortFlagOn` vs returning abort) is the key implementation detail and **must be verified during the plan** (and confirmed in the user's GUI run) — the rest of the design does not depend on which call is used, only that a geometry-press drag does not also rotate.

### 4.2 Accumulate
`_apply_select` changes its final step from `self._selection = set(ids)` to `self._selection |= self._visible_cells(ids)` (union). Highlight + button state refresh as before. `Escape` (`_clear_selection`) still empties the set; topology changes (load/trim/undo) still clear it.

### 4.3 Reuse vs new
Trim keeps using the existing `_begin_lasso`/`_end_lasso` (vtkInteractorStyleUser, plain-drag lasso). Select uses a new `_begin_select_lasso()` / `_end_select_lasso()` pair (trackball style + pick-gated observers) so the two interaction models stay isolated and Trim is provably unchanged. The overlay (`_start/_update/_end_lasso_overlay`) and `_apply_select` are shared.

## 5. Data flow (Select mode)
```
press on mesh  → pick hit → _lasso_active=True, start overlay, abort rotate
move           → append pt + overlay (abort)
release        → _apply_select(points):
                   engine.select_cells_in_polygon(poly, M, vp, view_dir, front_only=True)  → candidates
                   _visible_cells(candidates) via vtkSelectVisiblePoints (z-buffer)         → visible
                   self._selection |= visible  → highlight
press on empty → no abort → trackball rotates normally
Escape         → _clear_selection
```

## 6. Edge cases / error handling
- Polygon < 3 points (a click, not a drag, on the mesh) → no-op (no selection change), overlay cleared.
- No mesh / `_apply_select` candidates empty → selection unchanged.
- `vtkSelectVisiblePoints` returns nothing (e.g., fully occluded) → nothing added; selection unchanged.
- Mutual exclusion with Trim (the recently-fixed bug) must be preserved — entering Select still unchecks Trim and vice-versa; `_end_select_lasso` restores the saved style like `_end_lasso`.
- Clips present → Select/Grow/Smooth still disabled (unchanged guard).

## 7. Testing
- Engine selection (`select_cells_in_polygon`, grow, smooth) is **unchanged** — existing 15 tests still cover it; no new engine tests required.
- `_visible_cells` and the trackball/pick interaction are **render-dependent and GUI-only** → verified by the human in the running app (occlusion correctness; rotate-from-empty vs lasso-on-mesh; accumulate across views; Trim still works). The implementer verifies headlessly only: `py_compile`, `import`, and the full `pytest` suite still green.

## 8. Scope
**v1 (this spec):** visible-only selection (occlusion); Select-mode trackball nav with press-pick choosing lasso-vs-rotate; accumulate (union); Trim unchanged.

**Out of v1:** lasso-subtract / deselect modifier; applying occlusion to Trim; box/rectangle select; changing Trim's interaction; persisting selection across loads.

## 9. Integration anchors
- `_apply_select` `stl_clipper.py:3330` (add occlusion + union); `_toggle_select_mode` `:3315` (use new select-lasso begin/end); `_begin_lasso`/`_end_lasso` `:3171`/`:3188` (Trim keeps these; Select gets parallel `_begin_select_lasso`/`_end_select_lasso`); overlay `_start/_update/_end_lasso_overlay`; `_refresh_selection_highlight` `:3347`; `self.plotter.renderer`, `self.plotter.camera`, `self.plotter.iren.interactor`.
- Verified APIs (VTK 9.2.6): `vtkSelectVisiblePoints`, `vtkPropPicker`, `vtkInteractorStyleTrackballCamera`, `cell_centers`.
