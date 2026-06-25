# Freehand Region Trim — Design Spec

- **Date:** 2026-06-25
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)

## 1. Goal

Let the user draw an arbitrary freehand outline on the 3D viewport and permanently
delete the enclosed region of the surface mesh. This is a geometry-cleanup tool
(trim a stray branch, remove artifacts) used before meshing — distinct from the
existing plane/box clips, which define named inlet/outlet patches.

## 2. Requirements (decided during brainstorming)

| # | Decision | Value |
|---|----------|-------|
| Purpose | What the deleted region produces | **Permanent geometry trim** (destructive cleanup, not a patch, not a re-editable clip) |
| Depth | How deep the delete goes | **Through the whole model** (cookie-cutter: every cell whose screen projection is inside the outline, front + back + interior) |
| Shape | Outline type | **Arbitrary freehand lasso** (not rectangle) |
| Undo | Safety net | **Multi-level undo**, bounded history (~10 states) |

## 3. Chosen approach

**A — screen-space centroid point-in-polygon.** Project every cell centroid to
display space using the camera's view-projection matrix, then run a vectorized
even-odd point-in-polygon test against the drawn outline; delete inside cells by
keeping the complement. "Through-model" falls out for free because projection
ignores depth.

Rejected:
- **B — `vtkImplicitSelectionLoop` + `vtkExtractPolyDataGeometry`:** exact at cell
  boundaries but needs screen→world unprojection + loop-normal/perspective handling
  and is hard to unit-test headlessly. Not worth it for a centroid-granularity trim.
- **C — pyvista `enable_cell_picking(through=True)`:** nearly free but rectangle
  only; fails the "arbitrary shape" requirement. (Viable 1-hour fallback if A stalls.)

## 4. API verification (installed: pyvista 0.46.5 / VTK 9.2.6)

Confirmed present and usable:
- `vtkCamera.GetCompositeProjectionTransformMatrix(aspect, nearz, farz)` — world→clip matrix (handles parallel **and** perspective; viewport uses **parallel** projection, set at `stl_clipper.py:961`).
- `PolyData.cell_centers()`, `PolyData.extract_cells(ids)` — centroids + complement extraction.
- `vtkInteractorStyleDrawPolygon` class exists, **but `GetPolygonPoints()` is NOT wrapped in this Python build** (C++ return type `std::vector<vtkVector2i>` doesn't wrap). → We do **not** rely on that style's getter; instead we capture the lasso via interactor observers (`GetEventPosition()` on mouse move), which are fully wrapped.

DPR note: `_patch_dpr_picking()` (`stl_clipper.py:2089`) already normalizes the
interactor to **logical-pixel** space. Observer `GetEventPosition()` therefore returns
logical pixels; the projection viewport must use the logical render-window size so the
outline and the projected centroids live in the same coordinate space. The existing
plane/box pickers already operate correctly in this space — follow the same convention.

## 5. Architecture

### 5.1 Engine — `STLClipperEngine` (headless, no Qt/render window → unit-testable)

New state (in `__init__`, near `stl_clipper.py:220`):
```python
self._trim_history: collections.deque = collections.deque(maxlen=10)
```

New method — the pure core (this is what tests target):
```python
def trim_by_screen_polygon(self, polygon_xy, view_matrix, viewport):
    """Delete cells of original_mesh whose centroid projects inside polygon_xy.

    polygon_xy : list[(x, y)] display-space points (logical pixels), the freehand outline
    view_matrix: 4x4 world->clip (camera.GetCompositeProjectionTransformMatrix)
    viewport   : (width, height) logical pixels
    Returns the new self._wall_mesh, or None on no-op.
    """
```
Algorithm:
1. If `len(polygon_xy) < 3` → return None (no-op, no history push).
2. `centroids = original_mesh.cell_centers().points` (N x 3).
3. Homogeneous transform by `view_matrix` → divide by w → NDC; map NDC → display (x,y) using `viewport`. **Origin consistency:** VTK `GetEventPosition()` uses a bottom-left origin, so the NDC→display map must too (`y = (ndc_y*0.5+0.5)*height`, no y-flip) or the selection comes out vertically mirrored.
4. Vectorized even-odd point-in-polygon → `inside` boolean mask (N,).
5. Guards: `inside.sum() == 0` → return None (no-op); `inside.all()` → raise/return sentinel "would delete entire mesh" (refuse).
6. `self._trim_history.append(self.original_mesh.copy())`.
7. `self.original_mesh = self.original_mesh.extract_cells(np.where(~inside)[0]).extract_surface()`.
8. `return self.recompute_all()`  # clips re-apply on the trimmed base.

New method:
```python
def undo_trim(self) -> bool:
    if not self._trim_history:
        return False
    self.original_mesh = self._trim_history.pop()
    self.recompute_all()
    return True
```

Why this is testable: `trim_by_screen_polygon` takes the polygon, matrix, and
viewport as plain inputs — a pytest can build a known grid/cube mesh + an
orthographic matrix + a polygon over a known sub-region and assert exactly which
cells are removed, with no GUI.

### 5.2 GUI — `STLClipperApp`

- New checkable button **"✂ Trim region"** in the sidebar near the clip controls.
- **Enter trim mode:** stash current interactor style; set a non-rotating style
  (`vtkInteractorStyleUser`) on the VTK interactor; add observers:
  - `LeftButtonPressEvent` → clear point buffer, begin stroke.
  - `MouseMoveEvent` (button down) → append `iren.GetEventPosition()`.
  - `LeftButtonReleaseEvent` → finish stroke → trim.
  - (Optional) live `vtkActor2D` polyline overlay of the in-progress outline.
- **On stroke release:**
  1. Build `view_matrix` from `self.plotter.camera.GetCompositeProjectionTransformMatrix(aspect, -1, 1)` and read logical render-window size for `viewport`.
  2. `result = self.engine.trim_by_screen_polygon(points, view_matrix, viewport)`.
  3. On success → `self._refresh_display()` (`stl_clipper.py:3083`); on refusal → `self.status.showMessage(...)`.
  - Stay in trim mode for multiple strokes.
- **Exit trim mode / `Esc`:** restore the stashed interactor style.
- **`Ctrl+Z`** (`QShortcut`) → `self.engine.undo_trim()` then `_refresh_display()`.

Implementation-time detail to confirm (not blocking): exact handle for the VTK
interactor + `SetInteractorStyle` under pyvistaqt 0.11.3 (`self.plotter.iren` vs
`self.plotter.interactor`), and that `GetEventPosition()` reads logical pixels after
the DPR patch.

### 5.3 Data flow
```
lasso (interactor observers) -> display pts (logical px)
  -> camera.GetCompositeProjectionTransformMatrix + logical viewport
  -> engine.trim_by_screen_polygon(pts, M, vp)
       -> project centroids -> even-odd point-in-polygon -> keep complement
       -> push original_mesh to history -> original_mesh = trimmed -> recompute_all()
  -> _refresh_display()
Ctrl+Z -> engine.undo_trim() -> recompute_all() -> _refresh_display()
```

## 6. Key design decision

Trim mutates **`self.original_mesh`** (the base `recompute_all()` rebuilds from,
`stl_clipper.py:402`), not the visible `_wall_mesh`. Consequence: existing inlet/outlet
**clips survive** and re-apply on the trimmed geometry. (Alternative — trim the visible
mesh and clear clips, like the repair path at `stl_clipper.py:487` — was rejected
because it discards the caps.) A clip whose plane no longer intersects trimmed geometry
is handled by the existing failed-clip path (recently hardened in SC-01).

## 7. Edge cases / error handling

- Polygon < 3 points or zero cells selected → silent no-op, no history entry.
- All cells selected → refuse with a status-bar warning ("would delete entire mesh").
- Parallel and perspective both handled by the composite projection matrix.
- History bounded at 10 — oldest snapshot dropped silently; each entry is a full
  surface-mesh copy (memory-bounded).
- Trim with no mesh loaded → button disabled / no-op.

## 8. Testing

First real test suite for the package: `tests/test_trim.py` (pytest), all headless:
- Known grid/cube mesh + orthographic `view_matrix` + a polygon over a known
  sub-region → assert exact removed-cell count and that the complement is kept.
- `undo_trim()` restores the prior mesh and cell count.
- Guards: empty polygon → no-op; full-cover polygon → refuse; degenerate (<3 pts) → no-op.

## 9. Scope

**v1 (this spec):** freehand delete-inside, through-model, multi-level undo, the
"Trim region" toggle, `Ctrl+Z`, and headless unit tests for the engine method.

**Explicitly out of v1:** redo; keep-inside / delete-outside toggle; surface-only
(visible-cell) mode; rectangle mode; persisting trims to `clip_planes.json`; live
overlay polish beyond a basic outline.
