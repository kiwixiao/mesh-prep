# Two-Point Plane Cut Tool — Design Spec

- **Date:** 2026-07-01
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (module-level helper + `STLClipperApp`)
- **Relation:** An alternative, precise way to define the cut plane the ✂ Cut button already consumes (sub-feature A). Independent of B/C.

## 1. Goal

Add a **precise** way to place a cut plane: click a button, orbit/zoom freely, then **Shift+click two points anywhere in the viewer**. The two screen points define a plane that passes through them and is **parallel to the current view direction** — from the current eye it looks edge-on (the line you drew) and slices straight back along the line of sight. This sets the same `_current_plane_origin` / `_current_plane_normal` the drag-widget sets, so the existing **✂ Cut** applies unchanged. It gives exact control the normal-arrow drag lacks.

## 2. Decisions (from brainstorming)

| # | Decision |
|---|----------|
| Points anywhere | The two points are NOT required to lie on the surface. Each Shift+click is back-projected to the camera **focal plane** (no surface pick). Depth is irrelevant because the cut plane is parallel to the view axis. |
| Plane math | `normal = normalize(cross(view_dir, B − A))`, `origin = (A + B) / 2`. |
| After 2 clicks | **Preview only** (reuse the existing disc+arrow `_update_preview`), then the user presses the existing ✂ Cut. Nothing can nudge the plane afterward. |
| Interaction | Button enters capture mode; plain drag orbits/zooms; Shift+click places a point; a marker dot shows the first point; the second click sets the plane and exits capture. Re-click the button to redo; **Esc** cancels. |
| No change downstream | `cut_by_plane`, ✂ Cut, the add-plane widget, feature curves, flood-select, and the object tree are untouched. |

## 3. Engine interface (headless, unit-tested)

A module-level pure function in `stl_clipper.py` (no Qt, importable in tests):

```python
def plane_from_two_points(a, b, view_dir):
    """Cut plane through points a and b, parallel to view_dir. Returns
    (origin, normal) with a unit normal and origin at the midpoint, or None if the
    inputs are degenerate (a == b, or b - a parallel to view_dir → zero cross)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(view_dir, dtype=float)
    normal = np.cross(d, b - a)
    n = float(np.linalg.norm(normal))
    if n <= 1e-12:
        return None
    return (a + b) / 2.0, normal / n
```

## 4. App (GUI) layer

**Button** — after `self.btn_add_plane` (`stl_clipper.py:1546`):
```python
self.btn_two_point_plane = QPushButton("◪ 2-Point Plane")
self.btn_two_point_plane.setToolTip(
    "Shift+click two points in the viewer to define a cut plane along your line of sight")
self.btn_two_point_plane.clicked.connect(self._on_two_point_plane)
panel.addWidget(self.btn_two_point_plane)
```

**State** (init in `__init__`): `self._twopt_active = False`, `self._twopt_first = None` (world point A), `self._twopt_marker = None` (marker actor name).

**Enter capture** — `_on_two_point_plane()`:
- If no mesh, status hint and return.
- End any active Select/Trim lasso and uncheck those mode buttons (mirror the existing mode-exclusion so interactor styles never stack); clear any plane widget/box preview via the existing cleanup.
- `self._twopt_active = True`, `self._twopt_first = None`.
- Install a `_TwoPointStyle` interactor style via the same `iren.style =` setter used by `_begin_select_lasso`.
- Status: `"Shift+click two points to define the cut plane (Esc to cancel)."`

**Interactor style** — `_TwoPointStyle(vtk.vtkInteractorStyleTrackballCamera)`, mirroring `_SelectLassoStyle`:
- `LeftButtonPressEvent`: if `QApplication.keyboardModifiers() & Qt.ShiftModifier` → call `self._app._twopt_click(iren.get_event_position())` and **do not** forward (no camera move); else `OnLeftButtonDown()` (normal orbit).
- `MouseMoveEvent` / `LeftButtonReleaseEvent`: forward normally (`OnMouseMove` / `OnLeftButtonUp`) — free orbit/zoom.

**Point capture** — `_twopt_click(pos)`:
- `world = self._screen_to_focal_world(pos[0], pos[1])`.
- If `self._twopt_first is None`: store it, add a small sphere marker actor `"twopt_marker"` at `world` (`reset_camera=False`), status `"Point 1 set — Shift+click the second point."`
- Else: `res = plane_from_two_points(self._twopt_first, world, self.plotter.camera.direction)`. If `None`: status `"Pick two distinct points."` and keep waiting for a valid second click. Otherwise set `self._current_plane_origin, self._current_plane_normal = res`; remove the marker; `self._update_preview()` (existing disc+arrow); exit capture (`_end_two_point`); status `"Plane set — press ✂ Cut."`

**Screen → world (focal plane)** — `_screen_to_focal_world(x, y)`:
```python
ren = self.plotter.renderer
fp = np.asarray(self.plotter.camera.focal_point, dtype=float)
ren.SetWorldPoint(fp[0], fp[1], fp[2], 1.0)
ren.WorldToDisplay()
z = ren.GetDisplayPoint()[2]
ren.SetDisplayPoint(float(x), float(y), z)
ren.DisplayToWorld()
w = np.asarray(ren.GetWorldPoint(), dtype=float)
return w[:3] / w[3]
```

**Exit / cancel** — `_end_two_point()`: `self._twopt_active = False`, `self._twopt_first = None`, remove `"twopt_marker"`, restore the normal trackball style (same mechanism as `_end_select_lasso`). Extend the existing Escape handler (`_on_escape_selection`, `:3938`) to call `_end_two_point()` first when `self._twopt_active`. Re-clicking the button calls `_on_two_point_plane` again (restarts capture).

## 5. Data flow

```
click "◪ 2-Point Plane" → _on_two_point_plane → install _TwoPointStyle
orbit freely (plain drag)
Shift+click #1 → _twopt_click → _screen_to_focal_world → store A, drop marker
Shift+click #2 → _twopt_click → B = _screen_to_focal_world
    → plane_from_two_points(A, B, camera.direction) → (origin, normal)
    → set _current_plane_origin/normal → _update_preview → _end_two_point
press ✂ Cut (existing) → engine.cut_by_plane(origin, normal)
```

## 6. Edge cases

- Second click too close to the first / A–B colinear with the view axis → `plane_from_two_points` returns `None` → status prompt, stay in capture for a valid second click.
- Esc mid-capture, or re-click the button → clean up marker + restore trackball.
- No mesh → button no-op with a status hint.
- Entering capture while in Select/Trim mode → those are exited first, so only one interactor style is ever active.

## 7. Testing (headless pytest, `tests/test_selection.py`)

`plane_from_two_points`:
- A=(0,0,0), B=(1,0,0), view=(0,0,1) → normal is unit length, `dot(normal, view) ≈ 0`, `dot(normal, B−A) ≈ 0`, origin == (0.5, 0, 0).
- A=(1,2,3), B=(1,2,3) (equal) → None.
- A=(0,0,0), B=(0,0,2), view=(0,0,1) (B−A ∥ view) → None.
- A generic non-axis case → normal orthogonal to both `view_dir` and `B−A` (dots ≈ 0) and unit length.

(The pixel→world projection and interactor style are GUI-bound; verified by import + manual smoke, not unit tests.)

## 8. Scope

Adds: module-level `plane_from_two_points`; the `◪ 2-Point Plane` button; `_TwoPointStyle`; `_on_two_point_plane` / `_twopt_click` / `_screen_to_focal_world` / `_end_two_point`; Escape-handler extension. **Out of scope:** any change to `cut_by_plane` / ✂ Cut / the add-plane widget / the object tree; a draggable widget after the two clicks (explicitly rejected — precision comes from the clicks).

## 9. Integration anchors (verified 2026-07-01)

- `_current_plane_origin` / `_current_plane_normal` init `:1465`; consumed by `_on_cut` `:3531` (`cut_by_plane(origin, normal)`) and `_update_preview` `:3290`.
- `btn_add_plane` `:1546` (insert the new button after it); `btn_cut` `:1564`.
- Interactor-style install/restore pattern: `_SelectLassoStyle` (`:1236`-ish), `_begin_select_lasso` / `_end_select_lasso` (`iren.style =` setter); Shift via `QApplication.keyboardModifiers() & Qt.ShiftModifier`; pixel via `self.plotter.iren.get_event_position()`.
- `self.plotter.camera.direction` and `.focal_point` (pyvista Camera; `.direction` already used in `_apply_select`).
- Escape shortcut `_select_esc_shortcut` → `_on_escape_selection` `:3938` (extend for two-point cancel).
- VTK renderer projection: `vtkRenderer.SetWorldPoint/WorldToDisplay/GetDisplayPoint` and `SetDisplayPoint/DisplayToWorld/GetWorldPoint` (standard; plan verifies on the installed VTK 9.2.6).
