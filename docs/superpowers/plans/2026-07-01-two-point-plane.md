# Two-Point Plane Cut Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A button that lets the user orbit freely then Shift+click two points anywhere in the viewer to set a precise cut plane (through both points, parallel to the view direction), which the existing ✂ Cut then applies.

**Architecture:** A pure module-level function `plane_from_two_points(A, B, view_dir)` computes `(origin, normal)`. The GUI adds a button, a `_TwoPointStyle` interactor (mirroring `_SelectLassoStyle`), a screen→focal-plane projection helper, and click capture that sets `self._current_plane_origin/normal` and draws the existing preview. No change to `cut_by_plane` / ✂ Cut / the add-plane widget.

**Tech Stack:** Python 3.9, pyvista 0.46.5, VTK 9.2.6, PyQt5, numpy. No new dependencies.

## Global Constraints

- No new dependencies.
- `plane_from_two_points` is a pure module-level function (no Qt, no engine state) — headless-testable.
- Plane math: `normal = normalize(cross(view_dir, B - A))`, `origin = (A + B) / 2`; return `None` when `‖cross‖ <= 1e-12` (A==B or A-B parallel to the view axis).
- The tool only SETS `self._current_plane_origin` / `self._current_plane_normal` and draws the existing preview (`_update_preview`); it must NOT modify `cut_by_plane`, `_on_cut`, the add-plane drag widget, or anything downstream.
- Shift detection via `QApplication.keyboardModifiers() & Qt.ShiftModifier`; pixel via `self.plotter.iren.get_event_position()`; interactor style installed/restored via the `iren.style =` setter (the proven `_SelectLassoStyle` pattern) — NOT raw `SetInteractorStyle`.
- Clicking/capture must not reset the camera.
- Run tests with `conda run -n mesh-prep pytest`.

---

### Task 1: Engine `plane_from_two_points`

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add a module-level function immediately before `class STLClipperEngine`.
- Test: `tests/test_selection.py` — extend the top import and append tests.

**Interfaces:**
- Consumes: numpy only.
- Produces: `plane_from_two_points(a, b, view_dir) -> tuple[np.ndarray, np.ndarray] | None` — `(origin, unit_normal)` or `None` when degenerate.

- [ ] **Step 1: Write the failing tests**

At the top of `tests/test_selection.py`, change the import line
`from mesh_prep.stl_clipper import STLClipperEngine`
to
`from mesh_prep.stl_clipper import STLClipperEngine, plane_from_two_points`

Then append:

```python
def test_plane_from_two_points_basic():
    origin, normal = plane_from_two_points((0, 0, 0), (1, 0, 0), (0, 0, 1))
    assert np.allclose(origin, (0.5, 0, 0))
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert abs(np.dot(normal, (0, 0, 1))) < 1e-9        # plane is parallel to the view axis
    assert abs(np.dot(normal, (1, 0, 0))) < 1e-9        # plane contains the A-B line


def test_plane_from_two_points_generic_orthogonality():
    A = np.array([1.0, 1.0, 0.0]); B = np.array([2.0, 3.0, 1.0]); view = np.array([0.3, -0.2, 1.0])
    origin, normal = plane_from_two_points(A, B, view)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert abs(np.dot(normal, view)) < 1e-9
    assert abs(np.dot(normal, B - A)) < 1e-9
    assert np.allclose(origin, (A + B) / 2)


def test_plane_from_two_points_identical_points_none():
    assert plane_from_two_points((1, 2, 3), (1, 2, 3), (0, 0, 1)) is None


def test_plane_from_two_points_colinear_with_view_none():
    assert plane_from_two_points((0, 0, 0), (0, 0, 2), (0, 0, 1)) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k plane_from_two_points -q`
Expected: FAIL — `ImportError: cannot import name 'plane_from_two_points'`.

- [ ] **Step 3: Implement the function**

Insert immediately before `class STLClipperEngine` in `mesh_prep/stl_clipper.py`:

```python
def plane_from_two_points(a, b, view_dir):
    """Cut plane through points a and b, parallel to view_dir (so it appears edge-on
    from the current camera and slices along the line of sight). Returns
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k plane_from_two_points -q`
Expected: PASS — 4 passed.

- [ ] **Step 5: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 45 passed (41 prior + 4 new).

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_selection.py
git commit -m "Add plane_from_two_points: cut plane through 2 points along the view axis"
```

---

### Task 2: GUI two-point plane tool

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add `_TwoPointStyle` class (after `_SelectLassoStyle`, ~`:1290`); add the button after `btn_add_plane` (`:1546`); add state init near `:1465`; add four methods on `STLClipperApp`; extend `_on_escape_selection` (`:3938`).

**Interfaces:**
- Consumes: `plane_from_two_points` (Task 1); existing `self._current_plane_origin/normal`, `self._update_preview()`, `self._cancel_clip_widgets()`, `self.plotter.camera.direction`/`.focal_point`, `self.plotter.renderer`, `self.plotter.iren`.
- Produces: GUI behavior only.

- [ ] **Step 1: Add the `_TwoPointStyle` interactor class**

Insert after the `_SelectLassoStyle` class (near `stl_clipper.py:1290`):

```python
class _TwoPointStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera style for the two-point plane tool. Plain drag orbits/zooms;
    Shift+click places a plane-defining point (camera suppressed for that click).
    Mirrors _SelectLassoStyle's subclass + AddObserver + conditional-forward pattern."""

    def __init__(self, app):
        self._app = app
        self.AddObserver("LeftButtonPressEvent", self._on_press)
        self.AddObserver("MouseMoveEvent", self._on_move)
        self.AddObserver("LeftButtonReleaseEvent", self._on_release)

    def _on_press(self, _obj=None, _evt=None):
        try:
            if bool(QApplication.keyboardModifiers() & Qt.ShiftModifier):
                self._app._twopt_click(self._app.plotter.iren.get_event_position())
                return                                   # suppress camera on the click
            self.OnLeftButtonDown()                      # plain press -> orbit
        except Exception:
            logger.exception("two-point press handler failed")

    def _on_move(self, _obj=None, _evt=None):
        try:
            self.OnMouseMove()
        except Exception:
            logger.exception("two-point move handler failed")

    def _on_release(self, _obj=None, _evt=None):
        try:
            self.OnLeftButtonUp()
        except Exception:
            logger.exception("two-point release handler failed")
```

- [ ] **Step 2: Add state init**

In `STLClipperApp.__init__`, next to `self._current_plane_origin = None` (`:1465`), add:

```python
        self._twopt_active = False
        self._twopt_first = None
```

- [ ] **Step 3: Add the button**

After the `self.btn_add_plane` block (`:1546`-`:1548`), add:

```python
        self.btn_two_point_plane = QPushButton("◪ 2-Point Plane")
        self.btn_two_point_plane.setToolTip(
            "Shift+click two points in the viewer to define a cut plane along your line of sight")
        self.btn_two_point_plane.clicked.connect(self._on_two_point_plane)
        panel.addWidget(self.btn_two_point_plane)
```

- [ ] **Step 4: Add the four handler methods**

Add on `STLClipperApp` (place them near `_update_preview`, ~`:3290`):

```python
    def _on_two_point_plane(self):
        """Enter two-point plane capture mode: orbit freely, then Shift+click two
        points to define a cut plane parallel to the view direction."""
        if self.engine.original_mesh is None:
            self.status.showMessage("Load an STL first.")
            return
        for btn in (getattr(self, "_btn_select", None), getattr(self, "_btn_trim", None)):
            if btn is not None and btn.isChecked():
                btn.setChecked(False)                    # exit select/trim so styles never stack
        self._cancel_clip_widgets()
        self._twopt_active = True
        self._twopt_first = None
        self.plotter.remove_actor("twopt_marker", render=False)
        iren = self.plotter.iren
        self._twopt_saved_style = iren.style or iren.interactor.GetInteractorStyle()
        self._twopt_style = _TwoPointStyle(self)
        iren.style = self._twopt_style
        self.status.showMessage("Shift+click two points to define the cut plane (Esc to cancel).")

    def _screen_to_focal_world(self, x, y):
        """Back-project display pixel (x, y) onto the camera focal plane -> world xyz.
        Depth is irrelevant here since the cut plane is parallel to the view axis."""
        ren = self.plotter.renderer
        fp = np.asarray(self.plotter.camera.focal_point, dtype=float)
        ren.SetWorldPoint(fp[0], fp[1], fp[2], 1.0)
        ren.WorldToDisplay()
        z = ren.GetDisplayPoint()[2]
        ren.SetDisplayPoint(float(x), float(y), z)
        ren.DisplayToWorld()
        w = np.asarray(ren.GetWorldPoint(), dtype=float)
        return w[:3] / w[3]

    def _twopt_click(self, pos):
        """Handle one Shift+click during two-point capture."""
        if not self._twopt_active:
            return
        world = self._screen_to_focal_world(pos[0], pos[1])
        if self._twopt_first is None:
            self._twopt_first = world
            b = np.asarray(self.engine.original_mesh.bounds, dtype=float)
            r = 0.01 * float(np.linalg.norm(b[1::2] - b[0::2]))
            self.plotter.add_mesh(pv.Sphere(radius=r, center=world),
                                  color="yellow", name="twopt_marker", reset_camera=False)
            self.status.showMessage("Point 1 set — Shift+click the second point.")
            return
        res = plane_from_two_points(self._twopt_first, world, self.plotter.camera.direction)
        if res is None:
            self.status.showMessage("Pick two distinct points.")
            return
        self._current_plane_origin, self._current_plane_normal = res
        self._end_two_point()
        self._update_preview()
        self.status.showMessage("Plane set — press ✂ Cut.")

    def _end_two_point(self):
        """Exit two-point capture: clear state + marker, restore the trackball style."""
        self._twopt_active = False
        self._twopt_first = None
        self.plotter.remove_actor("twopt_marker", render=False)
        if getattr(self, "_twopt_saved_style", None) is not None:
            self.plotter.iren.style = self._twopt_saved_style
```

- [ ] **Step 5: Extend the Escape handler**

Replace `_on_escape_selection` (`:3938`) with:

```python
    def _on_escape_selection(self):
        if getattr(self, "_twopt_active", False):
            self._end_two_point()
            self.status.showMessage("Two-point plane cancelled.")
            return
        if getattr(self, "_select_mode", False):
            self._btn_select.setChecked(False)   # exits select mode via _toggle_select_mode
        self._clear_selection()
```

- [ ] **Step 6: Verify the module compiles and imports**

Run: `conda run -n mesh-prep python -c "import mesh_prep.stl_clipper; print('IMPORT_OK')"`
Expected: `IMPORT_OK`.

- [ ] **Step 7: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 45 passed.

- [ ] **Step 8: Manual GUI smoke (record result)**

No GUI test harness (see `mesh-prep/CLAUDE.md`); verify by running:

```bash
conda run -n mesh-prep mesh-prep /Users/xiaz9n/openfoam/AO_Native_001.stl
```
Steps: click **◪ 2-Point Plane** → orbit/zoom with plain drag (works freely) → **Shift+click** one point (a yellow marker appears) → **Shift+click** a second point (the disc+arrow plane preview appears through both points, edge-on to your view). Press **✂ Cut** → the surface is cut there (cyan feature curve appears). **Esc** mid-capture cancels cleanly; re-clicking the button restarts. Two clicks at the same spot show "Pick two distinct points".

- [ ] **Step 9: Commit**

```bash
git add mesh_prep/stl_clipper.py
git commit -m "Add two-point plane tool: Shift+click two points to define a cut plane"
```

---

## Self-Review

**1. Spec coverage:**
- §3 `plane_from_two_points` → Task 1 Step 3; degenerate → None covered by two tests. ✓
- §4 button, `_TwoPointStyle`, `_on_two_point_plane`, `_twopt_click`, `_screen_to_focal_world`, `_end_two_point`, Escape extension → Task 2 Steps 1-5. ✓
- §4 sets `_current_plane_origin/normal` + `_update_preview`, no change to `_on_cut` → `_twopt_click` only sets those two + calls preview. ✓
- §6 edge cases: degenerate second click → "Pick two distinct points" (stays in capture); Esc/rebutton cleanup; no mesh no-op; select/trim exited first → all in Task 2. ✓
- §7 tests (basic, generic orthogonality, 2 degenerate) → Task 1 Step 1. ✓
- Constraints: no downstream change, `iren.style =` install, Shift via keyboardModifiers, no camera reset (`reset_camera=False`, no reset call) → honored. ✓

**2. Placeholder scan:** No TBD/TODO; every code step complete; every run step has an exact command + expected output. ✓

**3. Type consistency:** `plane_from_two_points(...) -> (origin, normal) | None` consumed in `_twopt_click` as `res is None` check then unpack. `_screen_to_focal_world(x, y) -> np.ndarray(3)` consumed as `world`. `_twopt_active`/`_twopt_first`/`_twopt_saved_style`/`_twopt_style` names consistent across init, handlers, and Escape. Marker actor name `"twopt_marker"` consistent across add/remove. ✓
