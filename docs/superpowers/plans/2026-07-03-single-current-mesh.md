# Single Current Mesh — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the engine's two-mesh model (`original_mesh` + `_wall_mesh` preview) with one compounding `current_mesh` whose named boundary patches are per-face `patch_id` labels, so every panel reflects the current surface and edits stack.

**Architecture:** `STLClipperEngine` holds one `current_mesh: pv.PolyData` with a cell-data array `"patch_id"` (0 = wall, 1..N = named) and a `patch_names: dict[int,str]`. Every operation snapshots then mutates `current_mesh`; detection/tree/display/export all read it; undo restores a snapshot. Migration is incremental — each task keeps the existing 68 tests green (adapting only the tests whose API a task changes) and never breaks OpenFOAM export.

**Tech Stack:** Python 3.9, pyvista/VTK, PyQt5 (GUI only), pytest. Conda env `mesh-prep`.

## Global Constraints

- Run tests with: `conda run -n mesh-prep python -m pytest tests/ -q` (run in background per repo convention).
- Engine (`STLClipperEngine`) and its helpers must stay Qt-free (no `PyQt5`/`pyvistaqt` in method bodies). Only the module-level imports touch Qt.
- Do not change `openfoam_case.py`, the render layer, the Docker workflow, or `create_case.py`'s CLI contract.
- Patch label array name is the module constant `PATCH_ID = "patch_id"`. `patch_id` 0 always means wall.
- No auto-commit beyond the per-task commit steps below; never push. Work stays on branch `dev`.
- Every op returns the new mesh (or `None` on no-op) and pushes exactly one undo snapshot when it changes geometry.

---

### Task 1: Spike — verify `patch_id` survives every filter (build `_carry_labels`)

The whole design assumes cell data rides through the VTK filters we use. `vtkPolyDataPlaneClipper` (in `clip_with_plane`) is a specialized clipper that may drop cell data; `vtkClipPolyData`, `clean`, `triangulate`, `merge`, `remove_cells` generally pass it. This task proves it empirically and produces the fallback helper the rest of the plan depends on.

**Files:**
- Create: `tests/test_patch_labels.py`
- Modify: `mesh_prep/stl_clipper.py` (add `PATCH_ID` constant near line 60; add `_carry_labels` static method to `STLClipperEngine`)

**Interfaces:**
- Produces: module constant `PATCH_ID = "patch_id"`; `STLClipperEngine._carry_labels(result: pv.PolyData, source: pv.PolyData, default_id: int = 0) -> pv.PolyData` — guarantees `result.cell_data[PATCH_ID]` exists: if the filter preserved it, returns as-is; otherwise re-maps each result cell to its nearest source-cell label via a KD-tree on cell centers, filling `default_id` where source had none.

- [ ] **Step 1: Write the probe test**

```python
import numpy as np
import pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def _labeled_sphere():
    m = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    ids = np.zeros(m.n_cells, dtype=np.int64)
    ids[: m.n_cells // 3] = 1                      # label a third as patch 1
    m.cell_data[PATCH_ID] = ids
    return m


def test_carry_labels_preserves_existing():
    m = _labeled_sphere()
    same = STLClipperEngine._carry_labels(m.copy(), m)
    assert PATCH_ID in same.cell_data
    assert set(np.unique(same.cell_data[PATCH_ID])) == {0, 1}


def test_carry_labels_remaps_when_filter_drops_them():
    m = _labeled_sphere()
    stripped = m.copy()
    del stripped.cell_data[PATCH_ID]               # simulate a filter that dropped labels
    fixed = STLClipperEngine._carry_labels(stripped, m)
    assert PATCH_ID in fixed.cell_data
    # nearest-cell remap recovers roughly the original split (identical topology here)
    assert set(np.unique(fixed.cell_data[PATCH_ID])) == {0, 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_patch_labels.py -q`
Expected: FAIL — `cannot import name 'PATCH_ID'` / `_carry_labels` undefined.

- [ ] **Step 3: Add the constant and helper**

In `mesh_prep/stl_clipper.py`, near the other module constants (~line 60):

```python
PATCH_ID = "patch_id"   # cell-data array on current_mesh; 0 = wall, 1..N = named patch
```

Add to `STLClipperEngine` (near the other `@staticmethod` helpers):

```python
@staticmethod
def _carry_labels(result: pv.PolyData, source: pv.PolyData, default_id: int = 0) -> pv.PolyData:
    """Ensure result carries a PATCH_ID cell array. If a filter preserved it,
    keep it; otherwise remap each result cell to the nearest source cell's label."""
    if result is None or result.n_cells == 0:
        return result
    if PATCH_ID in result.cell_data and len(result.cell_data[PATCH_ID]) == result.n_cells:
        return result
    if source is None or PATCH_ID not in source.cell_data or source.n_cells == 0:
        result.cell_data[PATCH_ID] = np.full(result.n_cells, default_id, dtype=np.int64)
        return result
    src_centers = source.cell_centers().points
    src_labels = np.asarray(source.cell_data[PATCH_ID])
    res_centers = result.cell_centers().points
    from scipy.spatial import cKDTree          # scipy ships with pyvista's deps
    _, idx = cKDTree(src_centers).query(res_centers)
    result.cell_data[PATCH_ID] = src_labels[idx].astype(np.int64)
    return result
```

If `scipy` is not importable in the env, replace the KD-tree block with `pv.PolyData(res_centers).find_closest_point`-style nearest lookup on `source.cell_centers()` — verify import first with `conda run -n mesh-prep python -c "import scipy; print(scipy.__version__)"`.

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n mesh-prep python -m pytest tests/test_patch_labels.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the real-filter propagation probes**

Append to `tests/test_patch_labels.py` — these document actual VTK behavior and lock it:

```python
def test_labels_survive_plane_clip():
    m = _labeled_sphere()
    clipped = STLClipperEngine.clip_with_plane(m, (0, 0, 0), (0, 0, 1))
    clipped = STLClipperEngine._carry_labels(clipped, m)
    assert PATCH_ID in clipped.cell_data
    assert len(clipped.cell_data[PATCH_ID]) == clipped.n_cells


def test_labels_survive_merge():
    m = _labeled_sphere()
    cap = pv.Disc(center=(0, 0, 1), inner=0.0, outer=0.3).triangulate()
    cap.cell_data[PATCH_ID] = np.full(cap.n_cells, 2, dtype=np.int64)
    merged = m.merge(cap, merge_points=False)
    assert PATCH_ID in merged.cell_data
    assert set(np.unique(merged.cell_data[PATCH_ID])) == {0, 1, 2}
```

- [ ] **Step 6: Run and record which filters needed the fallback**

Run: `conda run -n mesh-prep python -m pytest tests/test_patch_labels.py -q`
Expected: PASS (4 tests). If `test_labels_survive_plane_clip` only passes because of `_carry_labels`, that confirms `vtkPolyDataPlaneClipper` drops labels — later tasks MUST wrap `clip_with_plane` in `_carry_labels`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_patch_labels.py mesh_prep/stl_clipper.py
git commit -m "Add patch_id label constant and _carry_labels propagation helper"
```

---

### Task 2: `current_mesh` + `patch_names` set on load

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `STLClipperEngine.__init__` (288–301), `load_stl` (303–313)
- Test: `tests/test_current_mesh.py` (create)

**Interfaces:**
- Produces: `self.current_mesh: Optional[pv.PolyData]`, `self.patch_names: dict[int, str]`, `self._next_patch_id: int`; `load_stl` sets `current_mesh` with `cell_data[PATCH_ID]` all 0 and `patch_names = {}`.
- Consumes: `PATCH_ID`, `_carry_labels` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def test_load_stl_initializes_current_mesh_labels(tmp_path):
    p = tmp_path / "s.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(p))
    eng = STLClipperEngine()
    eng.load_stl(str(p))
    assert eng.current_mesh is not None
    assert PATCH_ID in eng.current_mesh.cell_data
    assert np.all(eng.current_mesh.cell_data[PATCH_ID] == 0)   # all wall initially
    assert eng.patch_names == {}
    assert eng._next_patch_id == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py -q`
Expected: FAIL — `AttributeError: 'STLClipperEngine' object has no attribute 'current_mesh'`.

- [ ] **Step 3: Add fields and set them on load**

In `__init__`, after `self.original_mesh = None` add:

```python
self.current_mesh: Optional[pv.PolyData] = None
self.patch_names: dict[int, str] = {}
self._next_patch_id: int = 1
```

In `load_stl`, after `self.original_mesh = mesh` and before returning, add:

```python
current = mesh.triangulate()
current.cell_data[PATCH_ID] = np.zeros(current.n_cells, dtype=np.int64)
self.current_mesh = current
self.patch_names = {}
self._next_patch_id = 1
```

(Keep the existing `original_mesh`/`_wall_mesh` assignments for now — later tasks remove them.)

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `conda run -n mesh-prep python -m pytest tests/ -q`
Expected: PASS (existing 68 + new tests).

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_current_mesh.py
git commit -m "Introduce current_mesh + patch_names, initialized on load_stl"
```

---

### Task 3: Detection + object tree read `current_mesh`

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `detect_open_profiles` (819), `detect_nonmanifold_edges` (829), `detect_pieces` (839); GUI `_refresh_object_tree` (4148) mesh source
- Test: `tests/test_current_mesh.py` (extend)

**Interfaces:**
- Consumes: `current_mesh` (Task 2).
- Produces: `detect_*` operate on `self.current_mesh`; a new `named_patches() -> list[tuple[int, str, int]]` returning `(patch_id, name, face_count)` for the tree's "Named patches" section.

- [ ] **Step 1: Write the failing test**

```python
def test_detection_follows_current_mesh(tmp_path):
    eng = STLClipperEngine()
    cyl = pv.Cylinder(radius=1, height=4, resolution=40, capping=True).triangulate()
    eng.current_mesh = cyl
    eng.current_mesh.cell_data[PATCH_ID] = np.zeros(cyl.n_cells, dtype=np.int64)
    eng.original_mesh = None                      # prove detection no longer needs it
    assert eng.detect_pieces()                    # one component, no crash
    assert eng.detect_open_profiles() == []       # capped cylinder is watertight
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_detection_follows_current_mesh -q`
Expected: FAIL — `detect_pieces` reads `self.original_mesh` which is `None` → returns `[]`, assert fails.

- [ ] **Step 3: Repoint detection**

In `detect_open_profiles`, `detect_nonmanifold_edges`, `detect_pieces`, change the first line `m = self.original_mesh` to:

```python
m = self.current_mesh
```

Add the named-patch accessor:

```python
def named_patches(self):
    """(patch_id, name, face_count) for each named patch present on current_mesh."""
    if self.current_mesh is None or PATCH_ID not in self.current_mesh.cell_data:
        return []
    ids = np.asarray(self.current_mesh.cell_data[PATCH_ID])
    out = []
    for pid, name in sorted(self.patch_names.items()):
        out.append((pid, name, int(np.count_nonzero(ids == pid))))
    return out
```

In GUI `_refresh_object_tree` (line 4152) change `mesh = self.engine.original_mesh` to `mesh = self.engine.current_mesh`, and replace the "Named patches" section's source `self._tree_patches = self.engine.filled_patches` + its loop with `self.engine.named_patches()` tuples (item data `("patch", patch_id)`).

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n mesh-prep python -m pytest tests/ -q`
Expected: PASS. (The existing `detect_*` tests in `test_selection.py` set `original_mesh`; update them to also set `current_mesh` — or better, have those tests set `current_mesh`. Adjust the `_sphere_engine`/inline setups in `test_selection.py` to assign `eng.current_mesh` alongside `eng.original_mesh`.)

- [ ] **Step 5: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/
git commit -m "Detection and object tree read current_mesh; add named_patches()"
```

---

### Task 4: Viewport + counts read `current_mesh`

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `get_wall_mesh` (1016), `_refresh_display` wall source (~4430), face-count label
- Test: `tests/test_current_mesh.py` (extend)

**Interfaces:**
- Consumes: `current_mesh`.
- Produces: `get_wall_mesh()` returns `self.current_mesh` (kept as the display accessor name to minimize call-site churn).

- [ ] **Step 1: Write the failing test**

```python
def test_get_wall_mesh_returns_current(tmp_path):
    p = tmp_path / "s.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(p))
    eng = STLClipperEngine()
    eng.load_stl(str(p))
    assert eng.get_wall_mesh() is eng.current_mesh
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_get_wall_mesh_returns_current -q`
Expected: FAIL — returns `_wall_mesh`, not `current_mesh`.

- [ ] **Step 3: Repoint the accessor**

Change `get_wall_mesh` body to `return self.current_mesh`. In `_refresh_display`, the line `wall = self.engine.get_wall_mesh()` now yields `current_mesh` — no further change needed there. Color the wall actor by `PATCH_ID` (scalars) or keep flat wall color; for this task keep the existing flat wall render (coloring-by-patch is Task 7's concern).

- [ ] **Step 4: Run full suite**

Run: `conda run -n mesh-prep python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_current_mesh.py
git commit -m "Viewport reads current_mesh via get_wall_mesh"
```

---

### Task 5: `fill_profile` labels + merges into `current_mesh`

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `fill_profile` (855), add `_new_patch_id`
- Test: `tests/test_current_mesh.py` (extend); adapt `test_selection.py` fill tests

**Interfaces:**
- Consumes: `current_mesh`, `PATCH_ID`, `_carry_labels`.
- Produces: `_new_patch_id(name: str) -> int` (allocates `_next_patch_id`, records `patch_names`); `fill_profile(profile_edges, name) -> Optional[pv.PolyData]` now triangulates the hole, labels the cap with a new id, merges into `current_mesh`, snapshots first, returns `current_mesh`.

- [ ] **Step 1: Write the failing test**

```python
def _open_sphere(eng):
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    eng.current_mesh = sph
    eng.current_mesh.cell_data[PATCH_ID] = np.zeros(sph.n_cells, dtype=np.int64)


def test_fill_profile_labels_and_merges(tmp_path):
    eng = STLClipperEngine()
    _open_sphere(eng)
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = int(np.where(eng.current_mesh.cell_centers().points[:, 2] > 0.3)[0][0])
    eng.delete_cells(eng.flood_select(top))
    prof = eng.detect_open_profiles()[0]
    out = eng.fill_profile(prof, "inlet")
    assert out is eng.current_mesh
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert 1 in np.unique(ids)                       # the cap is labeled
    assert eng.patch_names[1] == "inlet"
```

(Note: `cut_by_plane`/`delete_cells`/`flood_select` are repointed to `current_mesh` as part of this task's prerequisites — see Step 3.)

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_fill_profile_labels_and_merges -q`
Expected: FAIL — old `fill_profile` appends to `filled_patches`, does not touch `current_mesh`.

- [ ] **Step 3: Repoint the mutating ops to `current_mesh`, then rewrite fill**

First repoint `cut_by_plane` (945), `delete_cells` (930), `trim_by_screen_polygon` (507), `smooth_cells` (642), and `flood_select` to read/write `self.current_mesh` instead of `self.original_mesh`, wrapping each VTK result in `self._carry_labels(result, self.current_mesh)` before assigning. (These are mechanical: replace `self.original_mesh` with `self.current_mesh` in those method bodies; after producing the new mesh call `new = self._carry_labels(new, self.current_mesh)`.)

Add and rewrite:

```python
def _new_patch_id(self, name: str) -> int:
    pid = self._next_patch_id
    self.patch_names[pid] = name
    self._next_patch_id += 1
    return pid

def fill_profile(self, profile_edges, name):
    """Triangulate an open profile into a named cap and merge it into current_mesh."""
    if self.current_mesh is None or profile_edges is None or profile_edges.n_cells == 0:
        return None
    strip = vtk.vtkStripper(); strip.SetInputData(profile_edges); strip.Update()
    tri = vtk.vtkContourTriangulator(); tri.SetInputData(strip.GetOutput()); tri.Update()
    cap = pv.wrap(tri.GetOutput())
    if cap is None or cap.n_cells == 0:
        cap = pv.PolyData(profile_edges.points).delaunay_2d()
    if cap is None or cap.n_cells == 0:
        return None
    cap = cap.triangulate()
    self._push_history()
    pid = self._new_patch_id(name)
    cap.cell_data[PATCH_ID] = np.full(cap.n_cells, pid, dtype=np.int64)
    self.current_mesh = self.current_mesh.merge(cap, merge_points=True)
    self.current_mesh = self._carry_labels(self.current_mesh, self.current_mesh)
    return self.current_mesh
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_fill_profile_labels_and_merges -q`
Expected: PASS.

- [ ] **Step 5: Adapt existing fill/selection tests and run full suite**

Update `test_selection.py` fill/cut/delete tests to assert against `current_mesh` and `patch_names` instead of `filled_patches`. Run: `conda run -n mesh-prep python -m pytest tests/ -q`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/
git commit -m "fill_profile and mutating ops operate on labeled current_mesh"
```

---

### Task 6: `clip_and_name` (trim + cap + label + merge)

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add `clip_and_name`; keep `clip_with_plane`/`clip_with_box`/`_generate_cap` as helpers; GUI `_on_confirm_clip` (3795) calls `clip_and_name`
- Test: `tests/test_current_mesh.py` (extend)

**Interfaces:**
- Consumes: `clip_with_plane` (379), `clip_with_box` (403), `_generate_cap` (337), `_new_patch_id`, `_carry_labels`.
- Produces: `clip_and_name(name: str, origin, normal, box_planes_data=None) -> Optional[pv.PolyData]` — snapshots, trims `current_mesh` (plane or box), generates the cap at the cut, labels cap faces with a new `patch_id`, merges into `current_mesh`, returns it (or `None` on no-op/empty).

- [ ] **Step 1: Write the failing test**

```python
def test_clip_and_name_trims_and_labels():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    eng.current_mesh = sph
    eng.current_mesh.cell_data[PATCH_ID] = np.zeros(sph.n_cells, dtype=np.int64)
    n0 = sph.n_cells
    out = eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))   # keep +z side, cap the cut
    assert out is eng.current_mesh
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert eng.patch_names[1] == "inlet"
    assert 1 in np.unique(ids)                                # cap labeled inlet
    assert 0 in np.unique(ids)                                # wall remains
    # trimmed: fewer wall faces than the original sphere had
    assert int(np.count_nonzero(ids == 0)) < n0
    assert len(eng._trim_history) == 1                        # one undo step
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_clip_and_name_trims_and_labels -q`
Expected: FAIL — `clip_and_name` undefined.

- [ ] **Step 3: Implement `clip_and_name`**

```python
def clip_and_name(self, name, origin, normal, box_planes_data=None):
    """Trim current_mesh by a plane (optionally box-scoped), cap the opening,
    label the cap as a new named patch, and merge it in. One compounding step."""
    if self.current_mesh is None:
        return None
    origin = np.asarray(origin, float); normal = np.asarray(normal, float)
    try:
        if box_planes_data:
            trimmed = self.clip_with_box(self.current_mesh, box_planes_data, origin, normal)
        else:
            trimmed = self.clip_with_plane(self.current_mesh, origin, normal)
    except RuntimeError:
        return None
    trimmed = self._carry_labels(trimmed, self.current_mesh)
    cap = self._generate_cap(self.current_mesh, origin, normal,
                             box_planes_data if box_planes_data else None)
    if cap is None or cap.n_cells == 0:
        return None
    cap = cap.triangulate()
    self._push_history()
    pid = self._new_patch_id(name)
    cap.cell_data[PATCH_ID] = np.full(cap.n_cells, pid, dtype=np.int64)
    merged = trimmed.merge(cap, merge_points=True)
    self.current_mesh = self._carry_labels(merged, trimmed)
    return self.current_mesh
```

In GUI `_on_confirm_clip` (3795), replace the `self.engine.add_clip(name, origin, normal, box_planes_data)` call with `self.engine.clip_and_name(name, origin, normal, box_planes_data)`, then `self._refresh_display()` and `self._refresh_object_tree()`.

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_clip_and_name_trims_and_labels -q`
Expected: PASS.

- [ ] **Step 5: Full suite**

Run: `conda run -n mesh-prep python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_current_mesh.py
git commit -m "Add clip_and_name: trim + cap + label in one compounding step"
```

---

### Task 7: Export by `patch_id` (equivalence-tested)

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `export_combined_stl` (1184), `export_separate_stl` (1518), and the STL-writing part of `export_openfoam`/`export_openfoam_case` that enumerates patches; add `patches_by_id`
- Test: `tests/test_export_labels.py` (create)

**Interfaces:**
- Consumes: `current_mesh`, `patch_names`, `_polydata_to_ascii_stl_block`.
- Produces: `patches_by_id() -> dict[int, pv.PolyData]` (threshold `current_mesh` by `PATCH_ID`); `patch_name_for(pid) -> str` (0 → "wall").

- [ ] **Step 1: Write the failing equivalence test**

```python
import numpy as np, pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def _two_patch_mesh():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    eng.current_mesh = sph
    ids = np.zeros(sph.n_cells, dtype=np.int64)
    ids[: sph.n_cells // 4] = 1
    eng.current_mesh.cell_data[PATCH_ID] = ids
    eng.patch_names = {1: "inlet"}
    eng._next_patch_id = 2
    return eng


def test_export_combined_has_wall_and_named_solids(tmp_path):
    eng = _two_patch_mesh()
    out = tmp_path / "m.stl"
    eng.export_combined_stl(str(out))
    text = out.read_text()
    assert "solid inlet" in text
    assert "solid wall" in text


def test_export_separate_writes_one_file_per_patch(tmp_path):
    eng = _two_patch_mesh()
    d = tmp_path / "sep"
    eng.export_separate_stl(str(d))
    assert (d / "inlet.stl").exists()
    assert (d / "wall.stl").exists()
    assert "solid inlet" in (d / "inlet.stl").read_text()
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_export_labels.py -q`
Expected: FAIL — current exporters iterate `self.clips`/`filled_patches`/`_wall_mesh`, so with none set they write only an empty wall.

- [ ] **Step 3: Implement label-based export**

```python
def patches_by_id(self):
    """{patch_id: PolyData} by grouping current_mesh faces on PATCH_ID."""
    out = {}
    if self.current_mesh is None or PATCH_ID not in self.current_mesh.cell_data:
        return out
    ids = np.asarray(self.current_mesh.cell_data[PATCH_ID])
    for pid in np.unique(ids):
        cells = np.nonzero(ids == pid)[0]
        out[int(pid)] = self.current_mesh.extract_cells(cells).extract_surface()
    return out

def patch_name_for(self, pid):
    return "wall" if pid == 0 else self.patch_names.get(pid, f"patch_{pid}")
```

Rewrite `export_combined_stl`:

```python
def export_combined_stl(self, filepath, scale_factor: float = 1.0):
    blocks = []
    for pid, mesh in sorted(self.patches_by_id().items()):
        blocks.append(self._polydata_to_ascii_stl_block(mesh, self.patch_name_for(pid), scale_factor))
    with open(filepath, "w") as f:
        f.write("".join(blocks))
```

Rewrite `export_separate_stl` analogously (one file `{name}.stl` per `patches_by_id` entry). In `export_openfoam`/`export_openfoam_case`, replace the loop that enumerates `self.clips`+`filled_patches`+wall with `self.patches_by_id()` + `patch_name_for`, so the multi-solid STL written to `constant/triSurface/` uses the same solids.

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n mesh-prep python -m pytest tests/test_export_labels.py -q`
Expected: PASS.

- [ ] **Step 5: End-to-end export smoke (real case)**

Run (background): generate a case from a labeled fixture through `export_openfoam_case` into a temp dir and assert `constant/triSurface/*.stl` contains `solid inlet` and `solid wall`, and `0/U` references the patch names. Confirm parity with the pre-refactor output on the same input.

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_export_labels.py
git commit -m "Export patches by patch_id label; preserve multi-solid STL output"
```

---

### Task 8: Undo snapshots `current_mesh`; simplify repair

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `_push_history` (1001), `undo_trim` (1005), `_apply_repair` (1076)
- Test: adapt `tests/test_trim.py`, `tests/test_selection.py`; extend `tests/test_current_mesh.py`

**Interfaces:**
- Produces: `_push_history` snapshots `self.current_mesh.copy()` (labels ride along); `undo_trim` restores it; `_apply_repair` sets `current_mesh` from the repaired mesh, re-carrying labels, and pushes one snapshot so repair is undoable.

- [ ] **Step 1: Write the failing test**

```python
def test_undo_restores_labels_after_fill():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    eng.current_mesh = sph
    eng.current_mesh.cell_data[PATCH_ID] = np.zeros(sph.n_cells, dtype=np.int64)
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = int(np.where(eng.current_mesh.cell_centers().points[:, 2] > 0.3)[0][0])
    eng.delete_cells(eng.flood_select(top))
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    assert 1 in np.unique(eng.current_mesh.cell_data[PATCH_ID])
    assert eng.undo_trim() is True                 # undo the fill
    assert 1 not in np.unique(eng.current_mesh.cell_data[PATCH_ID])
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_undo_restores_labels_after_fill -q`
Expected: FAIL — `_push_history` snapshots `original_mesh`, not `current_mesh`, so undo doesn't restore the labeled state.

- [ ] **Step 3: Repoint history/undo/repair**

```python
def _push_history(self):
    self._trim_history.append((self.current_mesh.copy(), list(self._feature_curves)))

def undo_trim(self) -> bool:
    if not self._trim_history:
        return False
    mesh, curves = self._trim_history.pop()
    self.current_mesh = mesh
    self._feature_curves = curves
    return True                                    # recompute_all no longer needed

def _apply_repair(self, repaired_mesh) -> None:
    self._push_history()
    repaired = repaired_mesh.triangulate()
    self.current_mesh = self._carry_labels(repaired, self.current_mesh)
```

(`_apply_repair`'s callers `repair_clean`/`repair_normals`/`auto_repair` now derive `repaired` from `self.current_mesh` instead of `self.original_mesh` — update those three method bodies' `self.original_mesh` references to `self.current_mesh`.)

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_undo_restores_labels_after_fill -q`
Expected: PASS.

- [ ] **Step 5: Adapt undo/repair tests, full suite**

Update `test_trim.py` and the repair tests in `test_selection.py` to use `current_mesh`. Run: `conda run -n mesh-prep python -m pytest tests/ -q`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/
git commit -m "Undo snapshots current_mesh (labels included); repair is undoable"
```

---

### Task 9: Remove the dead two-mesh fields and stale call sites

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — delete `original_mesh`, `_wall_mesh`, `clips`, `filled_patches`, `add_clip`, `recompute_all`, `set_cap_kind`, `remove_clip`, `rename_clip`, `ClipDefinition`/`FilledPatch` usage; update every remaining reference; adjacency caches key on `current_mesh`
- Test: full suite

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces: an engine whose only mesh state is `current_mesh` + `patch_names` + `_next_patch_id` + `_feature_curves` + `_trim_history` + adjacency caches.

- [ ] **Step 1: Find every remaining reference**

Run: `grep -nE "original_mesh|_wall_mesh|\.clips|filled_patches|recompute_all|add_clip|ClipDefinition|FilledPatch|set_cap_kind|remove_clip|rename_clip" mesh_prep/stl_clipper.py mesh_prep/create_case.py`
Expected: a finite list — resolve each (GUI clip list panel, `_refresh_clip_save_list`, export leftovers, adjacency `_adj_for_mesh`/`_flood_adj_for_mesh` keys → point at `current_mesh`).

- [ ] **Step 2: Write a guard test**

```python
def test_engine_has_no_two_mesh_fields(tmp_path):
    p = tmp_path / "s.stl"
    pv.Sphere(theta_resolution=12, phi_resolution=12).save(str(p))
    eng = STLClipperEngine()
    eng.load_stl(str(p))
    assert not hasattr(eng, "_wall_mesh")
    assert not hasattr(eng, "clips")
    assert not hasattr(eng, "filled_patches")
```

- [ ] **Step 3: Run to verify it fails**

Run: `conda run -n mesh-prep python -m pytest tests/test_current_mesh.py::test_engine_has_no_two_mesh_fields -q`
Expected: FAIL — fields still present.

- [ ] **Step 4: Delete fields + fix call sites**

Remove the fields from `__init__`, delete `add_clip`/`recompute_all`/`set_cap_kind`/`remove_clip`/`rename_clip`, remove the `clips`/`filled_patches` references in export (already replaced in Task 7) and the GUI clip-list panel. Point `_ensure_adjacency`/flood caches at `current_mesh`. Remove `ClipDefinition`/`FilledPatch` classes if no longer referenced (keep `ClipDefinition` only if the GUI still stores pending-clip params — otherwise delete).

- [ ] **Step 5: Run the full suite + GUI import smoke**

Run: `conda run -n mesh-prep python -m pytest tests/ -q` (expect PASS) and `conda run -n mesh-prep python -c "import mesh_prep.stl_clipper"` (expect clean import).

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py mesh_prep/create_case.py tests/
git commit -m "Remove two-mesh fields; current_mesh is the sole surface state"
```

---

## Manual verification (post-implementation, requires display)

The engine changes are covered headlessly. These GUI paths need a human run of `conda run -n mesh-prep mesh-prep <stl>`:

1. Add plane clip → the object tree, piece/profile counts, and highlights immediately reflect the trimmed surface (the original bug is gone).
2. Clip, then Ctrl+Z → surface and patch labels revert one step.
3. Clip + Fill several patches → Export → `constant/triSurface/*.stl` has one solid per patch + wall; run `./run_docker.sh` pMesh on a small case to confirm patches mesh.
