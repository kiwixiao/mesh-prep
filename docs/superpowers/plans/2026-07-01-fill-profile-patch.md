# Sub-feature D — Fill Open Profile into Named Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Fill button that triangulates a tree-selected open profile into a named cap patch, exports it as a named STL solid, and moves it to a "Named patches" node in the object tree.

**Architecture:** A `FilledPatch` record + `fill_profile()` (vtkStripper → vtkContourTriangulator → delaunay_2d fallback) stored in a new `engine.filled_patches` list; `unfilled_open_profiles()` hides filled loops via a signature. `export_combined_stl` emits each cap as a named solid. The GUI adds a Fill button, a Named-patches tree category, and cap rendering.

**Tech Stack:** Python 3.9, pyvista 0.46.5, VTK 9.2.6, PyQt5, numpy. No new dependencies.

## Global Constraints

- No new dependencies. Triangulation uses `vtk.vtkStripper` + `vtk.vtkContourTriangulator` (already used by `_generate_cap`), with `PolyData.delaunay_2d` fallback.
- Fills live in `engine.filled_patches` (new list), NOT `engine.clips` (which disables editing via `_edit_enabled`). The cap is a SEPARATE named solid; the wall keeps its opening; `fill_profile` does NOT modify `original_mesh`.
- Patch names are unique across `engine.clips`, `engine.filled_patches`, and `"wall"` (enforced at the Fill prompt).
- Clicking a tree item must not reset the camera (`reset_camera=False`).
- Run tests with `conda run -n mesh-prep pytest`.

---

### Task 1: Engine — FilledPatch + fill_profile + unfilled_open_profiles

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add `FilledPatch` dataclass after `ClipDefinition` (`:210`); add `filled_patches` init in `__init__` (`:286`) and clear in `load_stl` (`:300`); add three methods after `detect_pieces` (~`:819`).
- Test: `tests/test_selection.py` — append tests.

**Interfaces:**
- Consumes: `detect_open_profiles` (sub-feature C), `_color_for_name`, `vtk`, `pv`, numpy.
- Produces:
  - `FilledPatch(name, cap_mesh, signature, color)` dataclass
  - `fill_profile(self, profile_edges, name) -> FilledPatch | None`
  - `unfilled_open_profiles(self) -> list[pv.PolyData]`
  - `@staticmethod _profile_signature(edges) -> tuple`
  - `self.filled_patches: list[FilledPatch]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selection.py` (reuses `_sphere_engine`, `_cell_on_side`, `_n_open`):

```python
def test_fill_profile_caps_the_hole():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    prof = eng.detect_open_profiles()[0]
    patch = eng.fill_profile(prof, "inlet")
    assert patch is not None
    assert patch.name == "inlet"
    assert patch.cap_mesh.n_cells > 0
    assert len(eng.filled_patches) == 1
    merged = eng.original_mesh.merge(patch.cap_mesh, merge_points=True)
    assert _n_open(merged) == 0                       # cap closes the hole


def test_unfilled_open_profiles_excludes_filled():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    assert len(eng.unfilled_open_profiles()) == 1
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    assert len(eng.unfilled_open_profiles()) == 0


def test_fill_profile_none_on_empty():
    eng = _sphere_engine()
    assert eng.fill_profile(None, "x") is None
    assert eng.fill_profile(pv.PolyData(), "x") is None
    assert eng.filled_patches == []


def test_load_stl_resets_filled_patches(tmp_path):
    eng = STLClipperEngine()
    p1 = tmp_path / "s1.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(p1))
    eng.load_stl(str(p1))
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    assert len(eng.filled_patches) == 1
    eng.load_stl(str(p1))
    assert eng.filled_patches == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k "fill_profile or unfilled or resets_filled" -q`
Expected: FAIL — `AttributeError: 'STLClipperEngine' object has no attribute 'fill_profile'`.

- [ ] **Step 3: Add the `FilledPatch` dataclass**

Immediately after the `ClipDefinition` dataclass (ends near `stl_clipper.py:210`), add:

```python
@dataclass
class FilledPatch:
    """A named cap patch triangulated from an open profile's boundary loop."""
    name: str
    cap_mesh: pv.PolyData
    signature: tuple                       # (n_points, rounded centroid) of the filled loop
    color: tuple = (0.2, 0.6, 0.9)
```

- [ ] **Step 4: Add init + load reset**

In `STLClipperEngine.__init__`, after `self._feature_curves: list = []` (`:286`), add:

```python
        self.filled_patches: list = []     # named caps from fill_profile (sub-feature D)
```

In `load_stl`, after `self._feature_curves.clear()` (`:300`), add:

```python
        self.filled_patches.clear()
```

- [ ] **Step 5: Add the three methods**

After `detect_pieces` (~`stl_clipper.py:819`), add:

```python
    @staticmethod
    def _profile_signature(edges):
        """Stable key for a boundary loop: (point count, rounded centroid)."""
        c = np.asarray(edges.points).mean(axis=0)
        return (int(edges.n_points), tuple(np.round(c, 6)))

    def fill_profile(self, profile_edges, name):
        """Triangulate an open profile's boundary loop into a named cap patch and
        append it to filled_patches. Returns the FilledPatch, or None if the edges
        are empty or cannot be triangulated. Does not modify original_mesh."""
        if profile_edges is None or profile_edges.n_cells == 0:
            return None
        strip = vtk.vtkStripper()
        strip.SetInputData(profile_edges)
        strip.Update()
        tri = vtk.vtkContourTriangulator()
        tri.SetInputData(strip.GetOutput())
        tri.Update()
        cap = pv.wrap(tri.GetOutput())
        if cap is None or cap.n_cells == 0:
            cap = pv.PolyData(profile_edges.points).delaunay_2d()     # fallback
        if cap is None or cap.n_cells == 0:
            return None
        patch = FilledPatch(name=name, cap_mesh=cap,
                            signature=self._profile_signature(profile_edges),
                            color=_color_for_name(name))
        self.filled_patches.append(patch)
        return patch

    def unfilled_open_profiles(self):
        """Open profiles that have not been filled (matched by signature)."""
        filled = {fp.signature for fp in self.filled_patches}
        return [p for p in self.detect_open_profiles()
                if self._profile_signature(p) not in filled]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k "fill_profile or unfilled or resets_filled" -q`
Expected: PASS — 4 passed.

- [ ] **Step 7: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 49 passed (45 prior + 4 new).

- [ ] **Step 8: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_selection.py
git commit -m "Add fill_profile: triangulate an open profile into a named cap patch"
```

---

### Task 2: Export bridge — filled patches as named solids

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `export_combined_stl` (`:1053`).
- Test: `tests/test_selection.py` — append one test.

**Interfaces:**
- Consumes: `fill_profile` / `filled_patches` (Task 1), `export_combined_stl`, `_polydata_to_ascii_stl_block`.
- Produces: multi-solid STL that includes one `solid <name>` per filled patch.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_selection.py`:

```python
def test_export_combined_stl_includes_filled_patch(tmp_path):
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    out = tmp_path / "multi.stl"
    eng.export_combined_stl(str(out))
    text = out.read_text()
    assert "solid inlet" in text
    assert "solid wall" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k export_combined_stl_includes_filled -q`
Expected: FAIL — assertion error, `"solid inlet"` not in the output (only the wall solid is written).

- [ ] **Step 3: Add the filled-patch blocks**

In `export_combined_stl` (`stl_clipper.py:1053`), the body loops over `self.clips` building `blocks` then appends the `"wall"` block. Immediately **before** the line that appends the wall block (`blocks.append(self._polydata_to_ascii_stl_block(self._wall_mesh, "wall", scale_factor))`), add:

```python
        for patch in self.filled_patches:
            blocks.append(self._polydata_to_ascii_stl_block(patch.cap_mesh, patch.name, scale_factor))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k export_combined_stl_includes_filled -q`
Expected: PASS — 1 passed.

- [ ] **Step 5: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 50 passed (49 + 1 new).

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_selection.py
git commit -m "Export filled patches as named STL solids in export_combined_stl"
```

---

### Task 3: GUI — Fill button + Named-patches tree node

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — Fill button after `_btn_delete` (`:1839`); tree-state init near `:1969`; `_refresh_object_tree` (`:3960`); `_on_tree_item_clicked` (`:4000`); `_refresh_display` cap-draw; add `_on_fill_profile`.

**Interfaces:**
- Consumes: `engine.fill_profile`, `engine.unfilled_open_profiles`, `engine.filled_patches` (Tasks 1-2); existing `_refresh_display`, `_refresh_object_tree`, `_tree_mesh`, `QInputDialog`, `QMessageBox`.
- Produces: GUI behavior only.

- [ ] **Step 1: Add the Fill button**

After the `self._btn_delete` block (`stl_clipper.py:1837`-`:1839`), add:

```python
        self._btn_fill = QPushButton("🩹 Fill profile → patch")
        self._btn_fill.setToolTip("Fill the open profile selected in the Objects tree into a named patch")
        self._btn_fill.clicked.connect(self._on_fill_profile)
        panel.addWidget(self._btn_fill)
```

- [ ] **Step 2: Add tree-state init**

Where the tree caches are initialized (near `self._tree_pieces = []`, `:1969`), add:

```python
        self._tree_patches = []
        self._tree_n_patches = -1
        self._active_profile_index = None
```

- [ ] **Step 3: Update `_refresh_object_tree`**

Replace the guard + data-source lines at the top of `_refresh_object_tree` (`:3963`-`:3969`):

```python
        mesh = self.engine.original_mesh
        if mesh is self._tree_mesh:
            return
        self._tree_mesh = mesh
        self._tree_profiles = self.engine.detect_open_profiles()
        self._tree_nonmanifold = self.engine.detect_nonmanifold_edges()
        self._tree_pieces = self.engine.detect_pieces()
```

with:

```python
        mesh = self.engine.original_mesh
        npatch = len(self.engine.filled_patches)
        if mesh is self._tree_mesh and npatch == self._tree_n_patches:
            return
        self._tree_mesh = mesh
        self._tree_n_patches = npatch
        self._tree_profiles = self.engine.unfilled_open_profiles()
        self._tree_nonmanifold = self.engine.detect_nonmanifold_edges()
        self._tree_pieces = self.engine.detect_pieces()
        self._tree_patches = self.engine.filled_patches
```

Then, immediately after the "Disconnected pieces" category block (after its `else: QTreeWidgetItem(pc, ["(none)"]).setDisabled(True)`, `:3998`), add the Named-patches category:

```python
        pt = QTreeWidgetItem(tree, ["Named patches"])
        pt.setExpanded(True)
        if self._tree_patches:
            for i, p in enumerate(self._tree_patches):
                it = QTreeWidgetItem(pt, [f"{p.name} ({p.cap_mesh.n_cells} faces)"])
                it.setData(0, Qt.UserRole, ("patch", i))
        else:
            QTreeWidgetItem(pt, ["(none)"]).setDisabled(True)
```

- [ ] **Step 4: Update `_on_tree_item_clicked`**

In the `kind == "profile"` branch (`:4009`-`:4013`), after the `self.status.showMessage(...)` line, add:

```python
            self._active_profile_index = index
```

And add a new branch after the `kind == "piece"` branch (after its `self._update_button_states()`, `:4025`):

```python
        elif kind == "patch":
            patch = self._tree_patches[index]
            self.plotter.add_mesh(patch.cap_mesh, color="green", opacity=0.8,
                                  name="tree_highlight", reset_camera=False)
            self.status.showMessage(f"Patch '{patch.name}' — {patch.cap_mesh.n_cells} faces.")
```

- [ ] **Step 5: Draw the caps in `_refresh_display`**

In `_refresh_display`, immediately after the loop that draws feature-curve tubes (`for i, curve in enumerate(self.engine._feature_curves):` ... — the loop that adds `name=f"feature_curve_{i}"`), add:

```python
        for i, patch in enumerate(self.engine.filled_patches):
            if patch.cap_mesh is not None and patch.cap_mesh.n_cells > 0:
                self.plotter.add_mesh(patch.cap_mesh, color=patch.color,
                                      name=f"patch_{i}", reset_camera=False)
```

- [ ] **Step 6: Add `_on_fill_profile`**

Add on `STLClipperApp` (place near `_on_delete_selection`):

```python
    def _on_fill_profile(self):
        """Fill the open profile selected in the Objects tree into a named patch."""
        if self.engine.original_mesh is None:
            self.status.showMessage("Load an STL first.")
            return
        idx = self._active_profile_index
        profiles = self._tree_profiles
        if idx is None or not (0 <= idx < len(profiles)):
            self.status.showMessage("Select an open profile in the Objects tree first.")
            return
        used = ({c.name for c in self.engine.clips}
                | {p.name for p in self.engine.filled_patches} | {"wall"})
        if "inlet" not in used:
            default = "inlet"
        else:
            n = 1
            while f"outlet_{n}" in used:
                n += 1
            default = f"outlet_{n}"
        name, ok = QInputDialog.getText(self, "Patch Name", "Name for this patch:", text=default)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in used:
            QMessageBox.warning(self, "Duplicate Name", f"'{name}' is already used. Choose another.")
            return
        patch = self.engine.fill_profile(profiles[idx], name)
        if patch is None:
            self.status.showMessage("Could not fill this profile.")
            return
        self._active_profile_index = None
        self._refresh_display()
        self._refresh_object_tree()
        self.status.showMessage(f"Filled patch '{name}' ({patch.cap_mesh.n_cells} faces).")
```

- [ ] **Step 7: Verify the module compiles and imports**

Run: `conda run -n mesh-prep python -c "import mesh_prep.stl_clipper; print('IMPORT_OK')"`
Expected: `IMPORT_OK`.

- [ ] **Step 8: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 50 passed.

- [ ] **Step 9: Manual GUI smoke (record result)**

No GUI harness (see `mesh-prep/CLAUDE.md`); verify by running:

```bash
conda run -n mesh-prep mesh-prep /Users/xiaz9n/openfoam/AO_Native_001.stl
```
Steps: Cut → double-click a side → Delete faces (an **Open Profile** appears in the tree) → click that Open Profile → press **🩹 Fill profile → patch** → name it `inlet` → the hole is capped with a colored disk, and the tree entry moves from **Open Profiles** to **Named patches** (`inlet (…faces)`); clicking it highlights the cap green. Export (existing OpenFOAM export) writes a `solid inlet` in the multi-solid STL.

- [ ] **Step 10: Commit**

```bash
git add mesh_prep/stl_clipper.py
git commit -m "Add Fill button and Named-patches tree node for filled profiles"
```

---

## Self-Review

**1. Spec coverage:**
- §3 `FilledPatch`, `filled_patches` (+ load reset), `_profile_signature`, `fill_profile`, `unfilled_open_profiles` → Task 1. ✓
- §4 export bridge in `export_combined_stl` → Task 2. ✓
- §5 Fill button, active-profile tracking, `_on_fill_profile` (name default + dup/"wall" reject), cap display, Named-patches tree node, guard keyed on patch count, Open Profiles uses `unfilled_open_profiles` → Task 3 Steps 1-6. ✓
- §7 edge cases: no profile / stale index → hint; duplicate/"wall"/empty → reject; un-triangulable → None + "Could not fill"; `(none)` for empty patches → Task 3 + Task 1. ✓
- §8 tests (fill caps hole, unfilled excludes, none-on-empty, load reset, export includes solid) → Task 1 + Task 2. ✓
- Constraints: separate list (not clips), cap separate solid, no `original_mesh` mutation, unique names, no camera reset → honored. ✓

**2. Placeholder scan:** No TBD/TODO; every code step complete; every run step has an exact command + expected output. ✓

**3. Type consistency:** `fill_profile(profile_edges, name) -> FilledPatch|None` consumed in `_on_fill_profile` with `patch.cap_mesh.n_cells`/`patch.name`. `unfilled_open_profiles() -> list[pv.PolyData]` assigned to `_tree_profiles` and indexed by `_active_profile_index`. `FilledPatch(name, cap_mesh, signature, color)` fields used in export (`patch.name`, `patch.cap_mesh`), tree (`p.name`, `p.cap_mesh.n_cells`), display (`patch.color`). `_tree_patches`/`_tree_n_patches`/`_active_profile_index` consistent across init, refresh, click, fill. ✓
