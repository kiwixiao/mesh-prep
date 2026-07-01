# Sub-feature C — Object Tree Panel + Entity Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A left-side Object Tree panel that auto-detects and lists open profiles, non-manifold edges, and disconnected pieces, and highlights each on click (pieces also load into the face selection for deletion).

**Architecture:** Three headless engine detection methods (`detect_open_profiles`, `detect_nonmanifold_edges`, `detect_pieces`) built on `extract_feature_edges` + `connectivity('all')`. A `QTreeWidget` added as the left pane of the existing splitter, rebuilt by an identity-guarded `_refresh_object_tree()` hooked into `_refresh_display`, with an `itemClicked` handler that draws a `"tree_highlight"` actor (no camera move).

**Tech Stack:** Python 3.9, pyvista 0.46.5, VTK 9.2.6, PyQt5, numpy. No new dependencies.

## Global Constraints

- No new runtime dependencies (pyvista/VTK/PyQt5/numpy only).
- Detection methods are headless and pure (no Qt, no mutation of engine state).
- `connectivity('all')` labels cells with `RegionId` without reordering — `RegionId==r` indices are valid `original_mesh` cell ids.
- Do NOT reorganize the existing right-side tab panel, add Fill/Repair, rename entities, or repair non-manifold geometry. C is detection + tree + highlight only.
- Clicking must NOT move/reset the camera (`reset_camera=False`, no `reset_camera()` call).
- Tree refresh is identity-guarded so view-only `_refresh_display` calls (opacity / mesh-edge toggles) do not recompute detection.
- Run tests with `conda run -n mesh-prep pytest`.

---

### Task 1: Engine entity-detection methods

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add four methods on `STLClipperEngine` after `flood_select` (near `:780`).
- Test: `tests/test_selection.py` — append after the flood tests.

**Interfaces:**
- Consumes: `self.original_mesh`; `pv.PolyData.extract_feature_edges`, `.connectivity('all')`, `.extract_cells`, `.extract_surface`.
- Produces:
  - `detect_open_profiles(self) -> list[pv.PolyData]` (each = one boundary loop's line geometry)
  - `detect_nonmanifold_edges(self) -> list[pv.PolyData]` (each = one non-manifold edge group)
  - `detect_pieces(self) -> list[list[int]]` (each = sorted cell ids of one connected component)
  - `_split_edge_groups(self, edges) -> list[pv.PolyData]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selection.py` (reuses `_sphere_engine` / `_cell_on_side` added in the flood-select tests):

```python
def test_detect_open_profiles_counts_loops():
    eng = _sphere_engine()
    assert eng.detect_open_profiles() == []                # watertight sphere
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    assert len(eng.detect_open_profiles()) == 1            # one hole after delete


def test_detect_open_profiles_two_loops():
    eng = STLClipperEngine()
    cyl = pv.Cylinder(radius=1, height=4, resolution=40, capping=True).triangulate()
    eng.original_mesh = cyl
    eng._wall_mesh = cyl.copy()
    eng.cut_by_plane((0, 0, 1.0), (0, 0, 1))
    eng.cut_by_plane((0, 0, -1.0), (0, 0, 1))
    mid = int(np.where(np.abs(eng.original_mesh.cell_centers().points[:, 2]) < 0.5)[0][0])
    eng.delete_cells(eng.flood_select(mid))
    assert len(eng.detect_open_profiles()) == 2


def test_detect_open_profiles_no_mesh():
    assert STLClipperEngine().detect_open_profiles() == []


def test_detect_pieces_body_and_stray():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=20, phi_resolution=20).triangulate()
    stray = pv.PolyData(np.array([(10, 10, 10), (10.1, 10, 10), (10, 10.1, 10)], float),
                        np.array([3, 0, 1, 2]))
    combined = sph.merge(stray, merge_points=False)
    eng.original_mesh = combined
    eng._wall_mesh = combined.copy()
    pieces = eng.detect_pieces()
    assert len(pieces) == 2
    assert sorted(len(p) for p in pieces) == [1, 720]
    allids = sorted(i for p in pieces for i in p)
    assert allids == list(range(combined.n_cells))         # valid ids, full partition


def test_detect_pieces_single_body():
    eng = _sphere_engine()
    pieces = eng.detect_pieces()
    assert len(pieces) == 1
    assert len(pieces[0]) == eng.original_mesh.n_cells


def test_detect_pieces_no_mesh():
    assert STLClipperEngine().detect_pieces() == []


def test_detect_nonmanifold_edges():
    eng = STLClipperEngine()
    pts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)], float)
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 1, 3], [3, 0, 1, 4]])   # 3 tris share edge (0,1)
    eng.original_mesh = pv.PolyData(pts, faces)
    eng._wall_mesh = eng.original_mesh.copy()
    assert len(eng.detect_nonmanifold_edges()) == 1
    assert _sphere_engine().detect_nonmanifold_edges() == []        # clean sphere
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k detect -q`
Expected: FAIL — `AttributeError: 'STLClipperEngine' object has no attribute 'detect_open_profiles'`.

- [ ] **Step 3: Implement the four methods**

Insert after `flood_select` (near `stl_clipper.py:780`):

```python
    def _split_edge_groups(self, edges):
        """Split an edge PolyData into connected groups; one geometry per group
        (line cells preserved) for highlighting. [] if empty/None."""
        if edges is None or edges.n_cells == 0:
            return []
        conn = edges.connectivity('all')
        rid = np.asarray(conn['RegionId'])
        return [conn.extract_cells(np.nonzero(rid == r)[0]).extract_surface()
                for r in np.unique(rid)]

    def detect_open_profiles(self):
        """List of boundary-edge loops (open profiles / holes), one pv.PolyData per
        connected loop. [] if watertight or no mesh."""
        m = self.original_mesh
        if m is None:
            return []
        edges = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                        manifold_edges=False, non_manifold_edges=False)
        return self._split_edge_groups(edges)

    def detect_nonmanifold_edges(self):
        """List of non-manifold edge groups (edges shared by >2 faces), one
        pv.PolyData per connected group. [] if none or no mesh."""
        m = self.original_mesh
        if m is None:
            return []
        edges = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                        manifold_edges=False, non_manifold_edges=True)
        return self._split_edge_groups(edges)

    def detect_pieces(self):
        """List of connected components; each entry is a sorted list of cell ids into
        original_mesh. [] if no mesh. Single watertight body -> one entry."""
        m = self.original_mesh
        if m is None:
            return []
        conn = m.connectivity('all')
        rid = np.asarray(conn['RegionId'])
        return [sorted(int(c) for c in np.nonzero(rid == r)[0]) for r in np.unique(rid)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k detect -q`
Expected: PASS — 7 passed.

- [ ] **Step 5: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 41 passed (34 prior + 7 new).

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_selection.py
git commit -m "Add surface-entity detection: open profiles, non-manifold edges, pieces"
```

---

### Task 2: Object Tree panel (GUI)

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — PyQt5 imports (`:40`); splitter construction (`:1868`); add `_refresh_object_tree` and `_on_tree_item_clicked` on `STLClipperApp`; one-line hook at the end of `_refresh_display` (`:3934`).

**Interfaces:**
- Consumes: `engine.detect_open_profiles() -> list[pv.PolyData]`, `engine.detect_nonmanifold_edges() -> list[pv.PolyData]`, `engine.detect_pieces() -> list[list[int]]` (Task 1); existing `self._selection`, `self._refresh_selection_highlight()`, `self._update_button_states()`, `self.plotter`, `self.status`.
- Produces: GUI behavior only.

- [ ] **Step 1: Add QTreeWidget imports**

In the PyQt5 widget import block (near `stl_clipper.py:40`, which already imports `QListWidget, QListWidgetItem`), add `QTreeWidget` and `QTreeWidgetItem` to the imported names.

Run: `conda run -n mesh-prep python -c "from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem; print('OK')"`
Expected: `OK`.

- [ ] **Step 2: Add the tree to the splitter**

Replace the splitter block at `stl_clipper.py:1868` (currently adds `plotter.interactor` then `_tab_widget`) with:

```python
        # Object tree (left) — detected surface entities
        self._tree_mesh = False           # sentinel so the first refresh always builds
        self._tree_profiles = []
        self._tree_nonmanifold = []
        self._tree_pieces = []
        self._object_tree = QTreeWidget()
        self._object_tree.setHeaderLabel("Objects")
        self._object_tree.itemClicked.connect(self._on_tree_item_clicked)

        # Draggable splitter: object tree | viewport | tab panel
        splitter = QSplitter(Qt.Horizontal, central)
        splitter.addWidget(self._object_tree)
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(self._tab_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 1)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        layout.addWidget(splitter)
```

- [ ] **Step 3: Add `_refresh_object_tree` and `_on_tree_item_clicked`**

Add these two methods on `STLClipperApp` (place them next to `_refresh_selection_highlight`, near `stl_clipper.py:3766`):

```python
    def _refresh_object_tree(self):
        """Rebuild the object tree from detected surface entities. Identity-guarded:
        skips recompute when the mesh is unchanged since the last build, so view-only
        refreshes stay free."""
        mesh = self.engine.original_mesh
        if mesh is self._tree_mesh:
            return
        self._tree_mesh = mesh
        self._tree_profiles = self.engine.detect_open_profiles()
        self._tree_nonmanifold = self.engine.detect_nonmanifold_edges()
        self._tree_pieces = self.engine.detect_pieces()
        tree = self._object_tree
        tree.clear()

        prof = QTreeWidgetItem(tree, ["Open Profiles"])
        prof.setExpanded(True)
        if self._tree_profiles:
            for i, g in enumerate(self._tree_profiles):
                it = QTreeWidgetItem(prof, [f"Open Profile {i + 1} ({g.n_cells} edges)"])
                it.setData(0, Qt.UserRole, ("profile", i))
        else:
            QTreeWidgetItem(prof, ["(none)"]).setDisabled(True)

        nm = QTreeWidgetItem(tree, ["Non-manifold edges"])
        nm.setExpanded(True)
        if self._tree_nonmanifold:
            for i, g in enumerate(self._tree_nonmanifold):
                it = QTreeWidgetItem(nm, [f"Non-manifold group {i + 1} ({g.n_cells} edges)"])
                it.setData(0, Qt.UserRole, ("nonmanifold", i))
        else:
            QTreeWidgetItem(nm, ["(none)"]).setDisabled(True)

        pc = QTreeWidgetItem(tree, ["Disconnected pieces"])
        pc.setExpanded(True)
        if self._tree_pieces:
            for i, cells in enumerate(self._tree_pieces):
                it = QTreeWidgetItem(pc, [f"Piece {i + 1} ({len(cells)} faces)"])
                it.setData(0, Qt.UserRole, ("piece", i))
        else:
            QTreeWidgetItem(pc, ["(none)"]).setDisabled(True)

    def _on_tree_item_clicked(self, item, column):
        """Highlight the clicked entity (no camera move). Pieces also load into the
        face selection so Delete faces removes them."""
        data = item.data(0, Qt.UserRole)
        self.plotter.remove_actor("tree_highlight", render=False)
        if data is None:                                  # category header or (none)
            self.plotter.render()
            return
        kind, index = data
        if kind == "profile":
            geom = self._tree_profiles[index]
            self.plotter.add_mesh(geom, color="orange", line_width=6,
                                  name="tree_highlight", reset_camera=False)
            self.status.showMessage(f"Open Profile {index + 1} — {geom.n_cells} edges.")
        elif kind == "nonmanifold":
            geom = self._tree_nonmanifold[index]
            self.plotter.add_mesh(geom, color="red", line_width=6,
                                  name="tree_highlight", reset_camera=False)
            self.status.showMessage(f"Non-manifold group {index + 1} — {geom.n_cells} edges.")
        elif kind == "piece":
            cells = self._tree_pieces[index]
            self._selection = set(cells)
            self._refresh_selection_highlight()
            self.status.showMessage(
                f"Piece {index + 1} — {len(cells)} faces selected. Delete faces to remove.")
            self._update_button_states()
        self.plotter.render()
```

- [ ] **Step 4: Hook the refresh into `_refresh_display`**

At the very end of `_refresh_display` (`stl_clipper.py:3934`, after all actors are added and the final `self.plotter.render()` / return), add one line so the tree stays in sync after every topology change:

```python
        self._refresh_object_tree()
```

(The identity guard makes this a no-op on view-only refreshes.)

- [ ] **Step 5: Verify the module compiles and imports**

Run: `conda run -n mesh-prep python -c "import mesh_prep.stl_clipper; print('IMPORT_OK')"`
Expected: `IMPORT_OK`.

- [ ] **Step 6: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 41 passed (GUI change does not touch the headless engine tests).

- [ ] **Step 7: Manual GUI smoke (record result)**

No GUI test harness exists (see `mesh-prep/CLAUDE.md`); verify by running:

```bash
conda run -n mesh-prep mesh-prep /Users/xiaz9n/openfoam/AO_Native_001.stl
```
Expect: an **Objects** tree appears on the left with **Open Profiles / Non-manifold edges / Disconnected pieces** (a clean watertight aorta shows `(none)` / `(none)` / `Piece 1`). After a Cut → double-click one side → Delete faces, an **Open Profile 1** child appears; clicking it draws an **orange** loop on the cut with no camera jump. Clicking a **Piece** selects its faces (Delete removes it). Clicking a category header or `(none)` clears the highlight.

- [ ] **Step 8: Commit**

```bash
git add mesh_prep/stl_clipper.py
git commit -m "Add object tree panel with entity detection and click-to-highlight"
```

---

## Self-Review

**1. Spec coverage:**
- §4 `detect_open_profiles` / `detect_nonmanifold_edges` / `detect_pieces` / `_split_edge_groups` → Task 1 Step 3. ✓
- §5 tree in splitter (left), `_refresh_object_tree` identity-guarded, `_on_tree_item_clicked` (orange/red/piece→selection), hook in `_refresh_display` → Task 2 Steps 2-4. ✓
- §5 counts in labels, `(none)` for empty, `setData(Qt.UserRole,(kind,i))` → Task 2 Step 3. ✓
- §7 edge cases: watertight → (none)/(none)/Piece1; no mesh → sentinel builds (none) categories; header/(none) click clears highlight; view-only refresh guarded → covered by identity guard + `data is None` branch. ✓
- §8 tests (open profiles ×3, pieces ×3, non-manifold ×1) → Task 1 Step 1. ✓
- Constraints: no camera move (`reset_camera=False` + no reset), no right-panel reorg, no new deps → honored. ✓

**2. Placeholder scan:** No TBD/TODO; every code step is complete; every run step has an exact command + expected output. ✓

**3. Type consistency:** `detect_pieces -> list[list[int]]` consumed as `self._tree_pieces[index]` (list of ids) → `set(cells)`. `detect_open_profiles/nonmanifold -> list[pv.PolyData]` consumed as `geom` with `.n_cells` and passed to `add_mesh`. `setData(0, Qt.UserRole, (kind, i))` read back as `kind, index`. Sentinel `self._tree_mesh = False` vs identity check `mesh is self._tree_mesh` consistent. ✓
