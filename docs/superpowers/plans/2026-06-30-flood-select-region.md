# Sub-feature B — Double-click Flood-Select Region Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Select mode, a double-click selects the whole connected surface region under the cursor, stopping at feature curves (cuts), and unions it into the active selection.

**Architecture:** A headless engine method `flood_select(seed_cell)` does a BFS over cell edge-adjacency that excludes feature-curve edges (the "walls"). Barrier edges are reconstructed from the already-stored `_feature_curves` by mapping each curve point to a current mesh vertex (tolerance-guarded, so deleted regions drop out). The GUI adds double-click detection to the existing `_SelectLassoStyle` and a `_select_double_click` handler; no changes to the working cut/undo/lasso code.

**Tech Stack:** Python 3.9, pyvista 0.46.5, VTK 9.2.6, PyQt5, numpy. No new dependencies (scipy is NOT available — use `pyvista.PolyData.find_closest_point`).

## Global Constraints

- No new runtime dependencies. scipy is not a declared dep — do not import it. Use `mesh.find_closest_point(point) -> int`.
- Do NOT modify `cut_by_plane`, `cut_by_box`, `_feature_curves`, `_push_history`, `undo_trim`, or the existing lasso/single-click select. B is additive.
- Edge-adjacency **must** pair edges with `np.tile(np.arange(n_cells), 3)` for `cell_of`, NOT `np.repeat`. Using `repeat` silently builds a wrong adjacency. Test 1 (sphere) exists to catch this.
- Barrier point→vertex mapping is tolerance-guarded at `1e-6 * bbox_diagonal`; a feature-curve segment contributes a wall only if BOTH endpoints pass the guard.
- Cache pattern mirrors the existing `_ensure_adjacency` (`stl_clipper.py:531`): rebuild when `original_mesh` identity changes OR `len(_feature_curves)` changes.
- All engine methods headless (no Qt). Run tests with `conda run -n mesh-prep pytest`.

---

### Task 1: Engine `flood_select` + cached barrier/edge-adjacency

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add 4 init fields near `:269`; add three methods (`_barrier_edge_keys`, `_ensure_flood_adjacency`, `flood_select`) after `grow_cells`/`smooth_cells` (around `:607`).
- Test: `tests/test_selection.py` — append flood tests after `test_load_stl_resets_cut_state` (`:300`).

**Interfaces:**
- Consumes: `self.original_mesh` (pv.PolyData, triangles), `self._feature_curves` (list of pv.PolyData polylines), `pv.PolyData.find_closest_point`.
- Produces: `flood_select(self, seed_cell: int) -> list[int]` — sorted cell ids of the region; `[]` if seed out of range or no mesh; `[seed_cell]` on a non-triangle mesh.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selection.py`:

```python
def _sphere_engine(theta=24, phi=24):
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=theta, phi_resolution=phi)
    eng.original_mesh = sph
    eng._wall_mesh = sph.copy()
    return eng


def _cell_on_side(mesh, axis_idx, positive):
    col = mesh.cell_centers().points[:, axis_idx]
    idx = np.where(col > 0.3)[0] if positive else np.where(col < -0.3)[0]
    return int(idx[0])


def test_flood_select_no_cut_returns_whole_component():
    eng = _sphere_engine()
    n = eng.original_mesh.n_cells
    assert len(eng.flood_select(0)) == n          # closed sphere is one component


def test_flood_select_after_cut_separates_two_sides():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    mesh = eng.original_mesh
    top = _cell_on_side(mesh, 2, True)
    bot = _cell_on_side(mesh, 2, False)
    rt = set(eng.flood_select(top))
    rb = set(eng.flood_select(bot))
    assert rt and rb
    assert rt.isdisjoint(rb)                        # the cut is a wall
    assert len(rt) + len(rb) == mesh.n_cells        # partition the whole surface
    assert top in rt and bot in rb


def test_flood_select_two_cuts_bounded_region():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    eng.cut_by_plane((0, 0, 0), (1, 0, 0))
    mesh = eng.original_mesh
    region = eng.flood_select(0)
    assert 0 < len(region) < mesh.n_cells           # a quadrant, strictly smaller


def test_flood_select_survives_delete():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))         # remove one side
    m2 = eng.original_mesh
    r2 = eng.flood_select(0)                         # no crash, no leak beyond mesh
    assert 0 < len(r2) <= m2.n_cells


def test_flood_select_out_of_range_returns_empty():
    eng = _sphere_engine()
    assert eng.flood_select(-1) == []
    assert eng.flood_select(10 ** 9) == []


def test_flood_select_no_mesh_returns_empty():
    assert STLClipperEngine().flood_select(0) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k flood -q`
Expected: FAIL — `AttributeError: 'STLClipperEngine' object has no attribute 'flood_select'`.

- [ ] **Step 3: Add cache fields in `__init__`**

In `STLClipperEngine.__init__`, right after `self._feature_curves: list = []` (`stl_clipper.py:269`), add:

```python
        # Flood-select (sub-feature B) cached barrier edge-adjacency
        self._flood_adj_for_mesh = None
        self._flood_adj_n_curves = -1
        self._flood_adj = None
        self._flood_ok = False
```

- [ ] **Step 4: Implement the three methods**

Insert after `smooth_cells` returns (before `def delete_cells`, around `stl_clipper.py:672`):

```python
    def _barrier_edge_keys(self, mesh):
        """Edge keys (min*n_points + max) for every feature-curve segment, mapped to
        current mesh vertices. A segment contributes a wall only if BOTH endpoints
        map to a vertex within 1e-6 * bbox-diagonal, so feature-curve points whose
        region was deleted drop out instead of snapping to a wrong vertex."""
        n_points = mesh.n_points
        b = np.asarray(mesh.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        tol = 1e-6 * diag if diag > 0 else 1e-6
        barrier = set()
        for curve in self._feature_curves:
            if curve is None or curve.n_cells == 0:
                continue
            pts = np.asarray(curve.points)
            pid = np.empty(curve.n_points, dtype=np.int64)
            ok = np.zeros(curve.n_points, dtype=bool)
            for i in range(curve.n_points):
                j = int(mesh.find_closest_point(pts[i]))
                pid[i] = j
                ok[i] = float(np.linalg.norm(pts[i] - mesh.points[j])) <= tol
            for seg in curve.lines.reshape(-1, 3):
                i0, i1 = int(seg[1]), int(seg[2])
                if ok[i0] and ok[i1]:
                    u, v = int(pid[i0]), int(pid[i1])
                    if u != v:
                        if u > v:
                            u, v = v, u
                        barrier.add(u * n_points + v)
        return barrier

    def _ensure_flood_adjacency(self):
        """Build (once per mesh identity + feature-curve count) cell edge-adjacency
        that excludes feature-curve edges, so a flood never crosses a cut. Sets
        _flood_ok = False on a non-triangle mesh (no fast path). Mirrors the
        _ensure_adjacency cache pattern."""
        mesh = self.original_mesh
        if mesh is None:
            self._flood_adj_for_mesh = None
            self._flood_adj_n_curves = -1
            self._flood_adj = None
            self._flood_ok = False
            return
        if (self._flood_adj_for_mesh is mesh
                and self._flood_adj_n_curves == len(self._feature_curves)):
            return
        self._flood_adj_for_mesh = mesh
        self._flood_adj_n_curves = len(self._feature_curves)
        faces = mesh.faces
        if faces.size != 4 * mesh.n_cells or not bool(
                (faces.reshape(-1, 4)[:, 0] == 3).all()):
            self._flood_adj = None
            self._flood_ok = False
            return
        n_cells, n_points = mesh.n_cells, mesh.n_points
        tri = faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        barrier = self._barrier_edge_keys(mesh)
        e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        e.sort(axis=1)
        keys = e[:, 0] * n_points + e[:, 1]
        cell_of = np.tile(np.arange(n_cells, dtype=np.int64), 3)   # tile, NOT repeat
        order = np.argsort(keys, kind="stable")
        keys_s = keys[order]
        cells_s = cell_of[order]
        adj = [[] for _ in range(n_cells)]
        n = len(keys_s)
        i = 0
        while i < n:
            j = i
            while j < n and keys_s[j] == keys_s[i]:
                j += 1
            if (j - i) == 2 and int(keys_s[i]) not in barrier:
                c0 = int(cells_s[i])
                c1 = int(cells_s[i + 1])
                adj[c0].append(c1)
                adj[c1].append(c0)
            i = j
        self._flood_adj = adj
        self._flood_ok = True

    def flood_select(self, seed_cell):
        """Connected surface region containing seed_cell, with feature-curve edges as
        walls (the flood never crosses a cut). Returns a sorted list of cell ids; []
        if the seed is out of range or there is no mesh; [seed_cell] on a non-triangle
        mesh."""
        mesh = self.original_mesh
        if mesh is None:
            return []
        seed = int(seed_cell)
        if seed < 0 or seed >= mesh.n_cells:
            return []
        self._ensure_flood_adjacency()
        if not self._flood_ok:
            return [seed]
        adj = self._flood_adj
        seen = {seed}
        stack = [seed]
        while stack:
            c = stack.pop()
            for nb in adj[c]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return sorted(seen)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k flood -q`
Expected: PASS — 6 passed.

- [ ] **Step 6: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — all previously-passing tests still pass (28 + 6 = 34).

- [ ] **Step 7: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_selection.py
git commit -m "Add flood_select: region flood-fill bounded by feature curves"
```

---

### Task 2: GUI double-click wiring

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — `_SelectLassoStyle.__init__` (`:1242`) and `_on_release` (`:1274`); add `_pick_wall_cell` and `_select_double_click` on `STLClipperApp`; refactor `_select_click_pick` (`:3600`) to use `_pick_wall_cell`. Ensure `import time` at top of module.

**Interfaces:**
- Consumes: `self.engine.flood_select(cid)` (Task 1); existing `self._wall_actor`, `self._selection`, `self._btn_select`, `self._edit_enabled()`, `self._refresh_selection_highlight()`, `self._update_button_states()`.
- Produces: GUI behavior only (no new engine surface).

- [ ] **Step 1: Ensure `time` is imported**

Check the top of `mesh_prep/stl_clipper.py`. If `import time` is absent, add it beside the other stdlib imports.

Run: `conda run -n mesh-prep python -c "import mesh_prep.stl_clipper as m, inspect; print('time' in dir(m) or 'import time present')"`
Expected: prints truthy (module imports cleanly).

- [ ] **Step 2: Add double-click state to `_SelectLassoStyle.__init__`**

In `_SelectLassoStyle.__init__` (`:1242`), after `self._lasso_active = False`, add:

```python
        self._last_click_time = 0.0
        self._last_click_pos = None
```

- [ ] **Step 3: Add double-click detection in `_on_release`**

Replace the click branch at the end of `_on_release` (`:1281`-`:1286`, from `self.OnLeftButtonUp()` through the `_select_click_pick` call) with:

```python
            self.OnLeftButtonUp()
            pp = getattr(self, "_press_pos", None)
            rp = self._app.plotter.iren.get_event_position()
            if pp is None or abs(rp[0] - pp[0]) > 3 or abs(rp[1] - pp[1]) > 3:
                return                                    # a rotate/drag, not a click
            now = time.time()
            last_t = self._last_click_time
            last_p = self._last_click_pos
            if (last_p is not None and (now - last_t) <= 0.4
                    and abs(rp[0] - last_p[0]) <= 8 and abs(rp[1] - last_p[1]) <= 8):
                self._last_click_time = 0.0               # consume; no triple-click chain
                self._app._select_double_click(rp)
            else:
                self._last_click_time = now
                self._last_click_pos = rp
                self._app._select_click_pick(rp)
```

- [ ] **Step 4: Extract `_pick_wall_cell` and refactor `_select_click_pick`**

Replace `_select_click_pick` (`:3600`-`:3621`) with the shared helper plus the slimmed single-click handler:

```python
    def _pick_wall_cell(self, pos):
        """Cell id of the wall face under display position `pos`, or None. The picker
        is restricted to the wall actor so it cannot catch the centerline or feature
        curves. Shared by single-click and double-click selection."""
        wall_actor = getattr(self, "_wall_actor", None)
        mesh = self.engine.original_mesh
        if wall_actor is None or mesh is None:
            return None
        picker = vtk.vtkCellPicker()
        picker.InitializePickList()
        picker.AddPickList(wall_actor)
        picker.PickFromListOn()
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        cid = picker.GetCellId()
        if cid is None or cid < 0 or cid >= mesh.n_cells:
            return None
        return int(cid)

    def _select_click_pick(self, pos):
        """A single click in Select mode adds the one face under the cursor to the
        active selection."""
        if not self._btn_select.isChecked() or not self._edit_enabled():
            return
        cid = self._pick_wall_cell(pos)
        if cid is None:
            return
        self._selection.add(cid)
        self._refresh_selection_highlight()
        self.status.showMessage(f"Selected {len(self._selection)} faces.")
        self._update_button_states()

    def _select_double_click(self, pos):
        """A double click in Select mode floods the connected surface region under the
        cursor (bounded by feature curves) into the active selection."""
        if not self._btn_select.isChecked() or not self._edit_enabled():
            return
        cid = self._pick_wall_cell(pos)
        if cid is None:
            return
        region = self.engine.flood_select(cid)
        self._selection |= set(region)
        self._refresh_selection_highlight()
        self.status.showMessage(
            f"Flood-selected {len(region)} faces ({len(self._selection)} total).")
        self._update_button_states()
```

- [ ] **Step 5: Verify module compiles and imports**

Run: `conda run -n mesh-prep python -c "import mesh_prep.stl_clipper; print('IMPORT_OK')"`
Expected: `IMPORT_OK` (no SyntaxError, no NameError).

- [ ] **Step 6: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 34 passed (GUI change does not touch headless engine tests).

- [ ] **Step 7: Manual GUI smoke (record result)**

This repo has no GUI test harness (see `mesh-prep/CLAUDE.md`); verify by running:

```bash
conda run -n mesh-prep mesh-prep /Users/xiaz9n/openfoam/AO_Native_001.stl
```
Steps: position a cut plane → **✂ Cut** (cyan feature curve appears) → enter **Select** mode → **double-click** one side of the surface. Expect: that entire side highlights up to the cut, status shows `Flood-selected N faces (...)`, and **Delete faces** removes exactly that side. A single click still picks one face; the view does not jump.

- [ ] **Step 8: Commit**

```bash
git add mesh_prep/stl_clipper.py
git commit -m "Add double-click flood-select in Select mode"
```

---

## Self-Review

**1. Spec coverage:**
- §4 `flood_select` + `_ensure_flood_adjacency` + `_barrier_edge_keys` → Task 1 Steps 3-4. ✓
- §4 cache invalidation (identity + curve count) → `_ensure_flood_adjacency` guard. ✓
- §4 non-triangle fallback `[seed]`, out-of-range `[]`, no-mesh `[]` → tests `..._out_of_range`, `..._no_mesh`, and the `_flood_ok` branch. ✓
- §5 double-click detection, `_select_double_click`, shared `_pick_wall_cell`, Select-mode guard → Task 2 Steps 2-4. ✓
- §5 union with selection; single-click unchanged → `_select_double_click` uses `|=`; `_select_click_pick` behavior preserved. ✓
- §8 tests 1-6 (sphere, no-cut, two-cut, delete-survival, out-of-range) → Task 1 Step 1. ✓ (aorta-specific counts from §8.2 are covered structurally by the sphere partition test; no in-repo aorta fixture, and `.stl` is git-ignored.)
- Constraint: no change to cut/undo/lasso → only additive methods + one slim refactor of `_select_click_pick`. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the exact command and expected output. ✓

**3. Type consistency:** `flood_select(seed_cell) -> list[int]` used identically in tests and `_select_double_click`. `_pick_wall_cell(pos) -> Optional[int]` consumed by both click handlers. `_barrier_edge_keys(mesh) -> set[int]` consumed by `_ensure_flood_adjacency`. Cache field names identical across init and methods. ✓
