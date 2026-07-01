# Sub-feature R — One-click Auto-repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single "Auto-repair" button that runs clean + fix-normals in one pass and reports a before→after health summary.

**Architecture:** An engine method `auto_repair()` (reusing the existing `clean()` + `repair_normals`-style `compute_normals`, applied via the existing `_apply_repair`) plus a `_health_stats()` helper; a button in the existing Mesh Repair tab wired to a 3-line handler that reuses the existing status label and refresh.

**Tech Stack:** Python 3.9, pyvista 0.46.5, VTK 9.2.6, PyQt5, numpy. No new dependencies.

## Global Constraints

- No new dependencies. pyvista-only, best-effort (no true non-manifold repair exists in pyvista).
- `auto_repair` uses the SAME `compute_normals` params as the existing `repair_normals` (`cell_normals=False, point_normals=True, split_vertices=False, consistent_normals=True, auto_orient_normals=False`) and applies via the existing `_apply_repair`.
- Do NOT change the existing `repair_clean` / `repair_normals` / `_apply_repair` methods or the existing Clean/Fix-Normals buttons.
- Run tests with `conda run -n mesh-prep pytest`.

---

### Task 1: Engine — `auto_repair` + `_health_stats`

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add two methods after `repair_normals` (~`:1056`).
- Test: `tests/test_selection.py` — append tests.

**Interfaces:**
- Consumes: `original_mesh`, `_apply_repair` (existing), `detect_pieces` (sub-feature C), `extract_feature_edges` (pyvista).
- Produces: `auto_repair(self) -> str`; `_health_stats(self) -> dict` with keys `{"cells", "nm", "open", "pieces"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selection.py` (reuses `_sphere_engine`):

```python
def test_auto_repair_returns_summary_and_replaces_mesh():
    eng = _sphere_engine()
    before_obj = eng.original_mesh
    msg = eng.auto_repair()
    assert msg.startswith("Auto-repair — ")
    assert eng.original_mesh is not None
    assert eng.original_mesh is not before_obj          # replaced with the repaired mesh


def test_auto_repair_no_mesh():
    assert STLClipperEngine().auto_repair() == "No mesh loaded."


def test_health_stats_keys():
    eng = _sphere_engine()
    stats = eng._health_stats()
    assert set(stats.keys()) == {"cells", "nm", "open", "pieces"}
    assert stats["pieces"] == 1
    assert stats["cells"] == eng.original_mesh.n_cells
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k "auto_repair or health_stats" -q`
Expected: FAIL — `AttributeError: 'STLClipperEngine' object has no attribute 'auto_repair'`.

- [ ] **Step 3: Implement the two methods**

Insert after `repair_normals` (~`stl_clipper.py:1056`):

```python
    def _health_stats(self):
        """Mesh health snapshot for the auto-repair summary."""
        m = self.original_mesh
        nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=True).n_cells
        op = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=False).n_cells
        return {"cells": m.n_cells, "nm": nm, "open": op, "pieces": len(self.detect_pieces())}

    def auto_repair(self):
        """Clean + fix-normals in one pass; return a before→after health summary.
        Best-effort (pyvista has no true non-manifold repair). Replaces original_mesh
        and clears clips via _apply_repair, like the existing repair buttons."""
        if self.original_mesh is None:
            return "No mesh loaded."
        before = self._health_stats()
        m = self.original_mesh.clean()
        m = m.compute_normals(cell_normals=False, point_normals=True,
                              split_vertices=False, consistent_normals=True,
                              auto_orient_normals=False)
        self._apply_repair(m)
        after = self._health_stats()
        return ("Auto-repair — "
                f"faces {before['cells']}→{after['cells']}, "
                f"non-manifold {before['nm']}→{after['nm']}, "
                f"open edges {before['open']}→{after['open']}, "
                f"pieces {before['pieces']}→{after['pieces']}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n mesh-prep pytest tests/test_selection.py -k "auto_repair or health_stats" -q`
Expected: PASS — 3 passed.

- [ ] **Step 5: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 54 passed (51 prior + 3 new).

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py tests/test_selection.py
git commit -m "Add auto_repair engine method with before/after health summary"
```

---

### Task 2: GUI — Auto-repair button

**Files:**
- Modify: `mesh_prep/stl_clipper.py` — add a button in `_build_repair_tab` (~`:2477`, above `_lbl_repair_status` ~`:2499`); add `_on_auto_repair` (near `_on_repair_clean` ~`:2702`).

**Interfaces:**
- Consumes: `engine.auto_repair()` (Task 1); existing `self._lbl_repair_status`, `self._refresh_display`, `QPushButton`.
- Produces: GUI behavior only.

- [ ] **Step 1: Add the button**

In `_build_repair_tab`, immediately before the `self._lbl_repair_status = QLabel(...)` line (~`:2499`), add:

```python
        self._btn_auto_repair = QPushButton("🔧 Auto-repair")
        self._btn_auto_repair.setToolTip("Clean + fix normals in one pass, with a before/after summary")
        self._btn_auto_repair.clicked.connect(self._on_auto_repair)
        tab3.addWidget(self._btn_auto_repair)
```

- [ ] **Step 2: Add the handler**

Next to `_on_repair_clean` (~`stl_clipper.py:2702`), add:

```python
    def _on_auto_repair(self):
        msg = self.engine.auto_repair()
        self._lbl_repair_status.setText(msg)
        self._refresh_display()
```

- [ ] **Step 3: Verify the module compiles and imports**

Run: `conda run -n mesh-prep python -c "import mesh_prep.stl_clipper; print('IMPORT_OK')"`
Expected: `IMPORT_OK`.

- [ ] **Step 4: Run the whole suite (no regression)**

Run: `conda run -n mesh-prep pytest tests/ -q`
Expected: PASS — 54 passed.

- [ ] **Step 5: Manual GUI smoke (record result)**

No GUI harness (see `mesh-prep/CLAUDE.md`); verify by running:

```bash
conda run -n mesh-prep mesh-prep /Users/xiaz9n/openfoam/AO_Native_001.stl
```
Open the **Mesh Repair** tab → click **🔧 Auto-repair** → the status label shows `Auto-repair — faces N→M, non-manifold …→…, open edges …→…, pieces …→…`, the viewport redraws, and the object tree's health categories refresh. The existing **Clean Mesh** / **Fix Normals** buttons still work.

- [ ] **Step 6: Commit**

```bash
git add mesh_prep/stl_clipper.py
git commit -m "Add Auto-repair button to the Mesh Repair tab"
```

---

## Self-Review

**1. Spec coverage:**
- §4 `_health_stats` + `auto_repair` (clean + normals with the exact `repair_normals` params, `_apply_repair`, before→after summary, no-mesh message) → Task 1 Step 3. ✓
- §5 button in `_build_repair_tab` + `_on_auto_repair` (status label + refresh) → Task 2 Steps 1-2. ✓
- §8 tests (summary + replaced mesh, no-mesh message, health-stats keys) → Task 1 Step 1. ✓
- Constraints: no new deps, same normals params, existing methods/buttons untouched → honored. ✓

**2. Placeholder scan:** No TBD/TODO; every code step complete; every run step has an exact command + expected output. ✓

**3. Type consistency:** `auto_repair() -> str` consumed in `_on_auto_repair` as `msg` for `setText`. `_health_stats() -> dict{cells,nm,open,pieces}` consumed inside `auto_repair` with those exact keys and asserted in the test. `_btn_auto_repair`/`_on_auto_repair` names consistent. ✓
