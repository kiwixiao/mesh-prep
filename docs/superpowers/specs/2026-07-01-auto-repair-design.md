# Sub-feature R — One-click Auto-repair — Design Spec

- **Date:** 2026-07-01
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)
- **Relation:** A convenience wrapper over the existing Mesh Repair tab. Independent of A–D.

## 1. Goal

Add a single **Auto-repair** button that runs the two existing repairs (clean + fix-normals) in one click and reports a **before → after health summary** (non-manifold edges, open boundary edges, disconnected pieces, faces). It is a convenience over the existing separate "Clean Mesh" and "Fix Normals" buttons.

## 2. Context / what already exists

The Mesh Repair tab (`_build_repair_tab`) already has:
- **Clean Mesh** → `engine.repair_clean()` = `original_mesh.clean()` (merge duplicate points, drop degenerate triangles).
- **Fix Normals** → `engine.repair_normals()` = `original_mesh.compute_normals(cell_normals=False, point_normals=True, split_vertices=False, consistent_normals=True, auto_orient_normals=False)`.
- Both call `_apply_repair(mesh)` (replaces `original_mesh`, clears `clips`, resets `_wall_mesh`) and set `_lbl_repair_status`.

R adds a combined one-click path plus a health summary; it does not modify the two existing buttons.

## 3. Decision (dependency)

**pyvista-only, no new dependency.** `pymeshfix`/`trimesh`/`pymeshlab`/`open3d` are not installed. pyvista/vtk have **no true non-manifold repair**, so Auto-repair is **best-effort**: `clean()` fixes duplicate-point/degenerate issues and `compute_normals` makes winding consistent; non-manifold topology may remain. The before→after summary makes this transparent (the user sees non-manifold counts may not drop). This matches the stated intent: "we will not rely on it, just want it there."

## 4. Engine interface (headless, unit-tested)

`STLClipperEngine`:

```python
def _health_stats(self):
    """Mesh health snapshot for the repair summary."""
    m = self.original_mesh
    nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=True).n_cells
    op = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=False).n_cells
    return {"cells": m.n_cells, "nm": nm, "open": op, "pieces": len(self.detect_pieces())}

def auto_repair(self):
    """Clean + fix-normals in one pass; return a before→after health summary.
    Best-effort (pyvista has no true non-manifold repair). Replaces original_mesh
    and clears clips via _apply_repair (same as the existing repair buttons)."""
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

`compute_normals` params are copied verbatim from the existing `repair_normals` so behavior matches the standalone button.

## 5. App (GUI) layer

In `_build_repair_tab`, add a button above the existing `_lbl_repair_status`:
```python
self._btn_auto_repair = QPushButton("🔧 Auto-repair")
self._btn_auto_repair.setToolTip("Clean + fix normals in one pass, with a before/after summary")
self._btn_auto_repair.clicked.connect(self._on_auto_repair)
tab3.addWidget(self._btn_auto_repair)
```
Handler (mirrors `_on_repair_clean`):
```python
def _on_auto_repair(self):
    msg = self.engine.auto_repair()
    self._lbl_repair_status.setText(msg)
    self._refresh_display()
```
`_refresh_display` already refreshes the object tree (identity guard sees the new mesh), so the tree's health categories update after a repair.

## 6. Data flow

```
click "🔧 Auto-repair" → engine.auto_repair(): before = _health_stats()
    → clean() → compute_normals(...) → _apply_repair (replace mesh, clear clips)
    → after = _health_stats() → return summary
→ _lbl_repair_status = summary → _refresh_display (wall + object tree refreshed)
```

## 7. Edge cases

- No mesh → `"No mesh loaded."`, no-op.
- `clean()` can expose/create non-manifold edges when merging coincident duplicate geometry; the summary reports this honestly (non-manifold may rise). Not hidden.
- Auto-repair after cuts/fills: `_apply_repair` clears `clips` (as the existing buttons do) but leaves `_feature_curves`/`filled_patches`; since `clean()` changes points, those may be stale afterward. Repair is intended as a pre-editing step; this matches the existing repair buttons' behavior and is out of scope to change here.

## 8. Testing (headless pytest, `tests/test_selection.py`)

- `auto_repair` on a `_sphere_engine` returns a string starting with `"Auto-repair — "`; `engine.original_mesh` is not None afterward and is a new object (repaired).
- `auto_repair` on a fresh `STLClipperEngine` (no mesh) returns `"No mesh loaded."`.
- `_health_stats` on a clean sphere returns a dict whose keys are exactly `{"cells", "nm", "open", "pieces"}`, with `pieces == 1` and `cells == n_cells`.

## 9. Scope

**R:** `_health_stats`, `auto_repair`, the `🔧 Auto-repair` button + `_on_auto_repair`. **Out of R:** changing the existing Clean/Fix-Normals buttons, adding a real non-manifold repair (needs a new dependency), clearing feature curves/fills on repair.

## 10. Integration anchors (verified 2026-07-01)

- `_apply_repair` `stl_clipper.py:1030`; `repair_clean` `:1036`; `repair_normals` `:1046` (normals params to copy).
- `_build_repair_tab` `:2477` (button placement, above `_lbl_repair_status` `:2499`); `_on_repair_clean` `:2702` (handler pattern).
- `detect_pieces` (sub-feature C), `extract_feature_edges` (pyvista).
