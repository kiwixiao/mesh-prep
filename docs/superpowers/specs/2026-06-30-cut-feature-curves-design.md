# Sub-feature A — Cut (not cap) + Feature Curves — Design Spec

- **Date:** 2026-06-30
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)
- **Part of:** the cut→delete→fill→name surface-prep redesign (replaces parametric clip+cap). This is **sub-feature A of A–D**.

## 1. Roadmap context

The new surface-prep workflow replaces the current parametric clip+cap with: **cut the surface (feature curve) → flood-select a region bounded by feature curves → delete → open profiles appear → fill each profile → name the patch.** Built as a dependency chain:

| # | Sub-feature | Status |
|---|---|---|
| **A** | **Cut (not cap) + feature curves** | **this spec** |
| B | Region select bounded by feature curves (double-click flood-fill) + delete | later |
| C | Open-profile detection + list + select each | later |
| D | Fill profile → named patch + rename; bridge into OpenFOAM export; retire clip+cap | later |

The existing parametric clip+cap path stays in place and functional through A–C (export still depends on it); it is retired in D once the fill→patch path feeds export.

## 2. Goal (sub-feature A)

Add a **Cut** operation: a plane (and box) splits the active surface (`original_mesh`) along the intersection but **keeps it one connected, still-closed surface** — the cut becomes an internal **feature curve**, not an open boundary. Nothing is deleted. The cut curve is recorded and drawn so the user sees it. Mutates the base mesh and is undoable via the shared history.

## 3. Requirements (decided in brainstorming)

| # | Decision |
|---|----------|
| Cut keeps surface closed | After a cut the surface is still connected/manifold (no new open boundary edges); the cut is a **feature curve**, not an open profile. Verified: clip→`merge(merge_points=True)` gives 0 new open edges. |
| Feature curve | Recorded as polyline geometry (the plane∩surface intersection); coordinate-based so it survives later cell deletions. Displayed as a highlighted curve. |
| No cap | A cut does **not** generate a cap/patch (unlike the old clip). |
| Plane + box | `cut_by_plane(origin, normal)`; `cut_by_box(...)` follows the same clip-both-sides + merge pattern. |
| Undo | A cut mutates `original_mesh` and pushes the shared `_trim_history` (maxlen 10); `Ctrl+Z` (`undo_trim`) restores the pre-cut mesh **and** drops the feature curve it added. |
| Coexistence | The old clip+cap (`ClipDefinition`/`recompute_all`/cap export) is untouched in A; Cut is a new, separate action operating on the base mesh. |

## 4. Engine interface (headless, unit-tested)

`STLClipperEngine`:

- `cut_by_plane(origin, normal) -> pv.PolyData | None` — split the surface along the plane, keep it connected, record the cut curve. Returns the new `recompute_all()` (the displayed wall), or `None` on a no-op (no mesh, or the plane misses the surface so no cut is produced).
  ```python
  def cut_by_plane(self, origin, normal):
      if self.original_mesh is None:
          return None
      a, b = self.original_mesh.clip(normal, origin=origin, return_clipped=True)
      if a.n_cells == 0 or b.n_cells == 0:        # plane missed -> no cut
          return None
      cut_curve = a.extract_feature_edges(
          boundary_edges=True, feature_edges=False,
          manifold_edges=False, non_manifold_edges=False)
      self._trim_history.append(self.original_mesh.copy())
      self.original_mesh = a.merge(b, merge_points=True)
      self._feature_curves.append(cut_curve)
      return self.recompute_all()
  ```
- `cut_by_box(box_planes, origin, normal) -> pv.PolyData | None` — same pattern using a box region (clip the inside-box and outside-box halves, merge with `merge_points=True`, record the boundary curve). The exact pyvista box-clip both-sides call is verified during the plan (the existing clip code already builds `box_planes`).
- Feature-curve storage: `self._feature_curves: list[pv.PolyData]` (init `[]` in `__init__`). Each entry is the polyline of one cut.
- **Undo bookkeeping (concrete):** the shared history becomes feature-curve-aware. Each history entry stores a snapshot **pair** `(original_mesh.copy(), list(self._feature_curves))` captured *before* the mutation. `undo_trim` restores both `original_mesh` and `_feature_curves` from the popped pair. A small `_push_history()` helper captures the pair; **cut, trim, smooth, and delete all push via it** (trim/smooth/delete snapshot the unchanged `_feature_curves`, so their undo leaves feature curves intact; cut's snapshot is pre-append, so undoing a cut drops exactly its curve). This is the only change to the existing trim/smooth/delete undo, and it's behavior-preserving for them (verified by their existing tests still passing).

Verified APIs (pyvista 0.46.5): `PolyData.clip(normal, origin, return_clipped=True)` returns both halves; `merge(other, merge_points=True)` conforms them into one closed surface (0 new open edges on a sphere test); `extract_feature_edges(boundary_edges=True, ...)` returns the cut polyline.

## 5. App (GUI) layer

- **Define the cut:** reuse the existing interactive plane/box positioning widget (the same UI used to define a clip today). A new **"✂ Cut"** button reads the current plane (origin+normal) or box and calls `engine.cut_by_plane` / `cut_by_box`.
- **Draw feature curves:** `_refresh_display` adds each `engine._feature_curves` polyline as a highlighted line actor (e.g. bright cyan, `line_width≈4`, `name=f"feature_curve_{i}"`, `reset_camera=False`). They render on the closed surface so the user sees every cut.
- **Undo:** `Ctrl+Z` already calls `undo_trim`; after restoring the mesh it must also restore `_feature_curves` to the pre-cut state, then `_refresh_display`.
- Selection/edit buttons: unchanged in A (still gated by the old clips-guard; the new post-cut editing is sub-feature B).

## 6. Data flow

```
Cut press → engine.cut_by_plane(origin, normal):
   a,b = original_mesh.clip(..., return_clipped=True)
   record cut_curve = boundary edges of a
   original_mesh = a.merge(b, merge_points=True)   # still closed
   push _trim_history (mesh + feature-curve count)
   append cut_curve to _feature_curves
   → recompute_all → _refresh_display (draws surface + feature curves)
Ctrl+Z → undo_trim restores mesh + _feature_curves → _refresh_display
```

## 7. Edge cases

- Plane/box misses the surface (one half empty) → `None`, no-op, no history push, no feature curve.
- Cut exactly on existing geometry → still produces a (possibly degenerate) curve; acceptable, undoable.
- Multiple cuts → multiple feature curves accumulate in `_feature_curves`; each undo removes the last.
- Non-triangle input → `clip` triangulates as needed; the result is a surface (`recompute_all`/display already assume triangles for adjacency, which is fine since clip outputs triangles).

## 8. Testing (headless, pytest)

- `cut_by_plane` on a closed sphere through the center → result has **0 open boundary edges** (still closed), `n_cells` increased (split triangles), `len(_feature_curves) == 1` with > 0 segments, and the curve's points lie on the cut plane (within tolerance).
- `undo_trim` after a cut restores the original `n_cells` and `len(_feature_curves) == 0`.
- A plane that misses the mesh returns `None`, pushes no history, adds no feature curve.

## 9. Scope

**A (this spec):** `cut_by_plane`/`cut_by_box`, feature-curve recording + display, shared undo, "✂ Cut" button reusing the plane/box widget. Surface stays closed.

**Out of A:** flood-fill selection bounded by feature curves, delete-region (B); open-profile detection/list (C); fill→named patch + export bridge + clip retirement (D). No change to the existing clip+cap or export in A.

## 10. Integration anchors

- Engine: `recompute_all` `stl_clipper.py:443`; `_trim_history`/`undo_trim` (`:660`-ish); `ClipDefinition` `:200` (untouched); existing `geometry_quality` boundary-edge extraction `:702` (pattern reference).
- App: the existing clip plane/box positioning widget + its "add clip" button (reuse for "Cut"); `_refresh_display` (add feature-curve line actors, `reset_camera=False`); `Ctrl+Z`/`_undo_trim`.
- Verified pyvista APIs: `clip(return_clipped=True)`, `merge(merge_points=True)`, `extract_feature_edges(boundary_edges=True)`.
