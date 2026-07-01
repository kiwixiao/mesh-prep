# Sub-feature B — Double-click Flood-Select Region (bounded by feature curves) — Design Spec

- **Date:** 2026-06-30
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)
- **Part of:** the cut→delete→fill→name surface-prep redesign. This is **sub-feature B of A–D**.

## 1. Roadmap context

| # | Sub-feature | Status |
|---|-------------|--------|
| A | Cut (not cap) + feature curves | done (`_feature_curves`, `cut_by_plane`/`cut_by_box`) |
| **B** | **Double-click flood-select a region bounded by feature curves** | **this spec** |
| C | Open-profile detection + list + select each | later |
| D | Fill profile → named patch + export bridge; retire clip+cap | later |

B populates the **active selection** (`_selection`). Deleting the selected region (existing **Delete faces** button) is what later turns a feature curve into an open profile (sub-feature C).

## 2. Goal (sub-feature B)

In **Select mode**, a **double-click** on the surface selects the entire connected surface region under the cursor, **stopping at feature curves** (the edges a Cut created). The region is **added** to the current selection (union, like the lasso). No new UI panel; the existing Delete/Smooth/Grow buttons consume the resulting selection.

## 3. Key evidence (validated headlessly before design)

All confirmed on `AO_Native_001.stl` and a `pv.Sphere`, pyvista 0.46.5:

| Fact | Evidence |
|------|----------|
| A Cut re-triangulates: the feature curve is **real new mesh edges** | plane cut: cells 35062→35570, points +254; all 254 curve points coincide with mesh vertices at distance **0.0** |
| Feature-curve edges form **closed loops** | every barrier vertex has degree 2 |
| Blocking feature-curve edges in the **cell edge-adjacency** cleanly separates the two sides | sphere → 2 components [1600,1600]; aorta plane cut → [26729, 9195] |
| Works for **multiple cuts** | 2 cuts → 3 regions [26346, 8697, 1401] |
| **Survives deletes** via a tolerance guard | after deleting a region, orphaned feature-curve points fail the distance guard and their edges drop (691→306 barriers), remaining regions stay clean |

**Critical implementation detail (source of a long bug during prototyping):** when building per-cell edges as
`np.vstack([tri[:,[0,1]], tri[:,[1,2]], tri[:,[2,0]]])`, the cell owning row *r* is `np.tile(np.arange(n_cells), 3)[r]` — **not** `np.repeat(...)`. Using `repeat` misaligns every edge with the wrong cell and silently produces a broken adjacency that *looks* plausible. The plan's tests must assert separation on a known shape (a sphere) to catch this.

**Rejected alternative (region-label / cell_data provenance):** also works, but requires modifying the already-working `cut_by_plane`/`cut_by_box` to tag `cell_data["_region"]` and re-base offsets per cut. The barrier-edge approach needs **zero changes to the A cut code** and reuses the `_feature_curves` A already stores, so it is preferred.

## 4. Engine interface (headless, unit-tested)

`STLClipperEngine` gains one public method plus two cached private helpers. **No change to `cut_by_plane`/`cut_by_box`/`_feature_curves`.**

```python
def flood_select(self, seed_cell):
    """Return the sorted list of cell ids in the connected surface region
    containing seed_cell, where feature-curve edges act as walls (the flood
    never crosses a cut). Returns [] if seed_cell is out of range or there is
    no mesh; returns [seed_cell] on a non-triangle mesh (no fast path)."""
```

Helpers (mirroring the existing `_ensure_adjacency` cache pattern, identity + curve-count keyed):

- `_ensure_flood_adjacency()` — builds, once per (mesh identity, len(_feature_curves)):
  - `_flood_barrier`: `set[int]` of edge keys `min(u,v) * n_points + max(u,v)` for every feature-curve segment, mapped to **current** mesh point ids.
  - `_flood_adj`: CSR/list cell→cell adjacency over shared edges (share-count exactly 2) **excluding** barrier keys.
  - `_flood_ok`: `False` on a non-triangle mesh (then `flood_select` returns `[seed_cell]`).
- `_barrier_edge_keys(mesh)` — for each curve in `_feature_curves`, map each curve point to a mesh point via `mesh.find_closest_point(point)`, **guarded** by `dist <= 1e-6 * bbox_diagonal`; a segment contributes a barrier edge only if **both** endpoints pass the guard (drops edges whose region was deleted). Returns the edge-key set.

Adjacency construction (the validated, correct form):

```python
tri = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)     # triangle path only
e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
e.sort(axis=1)
keys = e[:, 0] * n_points + e[:, 1]
cell_of = np.tile(np.arange(n_cells), 3)                     # tile, NOT repeat
# group by identical key; for each key with exactly 2 owner cells that is not a
# barrier key, connect the two cells; BFS from seed over that adjacency.
```

Cache invalidation: rebuild when `self.original_mesh is not self._flood_adj_for_mesh` **or** `len(self._feature_curves) != self._flood_adj_n_curves`. (Delete/cut replace `original_mesh`; a new cut also appends a curve — both are caught.)

State added in `__init__`: `self._flood_adj_for_mesh = None`, `self._flood_adj_n_curves = -1`, and the cached arrays.

## 5. App (GUI) layer

- **Double-click detection** in `_SelectLassoStyle._on_release`: a plain click (press+release with < ~3 px movement) records `time.time()` and the pixel. If a second such click lands within **0.4 s** and within **~8 px** of the previous one, it is a double-click → flood; otherwise it is a single-face pick (existing behavior). Manual timing is used rather than VTK repeat-count, which is unreliable through Qt on macOS.
- New `STLClipperApp._select_double_click(pos)`:
  - Same guards as `_select_click_pick` (`_btn_select.isChecked()` and `_edit_enabled()`).
  - Pick the wall cell under `pos` (shared helper `_pick_wall_cell(pos) -> Optional[int]`, extracted from the existing `_select_click_pick` picker code so both single- and double-click use it — DRY).
  - `region = self.engine.flood_select(cid)`; `self._selection |= set(region)`.
  - `self._refresh_selection_highlight()`; status `"Flood-selected {len(region)} faces ({len(self._selection)} total)."`; `self._update_button_states()`.
- Single-click still adds one face. Because both **add** (union), the first click of a double-click (which adds one face inside the region) is harmless — the flood is a superset.
- **Scope:** Select mode only. Trim mode is unchanged. With no cut made yet there are no barriers, so a double-click floods the whole connected surface — safe, no error.

## 6. Data flow

```
double-click at pixel p
  → cid = _pick_wall_cell(p)                     (vtkCellPicker on _wall_actor)
  → region = engine.flood_select(cid):
        _ensure_flood_adjacency()                (barrier = feature-curve edges, cached)
        BFS(cid) over edge-adjacency minus barrier edges
  → _selection |= region → _refresh_selection_highlight
→ (existing) Delete faces → region removed → open profile appears (sub-feature C)
```

## 7. Edge cases

- Seed out of range / no mesh → `[]`. Non-triangle mesh → `[seed_cell]` (STL + clip always emit triangles, so the fast path is the norm).
- No feature curves → empty barrier → floods the whole connected component (safe).
- Double-click that misses the wall → picker returns no cell → no-op.
- Feature-curve points whose region was deleted → dropped by the tolerance guard (validated: 691→306 barriers after a delete), so no spurious walls.
- Multiple cuts → union of all feature-curve edges; each closed loop separates independently (validated: 3 regions after 2 cuts).

## 8. Testing (headless pytest, `tests/test_selection.py`)

1. **Sphere separation (guards the tile/repeat bug):** cut a `pv.Sphere` through the center; `flood_select` from a top-cap cell returns exactly the top half; from a bottom-cap cell the bottom half; the two are disjoint and together cover all cells.
2. **Aorta plane cut:** `flood_select` from a cell on each side returns that side's cell count (9195 / 26729 for the mid-x plane); regions disjoint; union == all cells.
3. **No cut:** `flood_select` on an uncut mesh returns the whole connected component (all cells).
4. **Two cuts:** three regions; `flood_select` from a seed returns one bounded region strictly smaller than the whole mesh.
5. **Survives delete:** after `delete_cells(region_from_flood)`, a `flood_select` on the remaining mesh returns a bounded region (no crash, no whole-mesh leak).
6. **Out-of-range seed** → `[]`.

## 9. Scope

**B (this spec):** `flood_select` + cached barrier/edge-adjacency helpers; double-click detection; `_select_double_click`; shared `_pick_wall_cell` helper. Populates `_selection` (union). Reuses existing Delete/Smooth/Grow.

**Out of B:** open-profile detection/listing (C); fill→named patch + export bridge + clip retirement (D). No change to `cut_by_plane`/`cut_by_box`, `_feature_curves`, undo, or the existing lasso/single-click select.

## 10. Integration anchors (verified line numbers, 2026-06-30)

- Engine: `_ensure_adjacency` `stl_clipper.py:531` (cache pattern to mirror); `grow_cells` `:581`; `cut_by_plane(origin, normal)` `:687`; `_feature_curves` init `:269`; `_push_history` `:743`.
- App: `_SelectLassoStyle._on_release` `:1274` (add double-click); `_select_click_pick` `:3600` (extract `_pick_wall_cell`); `_apply_select` `:3581` (union pattern); `_refresh_selection_highlight` `:3623`; `_edit_enabled` `:3559`; `_wall_actor` set in `_refresh_display` `:3808`.
- Verified pyvista/VTK APIs: `PolyData.find_closest_point(point) -> int`; `vtk.vtkCellPicker` restricted to the wall actor (already used by `_select_click_pick`).
