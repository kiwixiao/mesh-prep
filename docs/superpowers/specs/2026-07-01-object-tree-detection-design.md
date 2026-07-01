# Sub-feature C — Object Tree Panel + Surface-Entity Detection — Design Spec

- **Date:** 2026-07-01
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)
- **Part of:** the cut→delete→fill→name surface-prep redesign. Follows A (cut+feature curves) and B (flood-select+delete).

## 1. Roadmap context

| # | Sub-feature | Status |
|---|-------------|--------|
| A | Cut (not cap) + feature curves | done |
| B | Double-click flood-select bounded by feature curves + delete | done |
| **C** | **Object-tree panel + entity detection (open profiles, non-manifold edges, disconnected pieces)** | **this spec** |
| D | Fill an open profile → named patch | later (targets a tree-selected profile) |
| R | Auto mesh-repair button (normals + non-manifold) | later, independent |

C is the **container infrastructure**: a ParaView-style object tree that lists detected surface entities and highlights them on click. D and R hang off it; adding new entity types later never touches the operations code.

## 2. Goal (sub-feature C)

Add an **Object Tree** panel on the **left** of the viewer that auto-detects and lists three kinds of surface entities, refreshing after every edit. Clicking an entity highlights it in the 3D view (no camera move). Disconnected pieces additionally load into the face selection so the existing **Delete faces** removes them — the workflow for cleaning tiny stray fragments you can't see.

## 3. Decisions (from brainstorming)

| # | Decision |
|---|----------|
| Layout | Object Tree left, 3D viewer center, existing control tabs right (the tabs are the "operations" side). ParaView-style. |
| Categories | Three, all now: **Open Profiles**, **Non-manifold edges**, **Disconnected pieces**. |
| Refresh | **Automatic** after every topology change (delete / cut / undo / load), via `_refresh_display`. Detection is one cheap pyvista call each; an identity guard skips recompute on view-only refreshes. |
| Click | Draw a dedicated `"tree_highlight"` actor (cleared on next click / category-header click), **no camera reset**. Profiles → orange; non-manifold → red; pieces → highlight **and** set `_selection` to the piece's cells. |
| Scope guard | C does **not** reorganize the existing right panel, add Fill/Repair, or rename entities. |

## 4. Engine interface (headless, unit-tested)

Three pure detection methods on `STLClipperEngine`, plus one private helper. None mutate state.

```python
def detect_open_profiles(self):
    """List of boundary-edge loops (open profiles / holes), one pv.PolyData
    per connected loop. [] if watertight or no mesh."""
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
    """List of connected components; each entry is a sorted list of cell ids
    into original_mesh. [] if no mesh. Single watertight body -> one entry."""
    m = self.original_mesh
    if m is None:
        return []
    conn = m.connectivity('all')
    rid = np.asarray(conn['RegionId'])
    return [sorted(int(c) for c in np.nonzero(rid == r)[0]) for r in np.unique(rid)]

def _split_edge_groups(self, edges):
    """Split an edge PolyData into connected groups; return one geometry per
    group suitable for rendering (line cells preserved). [] if empty."""
    if edges is None or edges.n_cells == 0:
        return []
    conn = edges.connectivity('all')
    rid = np.asarray(conn['RegionId'])
    return [conn.extract_cells(np.nonzero(rid == r)[0]).extract_surface()
            for r in np.unique(rid)]
```

Verified pyvista 0.46.5 APIs / behavior:
- `extract_feature_edges(boundary_edges=True, ...)` → boundary loops; `(non_manifold_edges=True)` → non-manifold edges.
- `PolyData.connectivity('all')` labels every cell with `RegionId` **without reordering cells**, so `RegionId==r` indices map back to `original_mesh` cell ids (validated: body+stray → pieces `[720, 1]`, stray at id 0, deleting it leaves one piece).
- Open-profile split validated: cut sphere+delete → 1 loop (60 edges); cylinder 2-cut+delete band → 2 loops (40+40).

**Plan must confirm** `conn.extract_cells(ids).extract_surface()` yields a PolyData whose `lines` render (tube/line); if `extract_surface` drops the line cells, fall back to rendering the `extract_cells` UnstructuredGrid directly with `line_width`. This is a rendering-detail contingency, not a detection risk.

## 5. App (GUI) layer

**Panel placement** — modify the splitter (currently `plotter | _tab_widget`, `stl_clipper.py:1868`):
```python
self._object_tree = QTreeWidget()
self._object_tree.setHeaderLabel("Objects")
self._object_tree.itemClicked.connect(self._on_tree_item_clicked)
splitter = QSplitter(Qt.Horizontal, central)
splitter.addWidget(self._object_tree)          # left
splitter.addWidget(self.plotter.interactor)    # center
splitter.addWidget(self._tab_widget)           # right
splitter.setStretchFactor(0, 1)
splitter.setStretchFactor(1, 4)
splitter.setStretchFactor(2, 1)
splitter.setCollapsible(0, True)
splitter.setCollapsible(1, False)
splitter.setCollapsible(2, False)
```

**Tree build** — `_refresh_object_tree()`:
- Identity guard: if `self.engine.original_mesh is self._tree_mesh` (last built), return (skips recompute on view-only refreshes). Store `self._tree_mesh = self.engine.original_mesh` on rebuild.
- Rebuild three top-level category nodes; under each, one child per detected entity, labeled with a count:
  - `Open Profile {i} ({n} edges)`
  - `Non-manifold group {i} ({n} edges)`
  - `Piece {i} ({n} faces)`
- Each child stores its entity via `item.setData(0, Qt.UserRole, ("profile"|"nonmanifold"|"piece", index))`; the engine geometry is looked up fresh from the detect-lists cached on the app during the rebuild (`self._tree_profiles`, `self._tree_nonmanifold`, `self._tree_pieces`).
- Empty categories show a disabled `(none)` child.

**Hook** — call `self._refresh_object_tree()` at the end of `_refresh_display` (`stl_clipper.py:3934`); the identity guard makes view-only refreshes free. This single choke point covers delete, cut, undo, and load without touching each handler.

**Click** — `_on_tree_item_clicked(item, column)`:
- Read `(kind, index)` from `item.data(0, Qt.UserRole)`; if `None` (a category header or `(none)`), clear the highlight and return.
- Remove any existing `"tree_highlight"` actor (`reset_camera=False`, no camera move).
- `profile` → `add_mesh(self._tree_profiles[index], color="orange", line_width=6, name="tree_highlight", reset_camera=False)`.
- `nonmanifold` → same with `color="red"`.
- `piece` → `self._selection = set(self._tree_pieces[index])`; `self._refresh_selection_highlight()`; also add a `"tree_highlight"` marker is unnecessary since the selection highlight already shows it. Update `_update_button_states()` so Delete is enabled.
- Status bar message naming the clicked entity and its size.

## 6. Data flow

```
edit (delete/cut/undo/load) → _refresh_display → _refresh_object_tree (identity-guarded)
    → engine.detect_open_profiles / detect_nonmanifold_edges / detect_pieces
    → rebuild 3-category tree with counts
click child →
    profile/nonmanifold → draw "tree_highlight" actor (orange/red), no camera move
    piece → _selection = piece cells → _refresh_selection_highlight → Delete faces removes it
```

## 7. Edge cases

- Watertight single body → Open Profiles: (none), Non-manifold: (none), Disconnected pieces: 1. No error.
- No mesh loaded → all categories (none); no crash.
- Clicking a category header or `(none)` → clears the highlight.
- View-only refresh (opacity / mesh-edges toggle) → identity guard skips detection; tree unchanged.
- `_refresh_display` clears all actors including `"tree_highlight"`; the next click re-highlights. Acceptable.

## 8. Testing (headless pytest, `tests/test_selection.py`)

- `detect_open_profiles`: cut sphere + delete one side → len 1; cylinder 2-cut + delete band → len 2; watertight sphere → len 0; no mesh → [].
- `detect_pieces`: body + stray triangle → len 2 with sizes `[720, 1]` (sorted); single sphere → len 1 covering all cells; no mesh → []. Deleting the smallest piece's cells via `delete_cells` reduces pieces to 1.
- `detect_nonmanifold_edges`: a constructed non-manifold mesh (three triangles sharing one edge) → len 1; clean sphere → len 0.
- Piece cell ids returned by `detect_pieces` are valid indices into `original_mesh` (0 ≤ id < n_cells) and partition all cells (union == all, disjoint).

## 9. Scope

**C:** engine `detect_open_profiles` / `detect_nonmanifold_edges` / `detect_pieces` / `_split_edge_groups`; left `QTreeWidget` panel wired into the splitter; `_refresh_object_tree` (identity-guarded, hooked in `_refresh_display`); `_on_tree_item_clicked` highlight (+ piece→selection). Auto-refresh after edits.

**Out of C:** Fill (D), auto-repair (R), entity renaming, right-panel reorganization, non-manifold *repair* (only detection/highlight here).

## 10. Integration anchors (verified 2026-07-01)

- Engine: `original_mesh`; `delete_cells` `stl_clipper.py:672`; existing `extract_feature_edges` boundary usage (pattern) in `_n_open`/geometry-quality.
- App: splitter `:1868`; `_refresh_display` `:3934` (hook point, single choke); `_refresh_selection_highlight` `:3766`; `_selection` and `_update_button_states` `:2562`; existing `QTreeWidget` import needed (add to the PyQt5 imports near `:40`); existing Mesh Repair tab `_build_repair_tab` (future home for R).
- Verified APIs: `extract_feature_edges(boundary_edges|non_manifold_edges=True)`, `PolyData.connectivity('all')` + `RegionId`, `extract_cells(ids)`.
