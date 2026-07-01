# Sub-feature D — Fill an Open Profile into a Named Patch — Design Spec

- **Date:** 2026-07-01
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)
- **Part of:** the cut→delete→fill→name surface-prep redesign. Follows A (cut), B (flood-select+delete), C (object tree).

## 1. Goal

Add a **Fill** operation: select an open profile in the object tree, press Fill, name it, and the profile's boundary loop is triangulated into a **named cap patch** (like inlet/outlet). The cap is a separate STL `solid` in the OpenFOAM multi-solid export (the wall keeps its opening — the standard snappyHexMesh/cfMesh form). The filled profile moves from **Open Profiles** to a new **Named patches** node in the tree.

## 2. Decisions (from brainstorming)

| # | Decision |
|---|----------|
| Cap representation | The cap is a **separate named solid**; the wall mesh keeps its opening (correct for OpenFOAM multi-solid meshing). The fill does NOT merge into `original_mesh`. |
| Storage | Fills go in a **new `filled_patches` list**, NOT `engine.clips` — because `_edit_enabled()` disables editing whenever `clips` is non-empty, and the user must keep cutting/filling. |
| Triangulation | Reuse `_generate_cap`'s proven technique on the profile's boundary edges: `vtkStripper` (order the loop) → `vtkContourTriangulator` (fill) → `delaunay_2d` fallback. Validated: a detected 60-edge profile → 58-triangle cap that closes the hole (wall+cap merge → 0 open edges). |
| Tree | **Open Profiles** lists `unfilled_open_profiles()` (filled ones drop out, matched by a signature); a new **Named patches** node lists `filled_patches`. |
| Export | Extend `export_combined_stl` to emit each `filled_patches` cap as a named `solid` alongside the wall + any clip caps. |
| Scope guard | No separate rename/remove-patch UI (name is set at Fill), no retiring the old clip+cap path, no undo of fills. |

## 3. Engine interface (headless, unit-tested)

New dataclass (mirroring `ClipDefinition` at `stl_clipper.py:202`):
```python
@dataclass
class FilledPatch:
    name: str
    cap_mesh: pv.PolyData
    signature: tuple            # (n_points, rounded centroid) of the filled boundary loop
    color: tuple = (0.2, 0.6, 0.9)
```

`STLClipperEngine`:
- `self.filled_patches: list[FilledPatch] = []` (init in `__init__`); cleared in `load_stl` next to `self._feature_curves.clear()` (`:300`).
- `@staticmethod _profile_signature(edges) -> tuple` — `(edges.n_points, tuple(np.round(edges.points.mean(axis=0), 6)))`.
- `fill_profile(self, profile_edges, name) -> FilledPatch | None`:
  ```python
  def fill_profile(self, profile_edges, name):
      if profile_edges is None or profile_edges.n_cells == 0:
          return None
      strip = vtk.vtkStripper(); strip.SetInputData(profile_edges); strip.Update()
      tri = vtk.vtkContourTriangulator(); tri.SetInputData(strip.GetOutput()); tri.Update()
      cap = pv.wrap(tri.GetOutput())
      if cap is None or cap.n_cells == 0:
          cap = pv.PolyData(profile_edges.points).delaunay_2d()      # fallback
      if cap is None or cap.n_cells == 0:
          return None
      patch = FilledPatch(name=name, cap_mesh=cap,
                          signature=self._profile_signature(profile_edges),
                          color=_color_for_name(name))
      self.filled_patches.append(patch)
      return patch
  ```
- `unfilled_open_profiles(self) -> list[pv.PolyData]`:
  ```python
  def unfilled_open_profiles(self):
      filled = {fp.signature for fp in self.filled_patches}
      return [p for p in self.detect_open_profiles()
              if self._profile_signature(p) not in filled]
  ```

Verified pyvista/VTK APIs (9.2.6 / 0.46.5): `vtk.vtkStripper`, `vtk.vtkContourTriangulator`, `pv.wrap`, `PolyData.delaunay_2d` — `vtkContourTriangulator` is already used by `_generate_cap`.

## 4. Export bridge (unit-tested)

Extend `export_combined_stl` (`stl_clipper.py:1054`) to write filled-patch caps as named solids. After the existing clips loop and before the wall block:
```python
        for patch in self.filled_patches:
            blocks.append(self._polydata_to_ascii_stl_block(patch.cap_mesh, patch.name, scale_factor))
```
Both `export_openfoam` and `export_openfoam_case` route through `export_combined_stl`, so both are covered. Names are unique across wall / clips / filled_patches (enforced at the Fill prompt).

## 5. App (GUI) layer

- **Fill button** — `self._btn_fill = QPushButton("🩹 Fill profile → patch")` in the edit area near `_btn_delete` (`:1837`); enabled when a mesh is loaded.
- **Active profile** — in `_on_tree_item_clicked`, a `kind == "profile"` click stores `self._active_profile_index = index` (into `self._tree_profiles`, which now holds `unfilled_open_profiles()`); other clicks leave it, but Fill re-validates the index against the current list.
- **`_on_fill_profile()`**:
  - Guard: mesh present; `_active_profile_index` valid for `self._tree_profiles`; else status hint.
  - `QInputDialog.getText` for the name (default `inlet`, then `outlet_1`, `outlet_2`, … like `_on_confirm_clip`); reject empty, `"wall"`, and any name already used by a clip or a filled patch.
  - `patch = self.engine.fill_profile(self._tree_profiles[idx], name)`; if `None` → status "Could not fill this profile."
  - `self._refresh_display()` (draws caps) and `self._refresh_object_tree()` (profile → Named patches); status "Filled patch '{name}' ({cap.n_cells} faces)."
- **Display** — `_refresh_display` draws each `engine.filled_patches` cap: `add_mesh(patch.cap_mesh, color=patch.color, name=f"patch_{i}", reset_camera=False)`.
- **Tree** — `_refresh_object_tree` gains a third data source and a fourth guard key:
  - Guard now also keys on the patch count: `if mesh is self._tree_mesh and len(engine.filled_patches) == self._tree_n_patches: return` (so a Fill, which doesn't change the mesh identity, still triggers a rebuild). Init `self._tree_n_patches = -1`.
  - **Open Profiles** category uses `engine.unfilled_open_profiles()` (assigned to `self._tree_profiles`).
  - New **Named patches** category from `engine.filled_patches`, child label `"{name} ({cap n_cells} faces)"`, data `("patch", i)`.
  - `_on_tree_item_clicked` `kind == "patch"` → highlight `engine.filled_patches[i].cap_mesh` green as `"tree_highlight"` (no camera move), status with the name.

## 6. Data flow

```
click an Open Profile in the tree → _active_profile_index set
press Fill → name prompt →
  engine.fill_profile(profile_edges, name): vtkStripper → vtkContourTriangulator → FilledPatch → filled_patches.append
  → _refresh_display (cap drawn) → _refresh_object_tree (profile leaves Open Profiles, appears under Named patches)
export → export_combined_stl: wall solid + each filled patch as a named solid
```

## 7. Edge cases

- Fill with no profile selected / stale index → status hint, no-op.
- Duplicate name / `"wall"` / empty → rejected at the prompt.
- Un-triangulable profile (both contour and delaunay empty) → `fill_profile` returns `None` → "Could not fill this profile."
- After a fill, a later edit that changes that loop → its signature stops matching → it reappears under Open Profiles and the stale cap remains (a separate solid); undo does not remove fills. Acceptable for MVP; documented.
- No filled patches → Named patches shows `(none)` (same convention as the other categories).

## 8. Testing (headless pytest, `tests/test_selection.py`)

- `fill_profile`: cut+delete a sphere → `detect_open_profiles()[0]` → `fill_profile(profile, "inlet")` → returns a `FilledPatch`; `cap_mesh.n_cells > 0`; wall+cap merge has 0 open boundary edges; `engine.filled_patches` has one entry named `"inlet"`.
- `unfilled_open_profiles`: 1 before the fill, 0 after.
- `fill_profile` no-op: `None`/empty edges → `None`, no patch appended.
- `load_stl` resets `filled_patches` to empty.
- Export bridge: after a fill, `export_combined_stl(tmp)` writes a file whose text contains `solid inlet` and `solid wall`.

## 9. Scope

**D:** `FilledPatch`, `fill_profile`, `unfilled_open_profiles`, `_profile_signature`, `filled_patches` (+ load reset), export bridge in `export_combined_stl`, Fill button + active-profile tracking + fill flow, cap display, Named-patches tree node (+ Open-Profiles uses unfilled). **Out of D:** rename/remove-patch UI, retiring clip+cap, undo of fills, non-manifold repair.

## 10. Integration anchors (verified 2026-07-01)

- `ClipDefinition` dataclass `stl_clipper.py:202` (mirror for `FilledPatch`); `_color_for_name` `:71`.
- `load_stl` reset `:300` (`_feature_curves.clear()` — add `filled_patches.clear()`); `_generate_cap` `:326` (technique reference: stripper→contour triangulator→delaunay fallback).
- `export_combined_stl` `:1054`; `_polydata_to_ascii_stl_block(mesh, name, scale)` `:1017`.
- C tree: `_refresh_object_tree` (identity guard + three categories), `_on_tree_item_clicked` (kind dispatch), `_tree_profiles` cache, `_tree_mesh` sentinel; edit buttons `_btn_grow` `:1829`, `_btn_delete` `:1837`; `_edit_enabled` `:3858`.
