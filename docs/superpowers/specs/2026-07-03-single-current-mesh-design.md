# Single Current Mesh — Design Spec

**Date:** 2026-07-03
**Status:** Approved for planning
**Scope:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`), export path, object tree.

## Goal

Make the editor behave as a **single, compounding current mesh**:

- There is exactly one surface, "the current mesh."
- Every operation (clip, cut, delete, trim, fill, smooth, repair) mutates the current mesh; its result is the input to the next operation. Edits stack.
- Every panel (object tree, face counts, highlights, geometry status) always reflects the current mesh.
- Undo (Ctrl+Z) steps back exactly one operation.

This replaces today's two-mesh model and its confusing consequence: after "Add plane clip" the object tree still analyzed the untouched base STL.

## Current architecture (what exists today)

`STLClipperEngine` holds **two** surfaces plus side lists:

- `original_mesh` — base, editable surface. `detect_open_profiles`/`detect_nonmanifold_edges`/`detect_pieces` all read this. `_refresh_object_tree` reads this. Destructive ops (`trim_by_screen_polygon`, `delete_cells`, `cut_by_plane`, `smooth_cells`, `_apply_repair`) mutate this.
- `_wall_mesh` — non-destructive clip **preview**: `recompute_all()` does `working = original_mesh.copy()`, applies every `ClipDefinition` plane/box, stores into `_wall_mesh`. The 3D viewport draws this via `get_wall_mesh()`.
- `clips: list[ClipDefinition]` — non-destructive clip planes; each may carry a `cap_mesh` (closed) or `None` (open).
- `filled_patches: list[FilledPatch]` — named caps created by `fill_profile`.
- `_feature_curves` — polylines from `cut_by_plane`.
- `_trim_history: deque(maxlen=10)` — undo snapshots of `(original_mesh.copy(), list(_feature_curves))`.

Consequence (the reported bug): "Add plane clip" updates only `_wall_mesh`; `original_mesh` (what the tree reads) is unchanged, so the tree/highlight still shows the pre-clip STL.

Export today builds one STL solid per source: `export_combined_stl` iterates `clips` (closed caps) + `filled_patches` + the wall mesh; pMesh uses solid names directly as patch names.

## Target design

### One mesh + per-face patch labels

`STLClipperEngine` holds a single `current_mesh: pv.PolyData`. Patch identity is a **cell-data label array** on that mesh:

- `current_mesh.cell_data["patch_id"]` — integer per face. `0` = wall (unnamed). `1..N` = named patches.
- `self.patch_names: dict[int, str]` — maps `patch_id → name` (e.g. `{1: "inlet", 2: "outlet_1"}`).

Everything else (detection, tree, display, export, undo) derives from `current_mesh` + `patch_names`. The `original_mesh` / `_wall_mesh` split, the `clips` list, `filled_patches`, and the closed/open cap-kind concept are removed.

### Operations (all mutate `current_mesh`, snapshot-first)

Each op: (1) push undo snapshot, (2) transform `current_mesh`, (3) return the new mesh (or `None` on no-op). The GUI then refreshes all panels from `current_mesh`.

- **Plane clip (`clip_and_name`)** — Option 1 behavior: trim one side of `current_mesh` by the plane, keeping the same side the current `clip_with_plane`/`clip_with_box` logic keeps (side chosen by normal direction; optional box constraint unchanged). Then generate the cap disc at the cut, assign the cap faces a **new** `patch_id`, record the typed name in `patch_names`, and merge the cap into `current_mesh`. Wall faces keep `patch_id` 0; surviving named faces keep their ids (VTK propagates `cell_data` through clip).
- **Cut by plane** — unchanged intent: split in place, keep watertight, record a feature curve for reference. Mutates `current_mesh`; `patch_id` propagates.
- **Delete / Trim** — remove faces from `current_mesh`; `patch_id` of survivors preserved.
- **Fill** — triangulate an open profile into a cap, assign a new `patch_id` + name, merge into `current_mesh`. (Same mechanism as clip's cap; clip = trim+fill in one step.)
- **Smooth** — moves points only; ids/labels intact.
- **Repair (clean / normals / auto)** — `clean()`/`compute_normals()` preserve `cell_data`; resulting mesh becomes `current_mesh`. Repair resets nothing extra because there is no longer a separate base to desync from.

### Object tree & panels

`detect_open_profiles`, `detect_nonmanifold_edges`, `detect_pieces` read `current_mesh`. `_refresh_object_tree` builds from `current_mesh`; its existing identity guard rebuilds automatically because every op returns a new mesh object. "Named patches" in the tree are read from `patch_names` (grouping `current_mesh` faces by `patch_id`). The viewport draws `current_mesh`, coloring faces by `patch_id`.

### Export

Group `current_mesh` faces by `patch_id` into STL solids: `patch_id 0 → "wall"`, `patch_id k → patch_names[k]`. This reproduces today's multi-solid output (inlet / outlet_N / wall) that pMesh consumes — verified equivalent by test against the current exporters.

### Undo

`_push_history` snapshots `current_mesh.copy()` (the `patch_id` array and `patch_names` ride along). `undo_trim` restores it. This also fixes today's latent "filled patches not restored on undo" gap for free, since patches live in the mesh.

## The core risk: label propagation

The one hard part is keeping `patch_id` attached to the right faces as later operations reshape the mesh. Mitigation:

- Always carry `cell_data["patch_id"]` through every filter. VTK/pyvista propagate cell data through `clip`, `threshold`, `extract_cells`, `connectivity`, `triangulate`, `clean`, and point-only moves (smooth). New geometry (caps) is labeled explicitly at creation.
- Add a regression test per operation asserting: (a) face count changes as expected, (b) every surviving/added face has a valid `patch_id`, (c) `patch_names` stays consistent, (d) export round-trips the same solids.
- Where a VTK filter is found to drop cell data, wrap it to re-attach labels by nearest-face mapping (fallback, only if needed).

## Migration plan (incremental, test-first, export never broken)

Each step lands independently with tests green:

1. Introduce `current_mesh` + `patch_id`/`patch_names`; on `load_stl`, set `current_mesh` and `patch_id=0` everywhere. Keep old fields temporarily.
2. Point `detect_*` and `_refresh_object_tree` at `current_mesh`. (Fixes the reported tree bug.)
3. Point the viewport (`get_wall_mesh`→`current_mesh`) and face counts at `current_mesh`.
4. Reimplement `fill_profile` to label+merge into `current_mesh`.
5. Reimplement clip as `clip_and_name` (trim+cap+label+merge); remove `recompute_all`/`ClipDefinition`.
6. Switch export to group-by-`patch_id`; assert byte-equivalent solids vs current exporters on a fixture.
7. Reimplement undo snapshots on `current_mesh`; remove `_apply_repair`'s special reset.
8. Delete dead fields (`original_mesh`, `_wall_mesh`, `clips`, `filled_patches`) and update all call sites.

## Testing

- Extend the existing headless engine suite (`tests/`) — no Qt needed for engine-level ops.
- Per-operation label-propagation tests (above).
- Export-equivalence test: a fixture mesh with 2 named patches exports the same solids under old and new code paths.
- Keep the current 68 tests green throughout; adapt the clip/undo tests to the new API as those steps land.

## Out of scope

- No change to the OpenFOAM case generators (`openfoam_case.py`), Docker workflow, or render layer.
- No change to `create_case.py`'s CLI contract (still reads STL solid names as patch names).
- Non-destructive clip removal is intentionally dropped (replaced by undo).

## Decisions carried into the plan

- Viewport face color derives from `patch_names[patch_id]` via the existing `_color_for_name` (wall = the default wall color).
- Feature curves from `cut_by_plane` stay a **separate overlay list** (as today); they are reference geometry, not exported patches, so they are out of the `patch_id` label model.
