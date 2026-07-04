# Single-Mesh GUI Cutover — Plan (continuation of 2026-07-03-single-current-mesh)

Tasks 1–4 landed (read-side). This plan covers the coupled remainder: the engine
patch model and the GUI that consumes it, in two commits.

## Commit A — Engine: patches are patch_id labels on current_mesh

- `_new_patch_id(name)` allocates ids; `patches_by_id()` groups faces by label;
  `patch_name_for(pid)` (0 → "wall").
- `clip_and_name(name, origin, normal, box_planes_data=None)`: snapshot → trim
  (`clip_with_plane`/`clip_with_box`) → `_generate_cap` → label cap with new id →
  merge into current_mesh. Replaces `add_clip`/`recompute_all` semantics.
- `fill_profile(profile_edges, name)`: snapshot → triangulate cap → label →
  merge into current_mesh. Filling closes the hole, so
  `unfilled_open_profiles()` becomes `detect_open_profiles()`.
- Exports group by patch_id: `export_combined_stl`, `export_separate_stl`,
  `export_clipped_surface_stl` (current_mesh is already the capped surface),
  `export_openfoam`, `export_openfoam_case` (names from patch_names; inlet
  normals from mean inlet-cap cell normal, preserving the old sign convention).
- Undo snapshots `(current_mesh.copy(), feature_curves, dict(patch_names),
  _next_patch_id)`; repair pushes history (repair becomes undoable) and carries
  labels via `_carry_labels`.
- Mutating ops return `self.current_mesh` (recompute_all deleted). Fields
  `_wall_mesh`, `clips`, `filled_patches` and methods `add_clip`,
  `set_cap_kind`, `remove_clip`, `rename_clip`, `recompute_all` deleted;
  `ClipDefinition`/`FilledPatch` deleted when unreferenced.
- Tests updated to the new contract (trim/undo assert current_mesh; repair-undo
  restores pre-repair state; fill/export tests assert labels + solids).

## Commit B — GUI: consume the new model

- Clip confirm → `engine.clip_and_name(...)`.
- Clips panel list → named patches (`named_patches()`); rename edits
  `patch_names[pid]`; remove relabels faces to wall (0) and drops the name;
  cap-kind toggle removed.
- Object tree "Named patches" from `named_patches()`; click highlights
  `patches_by_id()[pid]`.
- `_refresh_display`: wall = faces with pid 0; one actor per named patch
  colored by `_color_for_name`.
- Centerline seeds from patch centroids (classify_patch on names); surface is
  current_mesh (already capped).
- Export handlers + button states + status text read `patch_names`.
- Editing enabled whenever current_mesh exists (ops compound; clips no longer
  lock editing).

## Verification

- Full pytest suite green after each commit; `import mesh_prep.stl_clipper` clean.
- Manual GUI checklist (user): clip trims+names in view+tree; Ctrl+Z reverts;
  fill names a hole; export case → triSurface STL has one solid per patch+wall;
  pMesh run consumes it.
