# Delete Selected Faces — Design Spec

- **Date:** 2026-06-30
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** dev
- **Component:** `mesh_prep/stl_clipper.py` (`STLClipperEngine` + `STLClipperApp`)
- **Builds on:** the active face selection (Select/Grow/Smooth) and the trim feature, both on dev.

## 1. Goal

Add a **Delete** button that permanently removes the actively-selected faces from the base mesh — a third consumer of the active selection alongside Grow and Smooth. Completes the cleanup loop: `Shift+lasso → Grow/Smooth → Delete → repeat → clip → export`.

## 2. Requirements (decided in brainstorming)

| # | Decision |
|---|----------|
| Operation | Delete every cell in `self._selection` from `original_mesh` (permanent, through the shared base mesh) |
| Undo | Pushes the shared `_trim_history` (maxlen 10); `Ctrl+Z` (`undo_trim`) restores — same history as trim and smooth |
| Confirmation | **No dialog** — immediate delete; undo covers accidents (consistent with Trim and Smooth) |
| After delete | Clear the selection (those cell ids no longer exist — topology changed) and refresh the display |
| Guard | Disabled when the selection is empty or when `engine.clips` is non-empty (edits target the unclipped base, same as Grow/Smooth) |
| Target mesh | The unclipped base `original_mesh` |

## 3. Engine interface (headless, unit-tested)

`delete_cells(cell_ids) -> pv.PolyData | None` on `STLClipperEngine`, placed after `smooth_cells`. Structurally identical to the trim deletion (`trim_by_screen_polygon`, `stl_clipper.py:494-497`):

```python
def delete_cells(self, cell_ids):
    """Permanently delete the given cells from original_mesh (shared undo).
    Returns the new _wall_mesh, or None on a no-op."""
    if self.original_mesh is None:
        return None
    n = self.original_mesh.n_cells
    ids = {int(c) for c in cell_ids if 0 <= int(c) < n}
    if not ids or len(ids) >= n:        # nothing to do, or would delete everything
        return None
    self._trim_history.append(self.original_mesh.copy())
    keep_ids = np.array([i for i in range(n) if i not in ids], dtype=np.int64)
    self.original_mesh = self.original_mesh.extract_cells(keep_ids).extract_surface()
    return self.recompute_all()
```

Returns `None` (no history push, no change) on: no mesh, empty/all-stale selection, or a selection covering the whole mesh. Otherwise mutates `original_mesh` and returns `recompute_all()`.

## 4. App (GUI) layer

- **Button:** "🗑 Delete faces" (`self._btn_delete`), added next to `self._btn_grow` / `self._btn_smooth` (`stl_clipper.py:1520-1526`); `clicked` → `_on_delete_selection`.
- **Handler `_on_delete_selection`:** guard `if not self._selection or not self._edit_enabled(): return`; call `result = self.engine.delete_cells(self._selection)`; if `result is None`, status "Nothing deleted." and return; else `self._clear_selection()` (topology changed), `self._refresh_display()`, status `f"Deleted faces — wall now {n} faces."`.
- **Enable/disable:** in `_update_button_states` (`:2480`), add `self._btn_delete.setEnabled(can_edit and has_sel)` alongside the grow/smooth lines, under the same `hasattr` guard.
- **Undo:** `Ctrl+Z` already calls `undo_trim`, which pops the shared history — no GUI change needed for undo.

## 5. Data flow

```
Delete press → engine.delete_cells(self._selection)
   → push _trim_history → original_mesh = extract_cells(keep).extract_surface()
   → recompute_all → _refresh_display
   → _clear_selection (ids gone)            (Ctrl+Z → undo_trim restores)
```

## 6. Edge cases

- Empty / all-stale selection → `delete_cells` returns None → status "Nothing deleted.", no history push.
- Selection covers the whole mesh → returns None (refuse to delete everything), status "Nothing deleted." (mirrors trim's whole-mesh guard, but as a no-op rather than raising — the GUI need not handle an exception).
- Clips present → button disabled (guard).
- After delete, `_refresh_display` preserves the camera (the camera-fix already in place); selection highlight is cleared by `_clear_selection`.

## 7. Testing

- **Engine (headless, pytest):** on a known mesh, `delete_cells({a few ids})` reduces `n_cells` by exactly that count, pushes one history entry, and `undo_trim()` restores the original `n_cells`. `delete_cells([])` and a whole-mesh selection both return None with no history push.
- **GUI:** the button (placement, enable/disable with clips/selection, visual result, Ctrl+Z) is verified by the human.

## 8. Scope

**v1:** `delete_cells` engine method + shared undo; "🗑 Delete faces" button consuming the active selection; clears selection after; clips/selection guard; engine tests.

**Out of v1:** confirmation dialog; deleting through active clips; delete-then-fill/cap the hole; per-face click delete.

## 9. Integration anchors

- Engine: `trim_by_screen_polygon` `:471-497` (deletion pattern to mirror), `smooth_cells` `:597` (insert after), `_trim_history`, `recompute_all`, `undo_trim` `:660`.
- App: grow/smooth buttons `:1520-1526`, `_update_button_states` `:2480/2515-2516`, `_on_grow_selection`/`_on_smooth_selection` `:3578/3585`, `_clear_selection`, `_edit_enabled`, `_refresh_display`.
- Verified APIs: `PolyData.extract_cells(ids).extract_surface()` (already used by trim).
