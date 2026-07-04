"""Task 1 spike: verify patch_id labels survive the VTK filters the
single-current-mesh design relies on, and that _carry_labels re-attaches them
when a filter drops them. Engine-level, no Qt/ParaView needed.
"""

import numpy as np
import pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def _labeled_sphere():
    m = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    ids = np.zeros(m.n_cells, dtype=np.int64)
    ids[: m.n_cells // 3] = 1                      # label a third as patch 1
    m.cell_data[PATCH_ID] = ids
    return m


def test_carry_labels_preserves_existing():
    m = _labeled_sphere()
    same = STLClipperEngine._carry_labels(m.copy(), m)
    assert PATCH_ID in same.cell_data
    assert set(np.unique(same.cell_data[PATCH_ID])) == {0, 1}


def test_carry_labels_remaps_when_filter_drops_them():
    m = _labeled_sphere()
    stripped = m.copy()
    del stripped.cell_data[PATCH_ID]               # simulate a filter that dropped labels
    fixed = STLClipperEngine._carry_labels(stripped, m)
    assert PATCH_ID in fixed.cell_data
    # nearest-cell remap recovers the original split (identical topology here)
    assert set(np.unique(fixed.cell_data[PATCH_ID])) == {0, 1}


def test_plane_clipper_label_propagation_raw():
    """Diagnostic: does vtkPolyDataPlaneClipper preserve cell data on its own?
    Records the raw behavior (no _carry_labels) so the plan's later tasks know
    whether they must wrap clip_with_plane."""
    m = _labeled_sphere()
    raw = STLClipperEngine.clip_with_plane(m, (0, 0, 0), (0, 0, 1))
    preserved = (PATCH_ID in raw.cell_data
                 and len(raw.cell_data[PATCH_ID]) == raw.n_cells)
    print(f"\n[SPIKE] vtkPolyDataPlaneClipper preserves {PATCH_ID}: {preserved}")
    # Either way, _carry_labels must yield a valid labeling:
    fixed = STLClipperEngine._carry_labels(raw, m)
    assert PATCH_ID in fixed.cell_data
    assert len(fixed.cell_data[PATCH_ID]) == fixed.n_cells


def test_labels_survive_merge():
    m = _labeled_sphere()
    cap = pv.Disc(center=(0, 0, 1), inner=0.0, outer=0.3).triangulate()
    cap.cell_data[PATCH_ID] = np.full(cap.n_cells, 2, dtype=np.int64)
    merged = m.merge(cap, merge_points=False)
    assert PATCH_ID in merged.cell_data
    assert set(np.unique(merged.cell_data[PATCH_ID])) == {0, 1, 2}
