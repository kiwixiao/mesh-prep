"""Smoothing near a cut must not dissolve the feature-curve flood barrier.

The barrier is matched by coordinates; smoothing moved the ring vertices and
the same flood seed then selected the entire surface (review finding 10).
smooth_cells now rides the curves along with the moved vertices.
"""
import numpy as np
import pyvista as pv

from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def test_flood_barrier_survives_smoothing():
    eng = STLClipperEngine()
    m = pv.Cylinder(resolution=60, capping=True).clean().triangulate()
    m.cell_data[PATCH_ID] = np.zeros(m.n_cells, dtype=np.int64)
    eng.original_mesh = m
    eng._trim_history.clear()
    eng.patch_names = {}
    eng._next_patch_id = 1
    eng._patch_normals = {}
    eng.cut_by_plane((0, 0, 0), (1, 0, 0))
    centers = eng.current_mesh.cell_centers().points
    seed = int(np.argmin(centers[:, 0]))                  # far -x side
    before = eng.flood_select(seed)
    n = eng.current_mesh.n_cells
    assert 0 < len(before) < n                            # bounded by the cut
    # smooth every cell touching the cut ring (a normal cleanup step)
    ring = [i for i, c in enumerate(centers) if abs(c[0]) < 0.25]
    assert ring
    eng.smooth_cells(ring, iterations=2, relaxation=0.5)
    after = eng.flood_select(int(np.argmin(
        eng.current_mesh.cell_centers().points[:, 0])))
    assert len(after) < eng.current_mesh.n_cells          # still bounded
    assert len(eng.drawable_feature_curves()) == 1        # curve still drawn
