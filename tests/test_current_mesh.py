"""Single-current-mesh migration tests (engine-level, no Qt)."""

import numpy as np
import pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def test_load_stl_initializes_current_mesh_labels(tmp_path):
    p = tmp_path / "s.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(p))
    eng = STLClipperEngine()
    eng.load_stl(str(p))
    assert eng.current_mesh is not None
    assert PATCH_ID in eng.current_mesh.cell_data
    assert np.all(eng.current_mesh.cell_data[PATCH_ID] == 0)   # all wall initially
    assert eng.patch_names == {}
    assert eng._next_patch_id == 1
