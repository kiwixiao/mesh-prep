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


def test_detection_follows_current_mesh():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    eng.current_mesh = sph                          # aliases original_mesh during migration
    eng.current_mesh.cell_data[PATCH_ID] = np.zeros(sph.n_cells, dtype=np.int64)
    assert len(eng.detect_pieces()) == 1            # one watertight component
    assert eng.detect_open_profiles() == []         # sphere is watertight


def test_named_patches_reports_labeled_faces():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    ids = np.zeros(sph.n_cells, dtype=np.int64)
    ids[:10] = 1
    sph.cell_data[PATCH_ID] = ids
    eng.current_mesh = sph
    eng.patch_names = {1: "inlet"}
    assert eng.named_patches() == [(1, "inlet", 10)]


def test_get_wall_mesh_returns_current(tmp_path):
    p = tmp_path / "s.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(p))
    eng = STLClipperEngine()
    eng.load_stl(str(p))
    assert eng.get_wall_mesh() is eng.current_mesh
