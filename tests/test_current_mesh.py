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


def _labeled_engine():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    sph.cell_data[PATCH_ID] = np.zeros(sph.n_cells, dtype=np.int64)
    eng.current_mesh = sph
    return eng


def test_clip_and_name_trims_and_labels():
    eng = _labeled_engine()
    n0 = eng.current_mesh.n_cells
    out = eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    assert out is eng.current_mesh
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert eng.patch_names == {1: "inlet"}
    assert 1 in np.unique(ids)                                # cap labeled inlet
    assert 0 in np.unique(ids)                                # wall remains
    assert int(np.count_nonzero(ids == 0)) < n0               # wall was trimmed
    assert len(eng._trim_history) == 1                        # one undo step


def test_clip_and_name_undo_restores_everything():
    eng = _labeled_engine()
    n0 = eng.current_mesh.n_cells
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    assert eng.undo_trim() is True
    assert eng.current_mesh.n_cells == n0
    assert eng.patch_names == {}
    assert eng._patch_normals == {}


def test_clip_and_name_miss_is_noop():
    eng = _labeled_engine()
    out = eng.clip_and_name("inlet", (0, 0, 100), (0, 0, 1))  # plane misses sphere
    assert out is None
    assert eng.patch_names == {}
    assert len(eng._trim_history) == 0


def test_clip_and_name_records_outward_normal():
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))          # keep +z side
    assert 1 in eng._patch_normals
    assert np.allclose(eng._patch_normals[1], (0, 0, -1))     # outward = -kept normal


def test_export_combined_groups_by_label(tmp_path):
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    out = tmp_path / "m.stl"
    eng.export_combined_stl(str(out))
    text = out.read_text()
    assert "solid inlet" in text
    assert "solid wall" in text
    assert text.rstrip().endswith("endsolid wall")            # wall block last


def test_export_separate_groups_by_label(tmp_path):
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    d = tmp_path / "sep"
    eng.export_separate_stl(str(d))
    assert (d / "inlet.stl").exists()
    assert (d / "wall.stl").exists()
    assert "solid inlet" in (d / "inlet.stl").read_text()


def test_remove_patch_relabels_to_wall():
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    n_cells = eng.current_mesh.n_cells
    assert eng.remove_patch(1) is True
    assert eng.patch_names == {}
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert set(np.unique(ids)) == {0}                          # all wall now
    assert eng.current_mesh.n_cells == n_cells                 # geometry kept
