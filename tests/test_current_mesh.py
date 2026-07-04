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


def test_clip_and_name_cap_is_on_plane_and_connected():
    # Order-sensitive guard: the faces labeled as the cap must actually BE the
    # cap — flat on the cut plane and one connected region. Catches any merge
    # cell-reordering that scatters labels onto wall faces.
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    m = eng.current_mesh
    ids = np.asarray(m.cell_data[PATCH_ID])
    cap_idx = np.nonzero(ids == 1)[0]
    assert len(cap_idx) > 0
    centers = m.cell_centers().points[cap_idx]
    assert np.abs(centers[:, 2]).max() < 1e-6          # flat on z=0 plane
    sub = m.extract_cells(cap_idx)
    rid = np.asarray(sub.connectivity('all').cell_data['RegionId'])
    assert len(np.unique(rid)) == 1                     # one connected disc


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


def _bifurcation_patch_engine():
    """Wall sphere + a patch whose faces form TWO disconnected islands
    (two small spheres), all labeled pid 1 — like a clip cap across a bifurcation."""
    eng = STLClipperEngine()
    wall = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    left = pv.Sphere(radius=0.2, center=(3, 0, 0), theta_resolution=8, phi_resolution=8).triangulate()
    right = pv.Sphere(radius=0.2, center=(-3, 0, 0), theta_resolution=8, phi_resolution=8).triangulate()
    wall.cell_data[PATCH_ID] = np.zeros(wall.n_cells, dtype=np.int64)
    left.cell_data[PATCH_ID] = np.ones(left.n_cells, dtype=np.int64)
    right.cell_data[PATCH_ID] = np.ones(right.n_cells, dtype=np.int64)
    combined = wall.merge(left, merge_points=False).merge(right, merge_points=False)
    eng.current_mesh = combined
    eng.patch_names = {1: "inlet"}
    eng._next_patch_id = 2
    eng._patch_normals = {1: (0.0, 0.0, -1.0)}
    return eng, left.n_cells, right.n_cells


def test_split_patch_separates_disconnected_components():
    eng, n_left, n_right = _bifurcation_patch_engine()
    new_pids = eng.split_patch(1)
    assert new_pids is not None and len(new_pids) == 2
    assert 1 not in eng.patch_names                      # parent name retired
    names = sorted(eng.patch_names.values())
    assert names == ["inlet_1", "inlet_2"]
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    counts = sorted(int(np.count_nonzero(ids == p)) for p in new_pids)
    assert counts == sorted([n_left, n_right])           # faces partitioned exactly
    # children inherit the parent's outward normal
    assert all(eng._patch_normals.get(p) == (0.0, 0.0, -1.0) for p in new_pids)


def test_split_patch_single_component_is_noop():
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))     # one connected cap
    assert eng.split_patch(1) is None
    assert eng.patch_names == {1: "inlet"}


def test_split_patch_undo_restores_parent():
    eng, _, _ = _bifurcation_patch_engine()
    eng.split_patch(1)
    assert eng.undo_trim() is True
    assert eng.patch_names == {1: "inlet"}
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert set(np.unique(ids)) == {0, 1}


def test_split_patch_invalid_pid_is_noop():
    eng = _labeled_engine()
    assert eng.split_patch(99) is None
    assert eng.split_patch(0) is None                     # wall is not splittable


def test_wall_subset_original_ids_map_back_to_wall_faces():
    # The viewport draws the pid-0 subset; the picker maps its cell ids back to
    # current_mesh via vtkOriginalCellIds. Verify that mapping is exact.
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    wall_idx = np.nonzero(ids == 0)[0]
    wall = eng.patches_by_id()[0]
    assert "vtkOriginalCellIds" in wall.cell_data
    sub = np.asarray(wall.cell_data["vtkOriginalCellIds"])
    mapped = wall_idx[sub]
    assert np.all(ids[mapped] == 0)                      # all map to wall faces
    c_sub = wall.cell_centers().points
    c_full = eng.current_mesh.cell_centers().points[mapped]
    assert np.allclose(c_sub, c_full, atol=1e-9)          # same physical faces


def test_rename_patch_is_undoable():
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    eng.rename_patch(1, "outlet_left")
    assert eng.patch_names[1] == "outlet_left"
    assert eng.undo_trim() is True
    assert eng.patch_names[1] == "inlet"


def test_cut_by_plane_preserves_labels():
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0.3), (0, 0, -1))   # named patch exists
    n_named = int(np.count_nonzero(
        np.asarray(eng.current_mesh.cell_data[PATCH_ID]) == 1))
    assert n_named > 0
    eng.cut_by_plane((0, 0, -0.2), (0, 0, 1))             # cut elsewhere
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert len(ids) == eng.current_mesh.n_cells
    assert int(np.count_nonzero(ids == 1)) >= n_named     # patch survived the cut


def test_export_clip_planes_has_normal_and_center(tmp_path):
    import json
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))      # keep +z, cap at z=0
    out = tmp_path / "clip_planes.json"
    eng.export_clip_planes(str(out))
    data = json.loads(out.read_text())
    (p,) = data["patches"]
    assert p["name"] == "inlet"
    assert np.allclose(p["outward_normal"], (0, 0, -1))    # out of the kept domain
    assert p["center"] is not None
    assert abs(p["center"][2]) < 1e-6                      # cap centroid on z=0 plane


def _flip_some_faces(mesh, n_flip):
    tri = mesh.faces.reshape(-1, 4).copy()
    tri[:n_flip, [1, 2]] = tri[:n_flip, [2, 1]]           # reverse winding
    return pv.PolyData(mesh.points.copy(), tri.ravel())


def test_check_normals_clean_sphere_outward():
    eng = _labeled_engine()
    r = eng.check_normals()
    assert r["consistent"] is True
    assert r["flipped_edges"] == 0
    assert r["outward"] is True                            # pv.Sphere winds outward


def test_check_normals_detects_flipped_faces():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    eng.current_mesh = _flip_some_faces(sph, 5)
    r = eng.check_normals()
    assert r["consistent"] is False
    assert r["flipped_edges"] > 0


def test_check_normals_inverted_sphere_inward():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    eng.current_mesh = _flip_some_faces(sph, sph.n_cells)  # flip ALL: consistent, inward
    r = eng.check_normals()
    assert r["consistent"] is True
    assert r["outward"] is False


def test_check_normals_open_surface_orientation_unknown():
    eng = _labeled_engine()
    eng.delete_cells(list(range(20)))                      # open a hole
    r = eng.check_normals()
    assert r["consistent"] is True
    assert r["outward"] is None                            # not closed -> undefined


def test_check_normals_no_mesh():
    r = STLClipperEngine().check_normals()
    assert r["consistent"] is None
    assert r["outward"] is None


def test_remove_patch_relabels_to_wall():
    eng = _labeled_engine()
    eng.clip_and_name("inlet", (0, 0, 0), (0, 0, 1))
    n_cells = eng.current_mesh.n_cells
    assert eng.remove_patch(1) is True
    assert eng.patch_names == {}
    ids = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert set(np.unique(ids)) == {0}                          # all wall now
    assert eng.current_mesh.n_cells == n_cells                 # geometry kept
