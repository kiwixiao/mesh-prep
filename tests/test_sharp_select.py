"""select_to_sharp_edges + assign_patch_from_cells.

The click-a-cap workflow: seed one face on a flat end cap, flood to the sharp
rim (dihedral angle threshold), assign the result as a named patch (outlet1).
"""
import numpy as np
import pyvista as pv

from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def _engine_with(mesh: pv.PolyData) -> STLClipperEngine:
    eng = STLClipperEngine()
    mesh = mesh.triangulate()
    mesh.cell_data[PATCH_ID] = np.zeros(mesh.n_cells, dtype=np.int64)
    eng.original_mesh = mesh
    eng._trim_history.clear()
    eng.patch_names = {}
    eng._next_patch_id = 1
    eng._patch_normals = {}
    return eng


def _cap_cells(mesh, axis, sign):
    """Cells of a cylinder end cap: face normal aligned with ±axis."""
    tri = mesh.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(mesh.points, dtype=float)
    fn = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
    return set(np.nonzero(fn @ (sign * np.asarray(axis, dtype=float)) > 0.999)[0].tolist())


def test_cube_face_bounded_by_90_degree_edges():
    eng = _engine_with(pv.Cube())
    m = eng.current_mesh
    top = sorted(_cap_cells(m, (0, 0, 1), 1))              # the +z quad = 2 triangles
    assert len(top) == 2
    got = eng.select_to_sharp_edges([top[0]], angle_deg=30.0)
    assert got == top                                      # stops at the cube edges


def test_cylinder_cap_selected_from_one_seed():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True))
    m = eng.current_mesh
    cap = _cap_cells(m, (1, 0, 0), 1)                      # +x end cap
    assert len(cap) > 10
    seed = next(iter(cap))
    got = set(eng.select_to_sharp_edges([seed], angle_deg=30.0))
    assert got == cap                                      # whole cap, nothing else


def test_multiple_seeds_grow_independently():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True))
    m = eng.current_mesh
    cap_a = _cap_cells(m, (1, 0, 0), 1)
    cap_b = _cap_cells(m, (1, 0, 0), -1)
    got = set(eng.select_to_sharp_edges(
        [next(iter(cap_a)), next(iter(cap_b))], angle_deg=30.0))
    assert got == cap_a | cap_b


def test_smooth_sphere_selects_everything():
    eng = _engine_with(pv.Sphere())
    got = eng.select_to_sharp_edges([0], angle_deg=30.0)
    assert len(got) == eng.current_mesh.n_cells


def test_feature_curve_is_a_wall_even_on_smooth_surface():
    """After a cut, the flood must stop at the feature curve exactly like
    flood_select does, even though the surface there is smooth."""
    eng = _engine_with(pv.Sphere())
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    centers = eng.current_mesh.cell_centers().points
    seed = int(np.argmax(centers[:, 2]))                   # a face well inside the north
    sharp = set(eng.select_to_sharp_edges([seed], angle_deg=60.0))
    flood = set(eng.flood_select(seed))
    assert sharp == flood                                  # same wall, same region
    assert len(sharp) < eng.current_mesh.n_cells


def test_bad_seeds_return_empty():
    eng = _engine_with(pv.Sphere())
    assert eng.select_to_sharp_edges([]) == []
    assert eng.select_to_sharp_edges([-1, 10 ** 9]) == []


def test_assign_patch_from_cells_labels_and_normal():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True))
    cap = sorted(_cap_cells(eng.current_mesh, (1, 0, 0), 1))
    n_before = eng.current_mesh.n_cells
    pid = eng.assign_patch_from_cells(cap, "outlet1")
    assert pid is not None
    assert eng.patch_names[pid] == "outlet1"
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert sorted(np.nonzero(labels == pid)[0].tolist()) == cap
    assert eng.current_mesh.n_cells == n_before            # pure relabel
    normal = np.asarray(eng._patch_normals[pid])
    assert np.allclose(normal, (1, 0, 0), atol=1e-6)       # cap faces point +x
    # One undo restores labels and the patch registry.
    assert eng.undo_trim() is True
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert not (labels == pid).any()
    assert pid not in eng.patch_names
    assert pid not in eng._patch_normals


def test_sharp_select_works_after_clip():
    """clip_and_name used to leave quads (breaking the all-triangle invariant),
    which silently reduced sharp-select to the seed face and made check_normals
    report unknown. The clip paths now re-triangulate."""
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True))
    eng.clip_and_name("inlet", (0.4, 0, 0), (1, 0, 0))
    m = eng.current_mesh
    f = m.faces
    assert f.size == 4 * m.n_cells and bool((f.reshape(-1, 4)[:, 0] == 3).all())
    # a wall seed floods the whole curved wall, far more than 1 face
    centers = m.cell_centers().points
    r = np.linalg.norm(centers[:, 1:], axis=1)
    wall_seed = int(np.argmax(r))                          # on the curved side
    got = eng.select_to_sharp_edges([wall_seed], angle_deg=30.0)
    assert len(got) > 50
    assert eng.check_normals()["consistent"] is not None   # indicator alive again


def test_cut_by_plane_keeps_triangles():
    eng = _engine_with(pv.Sphere())
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    m = eng.current_mesh
    f = m.faces
    assert f.size == 4 * m.n_cells and bool((f.reshape(-1, 4)[:, 0] == 3).all())


def test_assign_skips_normal_when_faces_cancel():
    """Selecting BOTH end caps (antipodal normals) must not record a noise
    direction as the patch normal — it would become the inlet velocity in 0/U."""
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True))
    m = eng.current_mesh
    both = sorted(_cap_cells(m, (1, 0, 0), 1) | _cap_cells(m, (1, 0, 0), -1))
    pid = eng.assign_patch_from_cells(both, "weird")
    assert pid is not None
    assert pid not in eng._patch_normals                   # no fabricated direction


def test_assign_purges_emptied_patches():
    """Re-assigning every face of an existing patch must drop the old name, so
    exports never reference a patch with zero faces."""
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True))
    cap = sorted(_cap_cells(eng.current_mesh, (1, 0, 0), 1))
    pid1 = eng.assign_patch_from_cells(cap, "outlet1")
    pid2 = eng.assign_patch_from_cells(cap, "outlet2")     # steals every face
    assert pid2 is not None
    assert pid1 not in eng.patch_names                     # emptied -> purged
    assert pid1 not in eng._patch_normals
    assert eng.patch_names[pid2] == "outlet2"
    # partial overlap keeps the survivor
    eng2 = _engine_with(pv.Cylinder(resolution=60, capping=True))
    cap2 = sorted(_cap_cells(eng2.current_mesh, (1, 0, 0), 1))
    p1 = eng2.assign_patch_from_cells(cap2, "outlet1")
    p2 = eng2.assign_patch_from_cells(cap2[: len(cap2) // 2], "outlet2")
    assert p1 in eng2.patch_names and p2 in eng2.patch_names


def test_assign_patch_noops():
    eng = _engine_with(pv.Sphere())
    assert eng.assign_patch_from_cells([], "x") is None
    assert eng.assign_patch_from_cells([0], "   ") is None
    assert eng.assign_patch_from_cells([0], "") is None
    assert len(eng._trim_history) == 0
    assert eng.patch_names == {}
