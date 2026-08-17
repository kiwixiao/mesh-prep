"""extrude_profile: flow extensions — extrude an open rim along its cut normal.

After a flat cut, the open rim extrudes straight along the best-fit plane
normal (== the cut normal), away from the body, so the vessel continues
naturally. The far end is a new open profile, cappable via fill_profile.
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


def _open_cylinder():
    """Cylinder with the +x cap removed — one flat open rim, normal +x.
    clean() first: pv.Cylinder's caps do NOT share vertices with the tube
    (hidden seams); welding makes the fixture a real conformal surface."""
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    m = eng.current_mesh
    tri = m.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(m.points, dtype=float)
    fn = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
    cap = np.nonzero(fn @ np.array([1.0, 0, 0]) > 0.999)[0].tolist()
    eng.delete_cells(cap)
    return eng


def _n_open(mesh) -> int:
    return mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False).n_cells


def test_extrude_extends_along_cut_normal():
    eng = _open_cylinder()
    prof = eng.detect_open_profiles()
    assert len(prof) == 1
    rim_edges = prof[0].n_cells
    x_before = float(np.asarray(prof[0].points)[:, 0].mean())   # ~ +0.5
    n_before = eng.current_mesh.n_cells

    msg = eng.extrude_profile(prof[0], length=0.6)
    assert msg is not None and "open profile" in msg

    # Still exactly one open rim, moved ~0.6 along +x (the cut normal).
    prof2 = eng.detect_open_profiles()
    assert len(prof2) == 1
    assert prof2[0].n_cells == rim_edges                        # same rim shape
    x_after = float(np.asarray(prof2[0].points)[:, 0].mean())
    assert abs((x_after - x_before) - 0.6) < 1e-6
    assert eng.current_mesh.n_cells > n_before
    # Radius preserved (straight extension, no taper).
    r = np.linalg.norm(np.asarray(prof2[0].points)[:, 1:], axis=1)
    assert np.allclose(r, 0.5, atol=1e-6)


def test_extrude_winding_stays_consistent_without_rewind():
    """Side walls are wound from the rim's directed boundary edges, so the
    merged surface needs no global re-wind."""
    eng = _open_cylinder()
    assert eng.check_normals()["consistent"] is True            # precondition
    eng.extrude_profile(eng.detect_open_profiles()[0], length=0.6)
    r = eng.check_normals()
    assert r["consistent"] is True
    assert r["flipped_edges"] == 0
    assert eng.detect_nonmanifold_edges() == []
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert set(np.unique(labels).tolist()) == {0}               # extension is wall


def test_extrude_then_fill_closes_the_vessel():
    eng = _open_cylinder()
    eng.extrude_profile(eng.detect_open_profiles()[0], length=0.6)
    eng.fill_profile(eng.detect_open_profiles()[0], "outlet1")
    assert _n_open(eng.current_mesh) == 0                       # watertight
    assert "outlet1" in eng.patch_names.values()


def test_extrude_curved_rim_on_sphere_hole():
    """A non-flat rim still extrudes cleanly along its best-fit normal."""
    eng = _engine_with(pv.Sphere())
    centers = eng.current_mesh.cell_centers().points
    eng.delete_cells([i for i, c in enumerate(centers) if c[2] > 0.45])
    prof = eng.detect_open_profiles()
    assert len(prof) == 1
    z_before = float(np.asarray(prof[0].points)[:, 2].mean())
    msg = eng.extrude_profile(prof[0], length=0.4)
    assert msg is not None
    prof2 = eng.detect_open_profiles()
    assert len(prof2) == 1
    z_after = float(np.asarray(prof2[0].points)[:, 2].mean())
    assert abs((z_after - z_before) - 0.4) < 0.02               # ~+z, away from body
    assert eng.check_normals()["consistent"] is True


def test_extrude_is_single_undo_step():
    eng = _open_cylinder()
    n_before = eng.current_mesh.n_cells
    eng.extrude_profile(eng.detect_open_profiles()[0], length=0.6)
    assert eng.current_mesh.n_cells > n_before
    assert eng.undo_trim() is True
    assert eng.current_mesh.n_cells == n_before


def test_extrude_warns_on_preexisting_flips():
    """Extruding an already-inconsistent surface replicates the winding conflict
    down every ring — the message must tell the user to Fix Normals first."""
    eng = _open_cylinder()
    m = eng.current_mesh
    faces = m.faces.reshape(-1, 4).copy()
    faces[0, 1:] = faces[0, 1:][::-1]                          # flip one triangle
    flipped = pv.PolyData(np.asarray(m.points), faces.ravel())
    eng2 = _engine_with(flipped)
    assert eng2.check_normals()["flipped_edges"] > 0            # fixture is broken
    msg = eng2.extrude_profile(eng2.detect_open_profiles()[0], length=0.6)
    assert msg is not None and "Fix Normals" in msg


def test_extrude_noops():
    eng = _open_cylinder()
    prof = eng.detect_open_profiles()[0]
    n = eng.current_mesh.n_cells
    hist = len(eng._trim_history)
    assert eng.extrude_profile(None, 1.0) is None
    assert eng.extrude_profile(pv.PolyData(), 1.0) is None
    assert eng.extrude_profile(prof, 0.0) is None
    assert eng.extrude_profile(prof, -1.0) is None
    assert eng.extrude_profile(prof, float("nan")) is None
    assert eng.current_mesh.n_cells == n
    assert len(eng._trim_history) == hist                       # nothing pushed
