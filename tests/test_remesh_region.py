"""remesh_region: delete a selected region and re-triangulate its rim.

The local fix for a non-manifold junction — select its attached faces, Grow,
Remesh — replaces the defect with a clean disc over the rim, keeping the
surface closed. Pre-existing openings are matched by geometry and left open.
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


def _n_open(mesh) -> int:
    return mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False).n_cells


def _sphere_with_fin() -> pv.PolyData:
    """Closed sphere plus one fin triangle glued onto an existing edge —
    that edge then has 3 faces (non-manifold), like a scan artifact."""
    sph = pv.Sphere(theta_resolution=20, phi_resolution=20).triangulate()
    tri = sph.faces.reshape(-1, 4)[:, 1:]
    a, b = int(tri[0][0]), int(tri[0][1])
    tip = (sph.points[a] + sph.points[b]) / 2 * 1.4        # off-surface tip
    pts = np.vstack([sph.points, tip])
    faces = np.concatenate([sph.faces, [3, a, b, len(pts) - 1]])
    return pv.PolyData(pts, faces)


def test_remesh_region_removes_nonmanifold_and_stays_closed():
    eng = _engine_with(_sphere_with_fin())
    groups = eng.detect_nonmanifold_edges()
    assert len(groups) == 1                                # fixture is defective
    assert _n_open(eng.current_mesh) > 0                   # fin's free edges

    # The GUI workflow: click the group (selects attached faces), Grow, Remesh.
    sel = eng.faces_on_edges(groups[0])
    assert len(sel) == 3                                   # 2 sphere + 1 fin face
    sel = eng.grow_cells(sel, rings=1)
    msg = eng.remesh_region(sel)

    assert msg is not None and "Remeshed region" in msg
    assert eng.detect_nonmanifold_edges() == []            # junction gone
    assert _n_open(eng.current_mesh) == 0                  # closed again
    r = eng.check_normals()
    assert r["consistent"] is True
    assert r["outward"] is True                            # re-wound + oriented
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert set(np.unique(labels).tolist()) == {0}          # rebuilt faces are wall


def test_remesh_region_preserves_preexisting_openings():
    """A named-opening-style hole elsewhere on the mesh must NOT be capped."""
    sph = pv.Sphere(theta_resolution=20, phi_resolution=20).triangulate()
    eng = _engine_with(sph)
    # Open a hole at the top (stands in for an inlet/outlet opening).
    top = [i for i, c in enumerate(sph.cell_centers().points) if c[2] > 0.45]
    eng.delete_cells(top)
    assert len(eng.detect_open_profiles()) == 1
    rim_before = eng.detect_open_profiles()[0].n_cells

    # Remesh a small region near the equator, far from the hole.
    eq = int(np.argmin(np.abs(eng.current_mesh.cell_centers().points[:, 2])))
    sel = eng.grow_cells([eq], rings=2)
    msg = eng.remesh_region(sel)

    assert msg is not None
    profiles = eng.detect_open_profiles()
    assert len(profiles) == 1                              # hole still open
    assert profiles[0].n_cells == rim_before               # and untouched
    assert eng.detect_nonmanifold_edges() == []


def test_remesh_region_is_single_undo_step():
    eng = _engine_with(_sphere_with_fin())
    before = eng.current_mesh.n_cells
    sel = eng.grow_cells(eng.faces_on_edges(eng.detect_nonmanifold_edges()[0]), rings=1)
    eng.remesh_region(sel)
    assert eng.current_mesh.n_cells != before or eng.detect_nonmanifold_edges() == []
    assert eng.undo_trim() is True
    assert eng.current_mesh.n_cells == before              # one undo restores all
    assert len(eng.detect_nonmanifold_edges()) == 1        # fin is back


def test_remesh_region_noop_on_bad_selection():
    eng = _engine_with(pv.Sphere())
    n = eng.current_mesh.n_cells
    assert eng.remesh_region([]) is None
    assert eng.remesh_region(range(n)) is None             # whole mesh
    assert eng.current_mesh.n_cells == n
    assert len(eng._trim_history) == 0                     # no history pushed
