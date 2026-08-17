"""remesh_surface: whole-surface isotropic remesh (ACVD via pyacvd).

Skipped where pyacvd is unavailable (the legacy conda env: its bundled OpenMP
clashes with vmtk's). Runs in the base-env suite.
"""
import numpy as np
import pytest
import pyvista as pv

pytest.importorskip("pyacvd")

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


def _min_angles(mesh):
    tri = mesh.faces.reshape(-1, 4)[:, 1:]
    p = np.asarray(mesh.points, dtype=float)
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]

    def ang(u, v):
        cosv = np.einsum('ij,ij->i', u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
        return np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    A, B = ang(b - a, c - a), ang(a - b, c - b)
    return np.minimum(np.minimum(A, B), np.maximum(180 - A - B, 0))


def test_remesh_surface_isotropic_and_faithful():
    eng = _engine_with(pv.Sphere(theta_resolution=40, phi_resolution=40))
    msg = eng.remesh_surface(4000)
    assert "Remeshed surface" in msg
    m = eng.current_mesh
    assert abs(m.n_cells - 4000) < 0.25 * 4000            # near target
    mn = _min_angles(m)
    assert np.median(mn) > 40.0                            # near-equilateral
    assert (mn < 15).mean() < 0.05
    # watertight preserved, geometry faithful (all points near unit sphere)
    b = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                manifold_edges=False, non_manifold_edges=False)
    assert b.n_cells == 0
    r = np.linalg.norm(np.asarray(m.points), axis=1)
    assert np.abs(r - 0.5).max() < 0.02                    # <4% of radius


def test_remesh_surface_carries_patch_labels():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    eng.clip_and_name("outlet1", (0.3, 0, 0), (1, 0, 0))
    area_cap0 = eng.current_mesh.extract_cells(
        np.nonzero(np.asarray(eng.current_mesh.cell_data[PATCH_ID]) == 1)[0]
    ).extract_surface().area
    eng.remesh_surface(6000)
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    assert 1 in eng.patch_names and (labels == 1).any()    # patch survived
    cap = eng.current_mesh.extract_cells(
        np.nonzero(labels == 1)[0]).extract_surface()
    assert abs(cap.area - area_cap0) < 0.35 * area_cap0    # approximate borders


def test_remesh_surface_single_undo():
    eng = _engine_with(pv.Sphere())
    n0 = eng.current_mesh.n_cells
    eng.remesh_surface(3000)
    assert eng.current_mesh.n_cells != n0
    assert eng.undo_trim() is True
    assert eng.current_mesh.n_cells == n0


def test_remesh_surface_noops():
    eng = _engine_with(pv.Sphere())
    assert "Target too small" in eng.remesh_surface(10)
    assert len(eng._trim_history) == 0
