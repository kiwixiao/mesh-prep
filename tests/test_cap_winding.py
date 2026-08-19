"""Caps merged by clip/fill/remesh must be wound like the surrounding surface.

Before the fix, every clip_and_name on a pristine mesh produced ~120 flipped
edges (cap faces oriented into the domain), which broke the normals health
line and silently reversed the inlet velocity derived from fill-created
patches in the OpenFOAM export.
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
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    m = eng.current_mesh
    tri = m.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(m.points, dtype=float)
    fn = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
    cap = np.nonzero(fn @ np.array([1.0, 0, 0]) > 0.999)[0].tolist()
    eng.delete_cells(cap)
    return eng


def test_clip_keeps_winding_consistent_and_outward():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    assert eng.check_normals() == {"consistent": True, "flipped_edges": 0,
                                   "outward": True}
    eng.clip_and_name("inlet", (0.3, 0, 0), (1, 0, 0))
    r = eng.check_normals()
    assert r["consistent"] is True
    assert r["flipped_edges"] == 0
    assert r["outward"] is True


def test_fill_named_keeps_winding_and_records_outward_normal():
    eng = _open_cylinder()
    assert eng.check_normals()["consistent"] is True
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    r = eng.check_normals()
    assert r["consistent"] is True and r["flipped_edges"] == 0
    assert r["outward"] is True                       # watertight again
    normal = np.asarray(eng._patch_normals[1])        # recorded for export
    assert np.allclose(normal, (1, 0, 0), atol=1e-6)  # true outward (+x)


def test_fill_as_wall_keeps_winding():
    eng = _open_cylinder()
    eng.fill_profile(eng.detect_open_profiles()[0])   # no name -> wall
    r = eng.check_normals()
    assert r["consistent"] is True and r["flipped_edges"] == 0


def test_remesh_patch_keeps_winding():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    eng.clip_and_name("inlet", (0.3, 0, 0), (1, 0, 0))
    assert eng.check_normals()["flipped_edges"] == 0  # after group-1 clip fix
    out = eng.remesh_patch(1)
    assert out is not None
    r = eng.check_normals()
    assert r["consistent"] is True and r["flipped_edges"] == 0
