import numpy as np
import pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine

VIEW = np.array([[0.5, 0, 0, -1],
                 [0, 0.5, 0, -1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]], dtype=float)
VIEWPORT = (100, 100)
BIG_POLY = [(0, 0), (100, 0), (100, 100), (0, 100)]  # covers x,y in world [0,4]


def _two_triangles():
    # tri 0 around x~0.3 (winding -> +z normal); tri 1 around x~3.3 (-> -z normal)
    pts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                    (3, 0, 0), (3, 1, 0), (4, 0, 0)], dtype=float)
    faces = np.array([3, 0, 1, 2, 3, 3, 4, 5])
    eng = STLClipperEngine()
    eng.original_mesh = pv.PolyData(pts, faces)
    eng._wall_mesh = eng.original_mesh.copy()
    return eng


def test_select_all_when_front_only_false():
    eng = _two_triangles()
    ids = eng.select_cells_in_polygon(BIG_POLY, VIEW, VIEWPORT, (0, 0, 1), front_only=False)
    assert sorted(ids) == [0, 1]


def test_select_front_only_excludes_back_facing():
    eng = _two_triangles()
    vd = np.array([0, 0, 1.0])
    normals = eng.original_mesh.cell_normals
    expected = [i for i in (0, 1) if float(normals[i] @ vd) < 0]
    ids = eng.select_cells_in_polygon(BIG_POLY, VIEW, VIEWPORT, vd, front_only=True)
    assert sorted(ids) == sorted(expected)
    assert len(ids) == 1  # exactly one of the two faces the camera


def test_select_empty_polygon_returns_empty():
    eng = _two_triangles()
    assert eng.select_cells_in_polygon([(0, 0), (1, 1)], VIEW, VIEWPORT, (0, 0, 1)) == []
