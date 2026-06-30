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


def _grid():
    eng = STLClipperEngine()
    eng.original_mesh = pv.Plane(i_resolution=5, j_resolution=5).triangulate()
    eng._wall_mesh = eng.original_mesh.copy()
    return eng


def test_grow_one_ring_adds_point_neighbors():
    eng = _grid()
    seed = [0]
    expected = sorted(set(seed) | set(eng.original_mesh.cell_neighbors(0, connections="points")))
    assert eng.grow_cells(seed, rings=1) == expected


def test_grow_two_rings_superset_of_one():
    eng = _grid()
    one = set(eng.grow_cells([0], rings=1))
    two = set(eng.grow_cells([0], rings=2))
    assert one.issubset(two)
    assert len(two) > len(one)


def test_smooth_cells_relaxes_spike_and_leaves_rest_fixed():
    eng = STLClipperEngine()
    grid = pv.Plane(i_resolution=5, j_resolution=5).triangulate()
    # raise the most-connected interior point into a spike
    deg = [len(grid.point_neighbors(i)) for i in range(grid.n_points)]
    spike = int(np.argmax(deg))
    pts = grid.points.copy()
    pts[spike, 2] = 1.0
    grid.points = pts
    eng.original_mesh = grid
    eng._wall_mesh = grid.copy()

    sel = list(grid.point_cell_ids(spike))      # cells touching the spike
    movable = set()
    for cid in sel:
        movable.update(int(p) for p in grid.get_cell(cid).point_ids)
    unsel_pts = [i for i in range(grid.n_points) if i not in movable]
    before_unsel = grid.points[unsel_pts].copy()

    out = eng.smooth_cells(sel, iterations=5, relaxation=0.5)
    assert out is not None
    assert eng.original_mesh.points[spike, 2] < 0.5            # spike relaxed toward neighbors
    assert np.allclose(eng.original_mesh.points[unsel_pts], before_unsel)  # others fixed
    assert len(eng._trim_history) == 1
    assert eng.undo_trim() is True
    assert eng.original_mesh.points[spike, 2] == 1.0          # restored


def test_smooth_cells_empty_is_noop():
    eng = STLClipperEngine()
    eng.original_mesh = pv.Plane(i_resolution=3, j_resolution=3).triangulate()
    eng._wall_mesh = eng.original_mesh.copy()
    assert eng.smooth_cells([], iterations=5) is None
    assert len(eng._trim_history) == 0
