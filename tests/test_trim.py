import numpy as np
import pyvista as pv
import pytest
from mesh_prep.stl_clipper import _points_in_polygon, STLClipperEngine


def test_points_in_polygon_square():
    square = [(0, 0), (50, 0), (50, 50), (0, 50)]
    xs = np.array([25.0, 75.0, 25.0, 75.0])
    ys = np.array([25.0, 25.0, 75.0, 75.0])
    inside = _points_in_polygon(xs, ys, square)
    assert inside.tolist() == [True, False, False, False]


# world->clip: maps x,y in [0,4] to NDC [-1,1]; z->0; w->1.
# Combined with VIEWPORT below, world (x, y) projects to display (25*x, 25*y).
VIEW = np.array([[0.5, 0, 0, -1],
                 [0, 0.5, 0, -1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]], dtype=float)
VIEWPORT = (100, 100)


def _engine_with_quad():
    """Engine holding 4 triangles with centroids exactly at
    (1,1), (3,1), (1,3), (3,3), z=0  ->  display (25,25),(75,25),(25,75),(75,75)."""
    centers = [(1, 1), (3, 1), (1, 3), (3, 3)]
    pts, faces = [], []
    for cx, cy in centers:
        b = len(pts)
        pts += [(cx, cy + 0.2, 0), (cx - 0.2, cy - 0.1, 0), (cx + 0.2, cy - 0.1, 0)]
        faces += [3, b, b + 1, b + 2]
    mesh = pv.PolyData(np.array(pts, float), np.array(faces))
    eng = STLClipperEngine()
    eng.original_mesh = mesh
    eng._wall_mesh = mesh.copy()
    return eng


def test_trim_removes_lassoed_cells():
    eng = _engine_with_quad()
    square = [(0, 0), (50, 0), (50, 50), (0, 50)]   # contains only display (25,25)
    result = eng.trim_by_screen_polygon(square, VIEW, VIEWPORT)
    assert result is not None
    assert eng.original_mesh.n_cells == 3
    assert eng._wall_mesh.n_cells == 3
    assert len(eng._trim_history) == 1


def test_trim_degenerate_polygon_is_noop():
    eng = _engine_with_quad()
    assert eng.trim_by_screen_polygon([(0, 0), (1, 1)], VIEW, VIEWPORT) is None
    assert eng.original_mesh.n_cells == 4
    assert len(eng._trim_history) == 0


def test_trim_selecting_nothing_is_noop():
    eng = _engine_with_quad()
    far = [(200, 200), (250, 200), (250, 250), (200, 250)]
    assert eng.trim_by_screen_polygon(far, VIEW, VIEWPORT) is None
    assert eng.original_mesh.n_cells == 4
    assert len(eng._trim_history) == 0


def test_trim_selecting_everything_refuses():
    eng = _engine_with_quad()
    allp = [(-10, -10), (1000, -10), (1000, 1000), (-10, 1000)]
    with pytest.raises(ValueError):
        eng.trim_by_screen_polygon(allp, VIEW, VIEWPORT)
    assert eng.original_mesh.n_cells == 4
    assert len(eng._trim_history) == 0


def test_undo_restores_previous_mesh():
    eng = _engine_with_quad()
    eng.trim_by_screen_polygon([(0, 0), (50, 0), (50, 50), (0, 50)], VIEW, VIEWPORT)
    assert eng.original_mesh.n_cells == 3
    assert eng.undo_trim() is True
    assert eng.original_mesh.n_cells == 4
    assert eng._wall_mesh.n_cells == 4


def test_undo_multi_level():
    eng = _engine_with_quad()
    # trim 1: removes display (25,25) -> 4 -> 3
    eng.trim_by_screen_polygon([(0, 0), (50, 0), (50, 50), (0, 50)], VIEW, VIEWPORT)
    # trim 2: removes display (75,25) -> 3 -> 2
    eng.trim_by_screen_polygon([(50, 0), (100, 0), (100, 50), (50, 50)], VIEW, VIEWPORT)
    assert eng.original_mesh.n_cells == 2
    assert eng.undo_trim() is True
    assert eng.original_mesh.n_cells == 3
    assert eng.undo_trim() is True
    assert eng.original_mesh.n_cells == 4
    assert eng.undo_trim() is False
