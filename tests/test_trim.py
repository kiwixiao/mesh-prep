import numpy as np
from mesh_prep.stl_clipper import _points_in_polygon


def test_points_in_polygon_square():
    square = [(0, 0), (50, 0), (50, 50), (0, 50)]
    xs = np.array([25.0, 75.0, 25.0, 75.0])
    ys = np.array([25.0, 25.0, 75.0, 75.0])
    inside = _points_in_polygon(xs, ys, square)
    assert inside.tolist() == [True, False, False, False]
