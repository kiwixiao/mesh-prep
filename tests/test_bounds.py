import numpy as np
import pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine


def test_bounds_info_none_without_mesh():
    assert STLClipperEngine().bounds_info() is None


def test_bounds_info_reports_bbox_and_dims():
    eng = STLClipperEngine()
    # triangle spanning x[0,4], y[0,2], z[0,0]
    pts = np.array([(0, 0, 0), (4, 0, 0), (0, 2, 0)], dtype=float)
    eng.original_mesh = pv.PolyData(pts, np.array([3, 0, 1, 2]))
    info = eng.bounds_info()
    assert (info["xmin"], info["xmax"]) == (0.0, 4.0)
    assert (info["ymin"], info["ymax"]) == (0.0, 2.0)
    assert (info["zmin"], info["zmax"]) == (0.0, 0.0)
    assert info["dx"] == 4.0 and info["dy"] == 2.0 and info["dz"] == 0.0
    assert info["max_dim"] == 4.0
