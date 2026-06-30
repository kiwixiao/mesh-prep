import numpy as np
from mesh_prep.stl_clipper import _project_to_display

# maps world x,y in [0,4] -> NDC [-1,1]; with viewport 100 -> display = 25*world
VIEW = np.array([[0.5, 0, 0, -1],
                 [0, 0.5, 0, -1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]], dtype=float)


def test_project_to_display_maps_world_to_pixels():
    pts = np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
    dx, dy = _project_to_display(pts, VIEW, (100, 100))
    assert np.allclose(dx, [25.0, 75.0])
    assert np.allclose(dy, [25.0, 25.0])
