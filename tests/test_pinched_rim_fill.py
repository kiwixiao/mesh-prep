"""A pinched (figure-8) open profile must fill as separate simple discs.

detect_open_profiles groups boundary edges by connectivity, so one profile can
be two rings meeting at a pinch vertex. Filling that as a single disc spanned
the pinch and produced non-manifold edges; non-manifold edges are excluded
from the flood adjacency, so a double-click on the resulting cap selected a
single face instead of the region (user report, reproduced on coa_001).
"""
import numpy as np
import pyvista as pv

from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def _pinched_engine():
    """Two tubes sharing exactly one rim vertex: their open ends form ONE
    connected boundary group with a degree-4 pinch vertex."""
    a = pv.Cylinder(resolution=12, capping=False, radius=0.5,
                    center=(0, 0, 0), direction=(0, 0, 1)).clean().triangulate()
    b = pv.Cylinder(resolution=12, capping=False, radius=0.5,
                    center=(0, 1.0, 0), direction=(0, 0, 1)).clean().triangulate()
    pts_b = np.asarray(b.points).copy()
    # weld one top-rim vertex of b onto the nearest top-rim vertex of a
    za = np.asarray(a.points)[:, 2].max()
    top_a = [i for i, p in enumerate(np.asarray(a.points)) if abs(p[2] - za) < 1e-9]
    top_b = [i for i, p in enumerate(pts_b) if abs(p[2] - za) < 1e-9]
    tgt = max(top_a, key=lambda i: np.asarray(a.points)[i][1])
    src = min(top_b, key=lambda i: pts_b[i][1])
    pts_b[src] = np.asarray(a.points)[tgt]
    b.points = pts_b
    merged = a.merge(b, merge_points=True).extract_surface().triangulate()
    merged.cell_data[PATCH_ID] = np.zeros(merged.n_cells, dtype=np.int64)
    eng = STLClipperEngine()
    eng.original_mesh = merged
    eng._trim_history.clear()
    eng.patch_names = {}
    eng._next_patch_id = 1
    eng._patch_normals = {}
    return eng


def _degrees(loop):
    seg = loop.lines.reshape(-1, 3)[:, 1:]
    return np.bincount(seg.ravel())


def test_pinched_rim_splits_into_simple_cycles():
    eng = _pinched_engine()
    pinched = [g for g in eng.detect_open_profiles() if (_degrees(g) > 2).any()]
    assert pinched, "fixture is not pinched"
    rim = pinched[0]
    cycles = eng._split_boundary_into_cycles(rim)
    assert len(cycles) >= 2                                   # split happened
    assert sum(c.n_cells for c in cycles) == rim.n_cells      # no edges lost
    for c in cycles:
        assert (_degrees(c) == 2).all()                       # each is simple


def test_simple_rim_is_returned_unchanged():
    eng = STLClipperEngine()
    m = pv.Cylinder(resolution=30, capping=True).clean().triangulate()
    m.cell_data[PATCH_ID] = np.zeros(m.n_cells, dtype=np.int64)
    eng.original_mesh = m
    eng._trim_history.clear(); eng.patch_names = {}
    eng._next_patch_id = 1; eng._patch_normals = {}
    tri = m.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(m.points, dtype=float)
    fn = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
    eng.delete_cells(np.nonzero(fn @ np.array([1.0, 0, 0]) > 0.999)[0].tolist())
    rim = eng.detect_open_profiles()[0]
    out = eng._split_boundary_into_cycles(rim)
    assert len(out) == 1 and out[0] is rim                    # untouched fast path


def test_filling_a_pinched_rim_stays_manifold_and_floodable():
    eng = _pinched_engine()
    pinched = [g for g in eng.detect_open_profiles() if (_degrees(g) > 2).any()]
    rim = pinched[0]
    assert eng.fill_profile(rim, "inlet") is not None
    assert eng.detect_nonmanifold_edges() == []                # no pinch-spanning disc
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    cap_ids = np.nonzero(labels == 1)[0]
    assert len(cap_ids) > 2
    # A double-click anywhere on the cap must flood that whole disc, never a
    # lone face (the two discs meet at a single VERTEX, so an edge flood
    # correctly stays within one of them).
    covered = set()
    for c in cap_ids:
        region = eng.flood_select(int(c))
        assert len(region) > 1, "cap face is isolated in the flood graph"
        covered |= {x for x in region if labels[x] == 1}
    assert covered == set(cap_ids.tolist())      # every cap face reachable
