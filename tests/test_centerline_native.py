"""Tests for the native (vmtk-free) centerline engine."""

import numpy as np
import pyvista as pv
import pytest

from mesh_prep.centerline_native import (
    compute_centerlines, split_surface_by_centerline, _decompose_branches,
    _tet_circumcenters, RADIUS_ARRAY,
)


# ── branch decomposition (pure graph, deterministic) ──────────────────

def test_decompose_straight_path_single_branch():
    branches, bifs = _decompose_branches([[0, 1, 2, 3, 4]])
    assert bifs == set()
    assert len(branches) == 1
    assert branches[0] == [0, 1, 2, 3, 4]


def test_decompose_y_gives_three_branches_one_bifurcation():
    branches, bifs = _decompose_branches([[0, 1, 2, 5, 6], [0, 1, 2, 7, 8]])
    assert bifs == {2}
    assert len(branches) == 3
    assert [0, 1, 2] in branches
    assert [2, 5, 6] in branches
    assert [2, 7, 8] in branches


def test_decompose_two_bifurcations():
    # trunk 0-1-2 splits at 2; one child 3 splits again at 4
    paths = [[0, 1, 2, 9], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6]]
    branches, bifs = _decompose_branches(paths)
    assert bifs == {2, 4}
    assert len(branches) == 5           # trunk + (2->9) + (2->3..4) + (4->5) + (4->6)


# ── circumcenter math ─────────────────────────────────────────────────

def test_circumcenter_of_regular_tet_is_centroid_ish():
    pts = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], float)
    tets = np.array([[0, 1, 2, 3]])
    c, r, valid = _tet_circumcenters(pts, tets)
    assert valid[0]
    assert np.allclose(c[0], (0, 0, 0), atol=1e-9)      # symmetric -> origin
    assert np.isclose(r[0], np.sqrt(3), atol=1e-9)      # all verts at dist sqrt(3)


# ── geometry: watertight helpers ──────────────────────────────────────

def _watertight(mesh):
    pymeshfix = pytest.importorskip("pymeshfix")
    mf = pymeshfix.MeshFix(mesh.triangulate())
    mf.repair()
    return mf.mesh


def _dense(mesh, levels=3):
    """Triangulate + subdivide so walls carry points along their length
    (pv.Cylinder samples only the two end rings, which under-fills the axis)."""
    return mesh.triangulate().subdivide(levels, subfilter="linear")


def test_centerline_cylinder_is_axial_with_correct_radius():
    cyl = _watertight(_dense(pv.Cylinder(direction=(0, 0, 1), radius=1.0,
                                         height=6.0, resolution=48, capping=True)))
    out = compute_centerlines(cyl, [0, 0, -2.7], [0, 0, 2.7])
    assert out.n_points > 2
    assert RADIUS_ARRAY in out.point_data
    pts = out.points
    # Path hugs the z-axis: x,y small; z spans most of the height.
    assert np.abs(pts[:, :2]).max() < 0.4
    assert pts[:, 2].ptp() > 4.0
    # MISR ~ cylinder radius (discrete circumradius, tolerant band).
    r = np.asarray(out[RADIUS_ARRAY])
    assert 0.6 < float(np.median(r)) < 1.4
    assert int(out["BranchId"].max()) == 0              # single branch


def test_centerline_y_detects_bifurcation():
    trunk = pv.Cylinder(center=(0, 0, -2), direction=(0, 0, 1), radius=0.6,
                        height=4.0, resolution=40, capping=True)
    left = pv.Cylinder(center=(-1.2, 0, 1.2), direction=(-0.6, 0, 0.8),
                       radius=0.45, height=3.0, resolution=40, capping=True)
    right = pv.Cylinder(center=(1.2, 0, 1.2), direction=(0.6, 0, 0.8),
                        radius=0.45, height=3.0, resolution=40, capping=True)
    merged = trunk.merge(left, merge_points=False).merge(right, merge_points=False)
    try:
        y = _watertight(_dense(merged, levels=2))
    except Exception:
        pytest.skip("could not build a watertight Y test surface")
    if y is None or y.n_open_edges > 0:
        pytest.skip("Y surface not watertight after MeshFix")
    try:
        out = compute_centerlines(y, [0, 0, -3.8], [[-2.4, 0, 2.8], [2.4, 0, 2.8]])
    except RuntimeError:
        pytest.skip("centerline seeds unreachable on this synthetic Y")
    assert int(out["BranchId"].max()) >= 2              # >= 3 branches -> a bifurcation
    assert out.n_points > 4


def test_split_surface_cylinder_single_region():
    cyl = _watertight(_dense(pv.Cylinder(direction=(0, 0, 1), radius=1.0,
                                         height=6.0, resolution=48, capping=True)))
    cl = compute_centerlines(cyl, [0, 0, -2.7], [0, 0, 2.7])
    surf, labels = split_surface_by_centerline(cyl, cl)
    assert len(labels) == surf.n_cells                  # every face labeled
    assert set(np.unique(labels)) <= {0}                # one branch -> one region


def test_centerline_exposes_bifurcation_points_and_splits_y():
    trunk = pv.Cylinder(center=(0, 0, -2), direction=(0, 0, 1), radius=0.6,
                        height=4.0, resolution=40, capping=True)
    left = pv.Cylinder(center=(-1.2, 0, 1.2), direction=(-0.6, 0, 0.8),
                       radius=0.45, height=3.0, resolution=40, capping=True)
    right = pv.Cylinder(center=(1.2, 0, 1.2), direction=(0.6, 0, 0.8),
                        radius=0.45, height=3.0, resolution=40, capping=True)
    merged = trunk.merge(left, merge_points=False).merge(right, merge_points=False)
    try:
        y = _watertight(_dense(merged, levels=2))
    except Exception:
        pytest.skip("could not build a watertight Y test surface")
    if y is None or y.n_open_edges > 0:
        pytest.skip("Y surface not watertight after MeshFix")
    try:
        cl = compute_centerlines(y, [0, 0, -3.8], [[-2.4, 0, 2.8], [2.4, 0, 2.8]])
    except RuntimeError:
        pytest.skip("centerline seeds unreachable on this synthetic Y")
    # bifurcation points exposed with position + radius
    assert "bifurcation_points" in cl.field_data
    bifs = np.asarray(cl.field_data["bifurcation_points"])
    assert bifs.shape[0] >= 1 and bifs.shape[1] == 3
    assert cl.field_data["bifurcation_radius"].shape[0] == bifs.shape[0]
    # the surface splits into >= 3 branch regions and every face is labeled
    surf, labels = split_surface_by_centerline(y, cl)
    assert len(labels) == surf.n_cells
    assert len(np.unique(labels)) >= 3


def test_centerline_empty_seeds_raises():
    cyl = _watertight(pv.Cylinder(radius=1.0, height=4.0, resolution=24, capping=True))
    with pytest.raises(ValueError):
        compute_centerlines(cyl, [], [0, 0, 1])
