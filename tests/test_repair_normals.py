"""Fix Normals auto-repair of small non-manifold blocks.

A few non-manifold edges (3+ faces on one edge) make a surface locally
non-orientable, so re-winding alone cannot reach a consistent winding — vtk will
not propagate a consistent orientation across a non-manifold edge. ``repair_normals``
detects this and, when only a small number of faces sit on those edges, removes
them and re-winds, which reconnects the surface into orientable pieces.
"""
import numpy as np
import pyvista as pv
import pytest

from mesh_prep.stl_clipper import STLClipperEngine, PATCH_ID


def _engine_with(mesh: pv.PolyData) -> STLClipperEngine:
    """Load a PolyData directly, mirroring load_stl's post-read init."""
    eng = STLClipperEngine()
    mesh = mesh.triangulate()
    mesh.cell_data[PATCH_ID] = np.zeros(mesh.n_cells, dtype=np.int64)
    eng.original_mesh = mesh
    eng._trim_history.clear()
    eng.patch_names = {}
    eng._next_patch_id = 1
    eng._patch_normals = {}
    return eng


def _mobius() -> pv.PolyData:
    """Minimal 5-triangle Mobius band: non-orientable, every edge manifold.

    Re-winding can never make it consistent (there is no consistent winding),
    and there is nothing to strip (no non-manifold edge)."""
    pts = np.array([[np.cos(2 * np.pi * i / 5), np.sin(2 * np.pi * i / 5),
                     (i % 2) * 0.3] for i in range(5)], float)
    tris = [(0, 1, 2), (1, 3, 2), (2, 3, 4), (3, 0, 4), (4, 0, 1)]
    return pv.PolyData(pts, np.hstack([[3, *t] for t in tris]))


def _mobius_with_fin() -> pv.PolyData:
    """Mobius band plus one fin triangle on edge (0,1), making that edge
    non-manifold (3 faces). Re-winding leaves one stranded flip; removing the
    three faces on edge (0,1) cuts the loop and leaves an orientable strip."""
    pts = np.array([[np.cos(2 * np.pi * i / 5), np.sin(2 * np.pi * i / 5),
                     (i % 2) * 0.3] for i in range(5)] + [[0.3, -0.4, 0.6]], float)
    tris = [(0, 1, 2), (1, 3, 2), (2, 3, 4), (3, 0, 4), (4, 0, 1), (0, 1, 5)]
    return pv.PolyData(pts, np.hstack([[3, *t] for t in tris]))


def test_clean_mesh_is_left_intact():
    """A healthy closed mesh: winding fixed, nothing deleted, oriented outward."""
    eng = _engine_with(pv.Sphere())
    before = eng.current_mesh.n_cells
    msg = eng.repair_normals()
    assert eng.current_mesh.n_cells == before          # no faces removed
    r = eng.check_normals()
    assert r["consistent"] is True
    assert r["outward"] is True
    assert "removed" not in msg.lower()


def test_small_nonmanifold_block_is_auto_removed():
    """The ENT002 symptom: re-winding leaves a stranded flip; the few
    non-manifold faces are stripped automatically and winding becomes consistent."""
    eng = _engine_with(_mobius_with_fin())
    # Preconditions: a non-manifold edge exists and plain re-winding cannot fix it.
    assert len(eng.detect_nonmanifold_edges()) >= 1
    rewound = eng.current_mesh.compute_normals(
        cell_normals=False, point_normals=True, split_vertices=False,
        consistent_normals=True, auto_orient_normals=False)
    assert eng._count_winding_flips(rewound) > 0

    before = eng.current_mesh.n_cells
    msg = eng.repair_normals()

    assert eng.check_normals()["flipped_edges"] == 0
    assert eng.detect_nonmanifold_edges() == []
    assert eng.current_mesh.n_cells < before           # offending faces removed
    assert "removed" in msg.lower()


def test_auto_removal_is_undoable():
    eng = _engine_with(_mobius_with_fin())
    before = eng.current_mesh.n_cells
    eng.repair_normals()
    assert eng.current_mesh.n_cells < before
    assert eng.undo_trim() is True
    assert eng.current_mesh.n_cells == before          # fully restored


def test_declines_when_block_exceeds_cap():
    """With the cap set to 0, no faces may be removed: report honestly, delete
    nothing, and point the user at MeshFix."""
    eng = _engine_with(_mobius_with_fin())
    before = eng.current_mesh.n_cells
    msg = eng.repair_normals(max_nonmanifold_faces=0)
    assert eng.current_mesh.n_cells == before          # nothing removed
    assert "meshfix" in msg.lower()


def test_nonorientable_without_nonmanifold_is_reported_not_deleted():
    """A pure Mobius band has no non-manifold edge to strip — re-wind, keep all
    faces, and tell the user to inspect it."""
    eng = _engine_with(_mobius())
    before = eng.current_mesh.n_cells
    msg = eng.repair_normals()
    assert eng.current_mesh.n_cells == before          # nothing removed
    assert eng.check_normals()["flipped_edges"] > 0
    assert "manually" in msg.lower()
