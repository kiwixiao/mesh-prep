"""Face-removing operations must purge patch names whose faces vanished.

A name with zero faces was still written into meshDict, the BC files and
controlDict function objects while the exported STL had no such solid:
guaranteed meshing/solver failure (review findings 3 and 4).
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


def test_delete_cells_purges_emptied_patch():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    eng.clip_and_name("inlet", (0.3, 0, 0), (1, 0, 0))
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    cap_cells = np.nonzero(labels == 1)[0].tolist()
    eng.delete_cells(cap_cells)
    assert 1 not in eng.patch_names                    # name gone with faces
    assert 1 not in eng._patch_normals
    # partial delete keeps the name
    eng2 = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    eng2.clip_and_name("inlet", (0.3, 0, 0), (1, 0, 0))
    labels = np.asarray(eng2.current_mesh.cell_data[PATCH_ID])
    some = np.nonzero(labels == 1)[0][:5].tolist()
    eng2.delete_cells(some)
    assert 1 in eng2.patch_names


def test_reclip_swallowing_earlier_patch_purges_it():
    """Cap a branch, then re-clip the same branch shorter: the old cap's faces
    are cut away entirely and its name must not survive into the export."""
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    out = eng.clip_and_name("outlet1", (0.3, 0, 0), (-1, 0, 0))  # keep x<0.3
    assert out is not None and eng.patch_names == {1: "outlet1"}
    out = eng.clip_and_name("outlet2", (0.0, 0, 0), (-1, 0, 0))  # keep x<0: outlet1 gone
    assert out is not None
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    present = set(np.unique(labels).tolist())
    assert set(eng.patch_names) <= present                 # registry == mesh
    assert "outlet1" not in eng.patch_names.values()
    assert "outlet2" in eng.patch_names.values()


def test_undo_restores_purged_patch():
    eng = _engine_with(pv.Cylinder(resolution=60, capping=True).clean())
    eng.clip_and_name("inlet", (0.3, 0, 0), (1, 0, 0))
    labels = np.asarray(eng.current_mesh.cell_data[PATCH_ID])
    eng.delete_cells(np.nonzero(labels == 1)[0].tolist())
    assert 1 not in eng.patch_names
    assert eng.undo_trim() is True
    assert eng.patch_names == {1: "inlet"}                 # snapshot restored
