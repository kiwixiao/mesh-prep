"""
VMTK Centerline Computation

Compute vessel centerlines from a watertight surface mesh using VMTK's
vmtkCenterlines algorithm. Designed to be called from a background thread
(QThread) so the GUI stays responsive.
"""

import pyvista as pv
from vmtk import vmtkscripts


def compute_centerlines(surface_vtk, source_points, target_points):
    """
    Compute centerlines using VMTK.

    Args:
        surface_vtk: pyvista PolyData of the closed (watertight) surface mesh.
        source_points: list of [x, y, z] inlet coordinates.
        target_points: list of [x, y, z] outlet coordinates.

    Returns:
        pv.PolyData of the centerline polyline(s).

    Raises:
        RuntimeError: if VMTK fails to compute centerlines.
        ValueError: if inputs are invalid.
    """
    if not source_points or not target_points:
        raise ValueError("Need at least one source (inlet) and one target (outlet) point.")

    # Flatten coordinate lists: [x1, y1, z1, x2, y2, z2, ...]
    flat_source = [coord for pt in source_points for coord in pt]
    flat_target = [coord for pt in target_points for coord in pt]

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface_vtk
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = flat_source
    cl.TargetPoints = flat_target
    cl.ExitOnError = 0
    cl.Execute()

    if cl.Centerlines is None or cl.Centerlines.GetNumberOfCells() == 0:
        raise RuntimeError("VMTK produced no centerlines — check that inlet/outlet points lie on the surface.")

    return pv.wrap(cl.Centerlines)
