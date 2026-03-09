"""
VMTK Centerline Computation

Compute vessel centerlines from an open surface mesh using VMTK's
vmtkCenterlines algorithm with pointlist seed selector. Seed points
are provided as flat coordinate lists and VMTK snaps them to the
nearest surface point internally. Designed to be called from a
background thread (QThread) so the GUI stays responsive.
"""

import logging

import pyvista as pv
from vmtk import vmtkscripts

logger = logging.getLogger(__name__)


def compute_centerlines(surface_vtk, source_points, target_points):
    """
    Compute centerlines using VMTK with pointlist seed selector.

    Args:
        surface_vtk: pyvista PolyData — open surface.
        source_points: flat list of floats [x, y, z, ...] for inlet(s).
        target_points: flat list of floats [x, y, z, ...] for outlet(s).

    Returns:
        pv.PolyData of the centerline polyline(s).

    Raises:
        RuntimeError: if VMTK fails to compute centerlines.
        ValueError: if inputs are invalid.
    """
    if not source_points or not target_points:
        raise ValueError("Need at least one source and one target point.")

    logger.info("VMTK input — surface: %d points, %d cells", surface_vtk.n_points, surface_vtk.n_cells)
    logger.info("  source_points=%s, target_points=%s", source_points, target_points)

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface_vtk
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = source_points
    cl.TargetPoints = target_points
    cl.Execute()

    logger.info("VMTK output — Centerlines type=%s, n_points=%s, n_cells=%s",
                 type(cl.Centerlines).__name__,
                 cl.Centerlines.GetNumberOfPoints() if cl.Centerlines else 0,
                 cl.Centerlines.GetNumberOfCells() if cl.Centerlines else 0)

    if cl.Centerlines is None or cl.Centerlines.GetNumberOfCells() == 0:
        raise RuntimeError("VMTK produced no centerlines — check that seed points are on or near the surface.")

    result = pv.wrap(cl.Centerlines)
    logger.info("Wrapped result: %d points, %d cells", result.n_points, result.n_cells)
    return result
