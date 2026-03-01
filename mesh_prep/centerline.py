"""
VMTK Centerline Computation

Compute vessel centerlines from an open surface mesh using VMTK's
vmtkCenterlines algorithm with profileidlist seed selector. VMTK handles
capping internally. Designed to be called from a background thread
(QThread) so the GUI stays responsive.
"""

import logging

import pyvista as pv
from vmtk import vmtkscripts

logger = logging.getLogger(__name__)


def compute_centerlines(surface_vtk, source_ids, target_ids):
    """
    Compute centerlines using VMTK with profileidlist seed selector.

    Args:
        surface_vtk: pyvista PolyData — open surface (vmtkCenterlines caps internally).
        source_ids: list of open-profile indices to use as inlets.
        target_ids: list of open-profile indices to use as outlets.

    Returns:
        pv.PolyData of the centerline polyline(s).

    Raises:
        RuntimeError: if VMTK fails to compute centerlines.
        ValueError: if inputs are invalid.
    """
    if not source_ids or not target_ids:
        raise ValueError("Need at least one source and one target profile ID.")

    logger.info("VMTK input — surface: %d points, %d cells", surface_vtk.n_points, surface_vtk.n_cells)
    logger.info("  source_ids=%s, target_ids=%s", source_ids, target_ids)

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface_vtk
    cl.SeedSelectorName = "profileidlist"
    cl.SourceIds = source_ids
    cl.TargetIds = target_ids
    cl.Execute()

    logger.info("VMTK output — Centerlines type=%s, n_points=%s, n_cells=%s",
                 type(cl.Centerlines).__name__,
                 cl.Centerlines.GetNumberOfPoints() if cl.Centerlines else 0,
                 cl.Centerlines.GetNumberOfCells() if cl.Centerlines else 0)

    if cl.Centerlines is None or cl.Centerlines.GetNumberOfCells() == 0:
        raise RuntimeError("VMTK produced no centerlines — check that profile IDs map correctly to open boundaries.")

    result = pv.wrap(cl.Centerlines)
    logger.info("Wrapped result: %d points, %d cells", result.n_points, result.n_cells)
    return result
