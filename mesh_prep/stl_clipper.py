"""
STL Boundary Patch Clipper

Interactive tool to decompose a single-surface STL into named boundary patches
(inlet, outlet, wall) for OpenFOAM snappyHexMesh. Place clipping planes at
vessel branch endpoints, name each cross-section cap, and export multi-solid
ASCII STL files.

Usage:
    mesh-prep [path/to/file.stl]
"""

import json
import logging
import os
import subprocess
import sys
import time
import types
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import openfoam_case
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

logger = logging.getLogger(__name__)

# Name-based cap colors for semantic identification
INLET_COLOR = (0.9, 0.2, 0.2)       # red
OUTLET_COLOR = (0.2, 0.4, 0.9)      # blue
DEFAULT_CAP_COLOR = (0.2, 0.8, 0.3) # green (fallback)

WALL_COLOR = (0.82, 0.82, 0.82)
PREVIEW_COLOR = (0.0, 1.0, 1.0)  # cyan for slice preview


def _color_for_name(name: str) -> tuple:
    """Return color based on patch name: red for inlet, blue for outlet, green otherwise."""
    lower = name.lower()
    if "inlet" in lower:
        return INLET_COLOR
    elif "outlet" in lower:
        return OUTLET_COLOR
    return DEFAULT_CAP_COLOR


class CenterlineWorker(QThread):
    """Background thread for VMTK centerline computation."""
    result_ready = pyqtSignal(object)   # pv.PolyData
    failed = pyqtSignal(str)            # error message

    def __init__(self, surface_mesh, source_points, target_points):
        super().__init__()
        self.surface_mesh = surface_mesh
        self.source_points = source_points
        self.target_points = target_points

    def run(self):
        try:
            from mesh_prep.centerline import compute_centerlines
            result = compute_centerlines(self.surface_mesh, self.source_points, self.target_points)
            self.result_ready.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


# Container names used by run_docker.sh (must match generate_run_docker_sh)
_DOCKER_CONTAINERS = [
    "meshprep-pmesh", "meshprep-checkmesh", "meshprep-decompose",
    "meshprep-solver", "meshprep-reconstruct",
]

# Stage markers emitted by run_docker.sh (echo "=== <text> ===")
_STAGE_MARKERS = {
    "Running pMesh": "pMesh",
    "Running checkMesh": "checkMesh",
    "Decomposing mesh": "decomposePar",
    "Running pimpleFoam": "pimpleFoam",
    "Reconstructing": "reconstructPar",
    "Pulling Docker": "docker pull",
    "Done": "Done",
}


class OpenFOAMWorker(QThread):
    """Background thread that runs run_docker.sh and streams output."""

    log_line = pyqtSignal(str)           # batched log text
    stage_changed = pyqtSignal(str)      # current pipeline stage
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    _BATCH_INTERVAL = 0.1  # seconds between line-batch emissions

    def __init__(self, case_dir: str, nprocs: int):
        super().__init__()
        self._case_dir = case_dir
        self._nprocs = nprocs
        self._cancel = False
        self._process = None  # type: Optional[subprocess.Popen]

    def cancel(self):
        """Request cancellation: stop Docker containers then terminate bash."""
        self._cancel = True
        # Stop any running Docker container
        try:
            subprocess.run(
                ["docker", "stop"] + _DOCKER_CONTAINERS,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception:
            pass
        # Terminate the bash process
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run(self):
        script = os.path.join(self._case_dir, "run_docker.sh")
        try:
            self._process = subprocess.Popen(
                ["bash", script, str(self._nprocs)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self._case_dir,
            )
        except Exception as e:
            self.failed.emit(f"Failed to start: {e}")
            return

        batch = []
        last_emit = time.monotonic()
        for raw_line in self._process.stdout:
            if self._cancel:
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")

            # Detect stage changes from echo markers
            for marker, stage in _STAGE_MARKERS.items():
                if marker in line:
                    self.stage_changed.emit(stage)
                    break

            batch.append(line)
            now = time.monotonic()
            if now - last_emit >= self._BATCH_INTERVAL:
                self.log_line.emit("\n".join(batch))
                batch.clear()
                last_emit = now

        # Flush remaining lines
        if batch:
            self.log_line.emit("\n".join(batch))

        self._process.stdout.close()
        rc = self._process.wait()
        self._process = None

        if self._cancel:
            self.failed.emit("Cancelled by user")
        elif rc != 0:
            self.failed.emit(f"run_docker.sh exited with code {rc}")
        else:
            self.finished_ok.emit()


@dataclass
class ClipDefinition:
    """Holds a single clipping definition and its extracted cap mesh."""
    name: str
    origin: np.ndarray          # cut plane origin (extracted from box face)
    normal: np.ndarray          # cut plane normal — points toward KEPT side
    box_planes: list = field(default_factory=list)  # 6 planes as [(normal, point), ...]
    cap_mesh: Optional[pv.PolyData] = field(default=None, repr=False)
    color: tuple = (0.9, 0.2, 0.2)
    cap_kind: str = "closed"    # "closed": cap the cut (CFD); "open": leave a hole


def _points_in_polygon(xs, ys, polygon):
    """Vectorized even-odd (ray-casting) point-in-polygon test.

    xs, ys : 1-D float arrays of point coordinates (same length N).
    polygon: sequence of (x, y) vertices, length M >= 3.
    Returns a boolean array (N,) — True where the point is inside the polygon.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > ys) != (yj > ys)
        x_at_y = (xj - xi) * (ys - yi) / (yj - yi + 1e-30) + xi
        inside ^= crosses & (xs < x_at_y)
        j = i
    return inside


class STLClipperEngine:
    """
    Core mesh clipping logic — no Qt dependency.

    Workflow:
        1. load_stl(path)
        2. add_clip(name, origin, normal)  [repeat]
        3. recompute_all()
        4. export_combined_stl(path) or export_separate_stl(dir)
    """

    def __init__(self):
        self.original_mesh: Optional[pv.PolyData] = None
        self.clips: list[ClipDefinition] = []
        self._wall_mesh: Optional[pv.PolyData] = None
        self._trim_history: deque = deque(maxlen=10)

    def load_stl(self, filepath: str) -> pv.PolyData:
        mesh = pv.read(filepath)
        if not isinstance(mesh, pv.PolyData):
            raise ValueError(f"Expected PolyData, got {type(mesh).__name__}")
        self.original_mesh = mesh
        self.clips.clear()
        self._wall_mesh = mesh.copy()
        return mesh

    # ------------------------------------------------------------------
    # Clipping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_loops_by_box(loops: pv.PolyData, box_planes_data: list) -> pv.PolyData:
        """Keep only loops whose centroid is inside the box. Binary filter — no cutting."""
        if loops.n_cells == 0:
            return loops
        kept = []
        points = loops.points
        for cid in range(loops.n_cells):
            cell_pts = points[loops.get_cell(cid).point_ids]
            centroid = cell_pts.mean(axis=0)
            inside = all(np.dot(n, centroid - p) <= 0 for n, p in box_planes_data)
            if inside:
                kept.append(cid)
        if not kept:
            return pv.PolyData()
        return loops.extract_cells(kept).extract_surface()

    @staticmethod
    def _generate_cap(original_mesh, origin, normal, box_planes_data=None):
        """Generate cap by slicing the original mesh and filling the contour.

        1. slice() → contour lines
        2. vtkStripper → connect into closed loops
        3. Filter loops by box (binary: keep whole loops inside, discard rest)
        4. vtkContourTriangulator → fill into disc
        5. Fallback: delaunay_2d()
        """
        if original_mesh is None or original_mesh.n_points < 3:
            return pv.PolyData()

        # Step 1: Slice → contour
        contour = original_mesh.slice(normal=normal, origin=origin)

        if contour.n_cells == 0:
            return pv.PolyData()

        # Step 2: Connect segments → closed loops
        stripper = vtk.vtkStripper()
        stripper.SetInputData(contour)
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        loops = pv.wrap(stripper.GetOutput())

        # Step 3: Filter loops by box (binary: keep entire loops inside box)
        if box_planes_data:
            loops = STLClipperEngine._filter_loops_by_box(loops, box_planes_data)

        # Step 4: Fill loops → triangulated disc
        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputData(loops)
        triangulator.Update()
        cap = pv.wrap(triangulator.GetOutput())

        # Step 5: Fallback
        if cap.n_cells == 0:
            cap = loops.delaunay_2d()

        return cap

    @staticmethod
    def clip_with_plane(
        mesh: pv.PolyData, origin: np.ndarray, normal: np.ndarray
    ) -> pv.PolyData:
        """Clip mesh with a plane. Returns the remaining mesh.

        Normal points toward the KEPT side.
        """
        vtk_plane = vtk.vtkPlane()
        vtk_plane.SetOrigin(*origin)
        vtk_plane.SetNormal(*normal)

        clipper = vtk.vtkPolyDataPlaneClipper()
        clipper.SetInputData(mesh)
        clipper.SetPlane(vtk_plane)
        clipper.SetCapping(False)
        clipper.Update()

        clipped = pv.wrap(clipper.GetOutput())
        if clipped.n_cells == 0:
            raise RuntimeError("Clipping produced an empty mesh — plane may not intersect the geometry.")

        return clipped

    @staticmethod
    def clip_with_box(mesh, box_planes_data, cut_origin, cut_normal):
        """Box-scoped wall cut using boolean intersection of box + plane.

        Removes geometry that is BOTH inside the box AND on the discard side
        of the plane. The box acts as a scope limiter, not a physical cutter.
        """
        # Build box implicit function
        box_planes = vtk.vtkPlanes()
        normals_array = vtk.vtkDoubleArray()
        normals_array.SetNumberOfComponents(3)
        points = vtk.vtkPoints()
        for normal, point in box_planes_data:
            normals_array.InsertNextTuple3(*normal)
            points.InsertNextPoint(*point)
        box_planes.SetNormals(normals_array)
        box_planes.SetPoints(points)

        # Build plane implicit function (normal points to KEEP side)
        vtk_plane = vtk.vtkPlane()
        vtk_plane.SetOrigin(*cut_origin)
        vtk_plane.SetNormal(*cut_normal)

        # Boolean intersection: inside box AND discard side of plane
        boolean_func = vtk.vtkImplicitBoolean()
        boolean_func.SetOperationTypeToIntersection()
        boolean_func.AddFunction(box_planes)
        boolean_func.AddFunction(vtk_plane)

        # Clip: remove the intersection, keep everything else
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(boolean_func)
        clipper.SetInsideOut(False)
        clipper.Update()

        result = pv.wrap(clipper.GetOutput())
        if result.n_cells == 0:
            raise RuntimeError("Clipping produced an empty mesh.")
        return result

    def add_clip(self, name: str, origin: np.ndarray, normal: np.ndarray,
                 box_planes_data: list = None,
                 cap_kind: str = "closed") -> ClipDefinition:
        if cap_kind not in ("closed", "open"):
            raise ValueError(f"cap_kind must be 'closed' or 'open', got {cap_kind!r}")
        color = _color_for_name(name)
        clip_def = ClipDefinition(
            name=name,
            origin=np.asarray(origin, dtype=float),
            normal=np.asarray(normal, dtype=float),
            box_planes=box_planes_data if box_planes_data else [],
            color=color,
            cap_kind=cap_kind,
        )
        self.clips.append(clip_def)
        self.recompute_all()
        return clip_def

    def set_cap_kind(self, index: int, cap_kind: str) -> None:
        """Switch a clip between 'closed' (with cap) and 'open' (no cap)."""
        if cap_kind not in ("closed", "open"):
            raise ValueError(f"cap_kind must be 'closed' or 'open', got {cap_kind!r}")
        if 0 <= index < len(self.clips):
            self.clips[index].cap_kind = cap_kind
            self.recompute_all()

    def remove_clip(self, index: int):
        if 0 <= index < len(self.clips):
            self.clips.pop(index)
            self.recompute_all()

    def rename_clip(self, index: int, new_name: str):
        if 0 <= index < len(self.clips):
            self.clips[index].name = new_name
            self.clips[index].color = _color_for_name(new_name)

    def recompute_all(self):
        """Apply all clips from the original mesh using box-scoped cutting."""
        if self.original_mesh is None:
            return None
        working = self.original_mesh.copy()
        for clip_def in self.clips:
            try:
                if clip_def.box_planes:
                    working = self.clip_with_box(
                        working, clip_def.box_planes, clip_def.origin, clip_def.normal
                    )
                else:
                    working = self.clip_with_plane(
                        working, clip_def.origin, clip_def.normal
                    )
                if clip_def.cap_kind == "closed":
                    clip_def.cap_mesh = self._generate_cap(
                        self.original_mesh, clip_def.origin, clip_def.normal,
                        clip_def.box_planes if clip_def.box_planes else None
                    )
                else:
                    # Open profile: leave a hole at the cut, no cap mesh.
                    clip_def.cap_mesh = None
            except RuntimeError:
                clip_def.cap_mesh = pv.PolyData()
        self._wall_mesh = working
        return self._wall_mesh

    def trim_by_screen_polygon(self, polygon_xy, view_matrix, viewport):
        """Permanently delete cells of original_mesh whose centroid projects
        inside the freehand outline polygon_xy (through-model). Mutates the base
        mesh and re-applies clips. Returns the new _wall_mesh, or None on a no-op.

        polygon_xy : list[(x, y)] display-space points (logical pixels)
        view_matrix: 4x4 array-like, world->clip (camera composite projection)
        viewport   : (width, height) in logical pixels
        Raises ValueError if the selection would delete the entire mesh.
        """
        if self.original_mesh is None:
            return None
        poly = np.asarray(polygon_xy, dtype=float)
        if poly.shape[0] < 3:
            return None
        matrix = np.asarray(view_matrix, dtype=float).reshape(4, 4)
        width, height = viewport
        centers = self.original_mesh.cell_centers().points          # (N, 3)
        n = centers.shape[0]
        homog = np.hstack([centers, np.ones((n, 1))])               # (N, 4)
        clip = homog @ matrix.T                                     # (N, 4)
        w = clip[:, 3].copy()
        w[w == 0] = 1e-12
        ndc = clip[:, :3] / w[:, None]
        disp_x = (ndc[:, 0] * 0.5 + 0.5) * width
        disp_y = (ndc[:, 1] * 0.5 + 0.5) * height                  # bottom-left origin
        inside = _points_in_polygon(disp_x, disp_y, poly)
        n_inside = int(inside.sum())
        if n_inside == 0:
            return None
        if n_inside == n:
            raise ValueError("Trim would delete the entire mesh")
        self._trim_history.append(self.original_mesh.copy())
        keep_ids = np.where(~inside)[0]
        self.original_mesh = self.original_mesh.extract_cells(keep_ids).extract_surface()
        return self.recompute_all()

    def undo_trim(self) -> bool:
        """Restore the mesh from before the most recent trim and re-apply clips.
        Returns True if a state was restored, False if there is no trim history."""
        if not self._trim_history:
            return False
        self.original_mesh = self._trim_history.pop()
        self.recompute_all()
        return True

    def get_wall_mesh(self) -> Optional[pv.PolyData]:
        return self._wall_mesh

    def geometry_quality(self) -> dict:
        """Return geometry quality metrics for the current wall mesh."""
        wall = self._wall_mesh
        if wall is None or wall.n_cells == 0:
            return {"open_edges": 0, "open_profiles": 0, "boundary_mesh": None}

        boundary = wall.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False,
        )
        n_open_edges = boundary.n_cells if boundary else 0

        if n_open_edges > 0 and boundary.n_cells > 0:
            connected = boundary.connectivity()
            n_profiles = int(connected["RegionId"].max()) + 1
        else:
            n_profiles = 0

        # Combined manifold check: wall + all caps, tolerance merge
        if self.clips:
            combined = wall.copy()
            for clip_def in self.clips:
                if clip_def.cap_mesh and clip_def.cap_mesh.n_cells > 0:
                    combined = combined.merge(clip_def.cap_mesh)
            mesh_diag = np.linalg.norm(
                np.ptp(np.array(wall.bounds).reshape(3, 2), axis=1)
            )
            combined = combined.clean(tolerance=mesh_diag * 1e-6)

            non_manifold = combined.extract_feature_edges(
                boundary_edges=False, feature_edges=False,
                manifold_edges=False, non_manifold_edges=True,
            )
            n_nm = non_manifold.n_cells
            is_mf = combined.is_manifold
        else:
            # No clips yet — check original mesh for defects only
            non_manifold = wall.extract_feature_edges(
                boundary_edges=False, feature_edges=False,
                manifold_edges=False, non_manifold_edges=True,
            )
            n_nm = non_manifold.n_cells
            is_mf = None  # indeterminate without caps

        return {
            "open_edges": n_open_edges,
            "open_profiles": n_profiles,
            "boundary_mesh": boundary,
            "non_manifold_edges": n_nm,
            "is_manifold": is_mf,
            "non_manifold_mesh": non_manifold,
        }

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def _apply_repair(self, repaired_mesh) -> None:
        """Replace original mesh with repaired version, clear all clips."""
        self.original_mesh = repaired_mesh
        self.clips.clear()
        self._wall_mesh = repaired_mesh.copy()

    def repair_clean(self) -> str:
        """Remove duplicate points and degenerate triangles from original mesh."""
        if self.original_mesh is None:
            return "No mesh loaded."
        before = self.original_mesh.n_cells
        repaired = self.original_mesh.clean()
        after = repaired.n_cells
        self._apply_repair(repaired)
        return f"Cleaned: {before} \u2192 {after} faces ({before - after} removed)"

    def repair_normals(self) -> str:
        """Fix inconsistent face normals on original mesh."""
        if self.original_mesh is None:
            return "No mesh loaded."
        repaired = self.original_mesh.compute_normals(
            cell_normals=False, point_normals=True,
            split_vertices=False, consistent_normals=True,
            auto_orient_normals=False,
        )
        self._apply_repair(repaired)
        return "Normals fixed (consistent winding)"

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def _polydata_to_ascii_stl_block(mesh: pv.PolyData, solid_name: str, scale_factor: float = 1.0) -> str:
        """Convert a PolyData to an ASCII STL solid block."""
        if mesh is None or mesh.n_cells == 0:
            return f"solid {solid_name}\nendsolid {solid_name}\n"

        # Ensure we have triangle faces and normals
        tri = mesh.triangulate()
        tri.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        lines = [f"solid {solid_name}"]
        normals = tri.cell_normals  # unit vectors — do NOT scale
        points = tri.points * scale_factor
        # Extract face connectivity
        faces = tri.faces
        idx = 0
        face_i = 0
        while idx < len(faces):
            n_pts = faces[idx]
            if n_pts != 3:
                idx += n_pts + 1
                face_i += 1
                continue
            v0, v1, v2 = faces[idx + 1], faces[idx + 2], faces[idx + 3]
            nx, ny, nz = normals[face_i]
            lines.append(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}")
            lines.append("    outer loop")
            for vi in (v0, v1, v2):
                x, y, z = points[vi]
                lines.append(f"      vertex {x:.6e} {y:.6e} {z:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
            idx += n_pts + 1
            face_i += 1
        lines.append(f"endsolid {solid_name}")
        return "\n".join(lines) + "\n"

    def export_combined_stl(self, filepath: str, scale_factor: float = 1.0):
        """Write a single ASCII STL with multiple solid blocks.

        Open-profile clips contribute no cap solid; their boundary is left as
        an open hole on the wall solid.
        """
        blocks = []
        for clip_def in self.clips:
            if clip_def.cap_kind != "closed":
                continue
            blocks.append(self._polydata_to_ascii_stl_block(clip_def.cap_mesh, clip_def.name, scale_factor))
        blocks.append(self._polydata_to_ascii_stl_block(self._wall_mesh, "wall", scale_factor))
        with open(filepath, "w") as f:
            f.write("".join(blocks))

    def export_clipped_surface_stl(
        self,
        filepath: str,
        scale_factor: float = 1.0,
        solid_name: str = "clipped",
    ) -> dict:
        """Save the clipped surface as a SINGLE-solid ASCII STL.

        The wall mesh is always included.  Closed-profile clips contribute
        their caps merged into the same solid (creating a watertight surface
        at those cuts).  Open-profile clips contribute nothing — their cut
        leaves a hole on the wall.

        Intended for downstream tools that expect a single triangulated
        surface (e.g. VMTK), not multi-patch CFD meshers.

        Returns a dict with face counts and clip categorisation.
        """
        if self._wall_mesh is None or self._wall_mesh.n_cells == 0:
            raise RuntimeError("No wall mesh to export.")

        # Use vtkAppendPolyData so the result stays a PolyData (merge() can
        # promote to UnstructuredGrid, which _polydata_to_ascii_stl_block
        # cannot consume).
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(self._wall_mesh)
        n_caps = 0
        for clip_def in self.clips:
            if clip_def.cap_kind == "closed" \
                    and clip_def.cap_mesh is not None \
                    and clip_def.cap_mesh.n_cells > 0:
                appender.AddInputData(clip_def.cap_mesh)
                n_caps += 1
        appender.Update()
        combined = pv.wrap(appender.GetOutput())

        block = self._polydata_to_ascii_stl_block(combined, solid_name, scale_factor)
        with open(filepath, "w") as f:
            f.write(block)

        return {
            "filepath": filepath,
            "n_wall_cells": int(self._wall_mesh.n_cells),
            "n_caps_merged": n_caps,
            "n_open_clips": sum(1 for c in self.clips if c.cap_kind == "open"),
            "n_total_cells": int(combined.n_cells),
        }

    def export_clip_planes(self, filepath: str):
        """Save clip plane origins and normals as JSON."""
        data = {
            "clips": [
                {
                    "name": c.name,
                    "origin": c.origin.tolist(),
                    "normal": c.normal.tolist(),
                }
                for c in self.clips
            ]
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def export_openfoam(self, case_dir: str, stl_filename: str, scale_factor: float = 1.0) -> dict:
        """
        Export for OpenFOAM: multi-solid STL to case_dir/constant/triSurface/
        and clip plane data as JSON to case_dir/.

        Returns dict with output paths for confirmation.
        """
        tri_dir = os.path.join(case_dir, "constant", "triSurface")
        os.makedirs(tri_dir, exist_ok=True)

        stl_path = os.path.join(tri_dir, stl_filename)
        self.export_combined_stl(stl_path, scale_factor)

        planes_path = os.path.join(case_dir, "clip_planes.json")
        self.export_clip_planes(planes_path)

        return {"stl_path": stl_path, "planes_path": planes_path}

    def export_openfoam_case(
        self,
        case_dir: str,
        stl_filename: str,
        centerline_mesh=None,
        scale_factor: float = 1.0,
        template_params: dict = None,
        simulation_type: str = "les",
    ) -> dict:
        """Export a complete OpenFOAM case directory for cfMesh pMesh.

        Writes the multi-solid STL to ``constant/triSurface/``, generates
        ``system/meshDict`` for cfMesh pMesh, and ``run_docker.sh`` for
        Docker-based meshing + solving.  pMesh creates patch names directly
        from STL solid names (no geometry prefix).

        Parameters
        ----------
        case_dir : str
            Root of the OpenFOAM case directory.
        stl_filename : str
            STL file name (written into constant/triSurface/).
        centerline_mesh : pyvista.PolyData, optional
            Centerline polyline (kept for API compatibility).
        template_params : dict, optional
            When provided, overrides default generator values and copies static
            template files.  Keys: mesh, solver, fluid, turbulence, inlet,
            decompose (mirrors config.json structure).

        Returns
        -------
        dict
            Paths of all written files, keyed by category.
        """
        import shutil

        # 1. Existing export: STL + clip_planes.json
        base_result = self.export_openfoam(case_dir, stl_filename, scale_factor)

        # 2. Collect patch metadata
        stl_stem = stl_filename.rsplit(".", 1)[0] if "." in stl_filename else stl_filename
        patch_names = [c.name for c in self.clips] + ["wall"]
        inlet_normals = {
            c.name: tuple(-c.normal)    # Flip to STL/CFD convention: outward-pointing
            for c in self.clips
            if "inlet" in c.name.lower()
        }

        # 3. Create subdirectories
        dirs = {
            "system": os.path.join(case_dir, "system"),
            "constant": os.path.join(case_dir, "constant"),
            "zero": os.path.join(case_dir, "0"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        written = dict(base_result)

        # Extract template params (or empty dicts for backward compat)
        p = template_params or {}
        mesh_p = p.get("mesh", {})
        solver_p = p.get("solver", {})
        fluid_p = p.get("fluid", {})
        turb_p = p.get("turbulence", {})
        inlet_p = p.get("inlet", {})
        decomp_p = p.get("decompose", {})

        # 4. Write system/ dictionaries
        #    Static files (fvSchemes, fvSolution, meshQualityDict) are copied
        #    from the template directory when template_params is provided.
        system_files = {
            "meshDict": openfoam_case.generate_mesh_dict(
                stl_filename, patch_names,
                max_cell_size=mesh_p.get("maxCellSize", 0.8),
                boundary_cell_size=mesh_p.get("boundaryCellSize", 0.35),
                num_layers=mesh_p.get("nLayers", 3),
                thickness_ratio=mesh_p.get("thicknessRatio", 0.5),
                wall_cell_size=mesh_p.get("wallCellSize"),
            ),
            "controlDict": openfoam_case.generate_control_dict(
                outlet_patches=[
                    c.name for c in self.clips if "outlet" in c.name.lower()
                ],
                geo_name="",
                end_time=solver_p.get("endTime", 1.6),
                delta_t=solver_p.get("deltaT", 1e-5),
                write_interval=solver_p.get("writeInterval", 0.01),
                max_co=solver_p.get("maxCo", 0.5),
                max_delta_t=solver_p.get("maxDeltaT", 1e-3),
                simulation_type=simulation_type,
            ),
            "decomposeParDict": openfoam_case.generate_decompose_par_dict(
                n_procs=decomp_p.get("nProcs", 4),
            ),
        }

        # Copy static template files if template_params provided
        if template_params is not None:
            tmpl_name = "rans_aorta" if simulation_type == "rans" else "les_aorta"
            tmpl_dir = openfoam_case.get_template_dir(tmpl_name)
            for static_name in ("fvSchemes", "fvSolution", "meshQualityDict"):
                src = tmpl_dir / static_name
                if src.exists():
                    dst = os.path.join(dirs["system"], static_name)
                    shutil.copy2(str(src), dst)
                    written[static_name] = dst
        else:
            system_files["fvSchemes"] = openfoam_case.generate_fv_schemes(
                simulation_type=simulation_type,
            )
            system_files["fvSolution"] = openfoam_case.generate_fv_solution(
                simulation_type=simulation_type,
            )
            system_files["meshQualityDict"] = openfoam_case.generate_mesh_quality_dict()

        for name, content in system_files.items():
            path = os.path.join(dirs["system"], name)
            with open(path, "w") as f:
                f.write(content)
            written[name] = path

        # 5. Write constant/ dictionaries
        const_files = {
            "transportProperties": openfoam_case.generate_transport_properties(
                nu=fluid_p.get("nu", 3.3e-6),
            ),
            "turbulenceProperties": openfoam_case.generate_turbulence_properties(
                simulation_type=simulation_type,
                cs=turb_p.get("Cs", 0.1),
            ),
        }
        for name, content in const_files.items():
            path = os.path.join(dirs["constant"], name)
            with open(path, "w") as f:
                f.write(content)
            written[name] = path

        # volumetricFlowRate.csv — copy from template if available, else generate
        csv_path = os.path.join(dirs["constant"], "volumetricFlowRate.csv")
        if template_params is not None:
            tmpl_csv = tmpl_dir / "volumetricFlowRate.csv"
            if tmpl_csv.exists():
                shutil.copy2(str(tmpl_csv), csv_path)
            else:
                with open(csv_path, "w") as f:
                    f.write(openfoam_case.generate_volumetric_flow_rate_csv())
        else:
            with open(csv_path, "w") as f:
                f.write(openfoam_case.generate_volumetric_flow_rate_csv())
        written["volumetricFlowRate.csv"] = csv_path

        # 6. Write 0/ boundary conditions (no geo_name prefix with pMesh)
        vel_mag = inlet_p.get("velocityMagnitude", 0.3)
        bc_files = {
            "p": openfoam_case.generate_p(patch_names, geo_name=""),
            "U": openfoam_case.generate_U(
                patch_names, inlet_normals, geo_name="",
                velocity_magnitude=vel_mag,
                simulation_type=simulation_type,
            ),
            "nut": openfoam_case.generate_nut(
                patch_names, geo_name="",
                simulation_type=simulation_type,
            ),
        }
        if simulation_type == "rans":
            bc_files["k"] = openfoam_case.generate_k(
                patch_names, geo_name="",
                k_value=turb_p.get("k", 3.375e-4),
            )
            bc_files["omega"] = openfoam_case.generate_omega(
                patch_names, geo_name="",
                omega_value=turb_p.get("omega", 30),
            )
        for name, content in bc_files.items():
            path = os.path.join(dirs["zero"], name)
            with open(path, "w") as f:
                f.write(content)
            written[name] = path

        # 7. Write shell scripts (executable)
        scripts = {
            "env.sh": openfoam_case.generate_env_sh(),
            "Allrun": openfoam_case.generate_allrun(meshing_method="pmesh"),
            "Allclean": openfoam_case.generate_allclean(meshing_method="pmesh"),
            "run_docker.sh": openfoam_case.generate_run_docker_sh(),
        }
        for name, content in scripts.items():
            path = os.path.join(case_dir, name)
            with open(path, "w") as f:
                f.write(content)
            os.chmod(path, 0o755)
            written[name] = path

        # 8. ParaView visualization: .foam file + visualize.py
        foam_path = os.path.join(case_dir, f"{stl_stem}.foam")
        open(foam_path, "w").close()  # empty file — ParaView convention
        written[".foam"] = foam_path

        clips_data = [
            {
                "name": c.name,
                "origin": tuple(float(v) for v in c.origin),
                "normal": tuple(float(v) for v in c.normal),
            }
            for c in self.clips
        ]
        viz_content = openfoam_case.generate_visualize_py(stl_filename, clips_data)
        viz_path = os.path.join(case_dir, "visualize.py")
        with open(viz_path, "w") as f:
            f.write(viz_content)
        os.chmod(viz_path, 0o755)
        written["visualize.py"] = viz_path

        # diagnose_patches.py (uses direct names, no geo prefix)
        diag_content = openfoam_case.generate_diagnose_patches_py(
            stl_filename, patch_names, inlet_normals, geo_name="",
            velocity_magnitude=vel_mag,
        )
        diag_path = os.path.join(case_dir, "diagnose_patches.py")
        with open(diag_path, "w") as f:
            f.write(diag_content)
        os.chmod(diag_path, 0o755)
        written["diagnose_patches.py"] = diag_path

        # plot_residuals.py (solver convergence monitoring)
        resid_content = openfoam_case.generate_plot_residuals_py()
        resid_path = os.path.join(case_dir, "plot_residuals.py")
        with open(resid_path, "w") as f:
            f.write(resid_content)
        os.chmod(resid_path, 0o755)
        written["plot_residuals.py"] = resid_path

        return written

    def export_separate_stl(self, output_dir: str, scale_factor: float = 1.0):
        """Write one STL file per patch into output_dir.

        Open-profile clips contribute no cap file.
        """
        os.makedirs(output_dir, exist_ok=True)
        for clip_def in self.clips:
            if clip_def.cap_kind != "closed":
                continue
            path = os.path.join(output_dir, f"{clip_def.name}.stl")
            with open(path, "w") as f:
                f.write(self._polydata_to_ascii_stl_block(clip_def.cap_mesh, clip_def.name, scale_factor))
        wall_path = os.path.join(output_dir, "wall.stl")
        with open(wall_path, "w") as f:
            f.write(self._polydata_to_ascii_stl_block(self._wall_mesh, "wall", scale_factor))


class STLClipperApp(QMainWindow):
    """Qt GUI with embedded PyVista viewport and control panel."""

    def __init__(self, initial_file: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("STL Boundary Patch Clipper")
        self.resize(1400, 900)

        self.engine = STLClipperEngine()
        self._loaded_filepath = None
        self._plane_widget_active = False       # plane widget on screen
        self._constraint_box_active = False     # optional constraint box on screen
        self._current_plane_origin = None       # from plane widget callback
        self._current_plane_normal = None       # from plane widget callback
        self._current_box_planes_data = None    # from constraint box callback (optional)
        self._preview_actor = None
        self._arrow_actor = None
        self._plane_confirmed = False       # plane locked after user confirms position
        self._static_plane_actor = None     # static visual replacing interactive widget

        # Box constraint state (VTK box widget + spinbox/slider dual-input)
        self._box_center = None              # np.array([cx, cy, cz])
        self._box_rotation_deg = None        # np.array([rx, ry, rz]) degrees
        self._box_half_extents = None        # np.array([hx, hy, hz])
        self._box_initial_center = None      # center used in PlaceWidget
        self._box_initial_half_extents = None  # half-extents used in PlaceWidget
        self._box_actor = None               # wireframe box pyvista actor (fallback)
        self._updating_box_controls = False  # recursion guard

        # Centerline state
        self._centerline_mesh = None          # pv.PolyData from VMTK
        self._centerline_worker = None        # CenterlineWorker QThread
        self._boundary_mesh = None            # pv.PolyData for boundary edge visualization
        self._non_manifold_mesh = None        # pv.PolyData for non-manifold edge visualization

        # OpenFOAM runner state
        self._last_case_dir = None            # set after successful export
        self._openfoam_worker = None          # OpenFOAMWorker QThread

        self._build_ui()
        self._build_menu()
        self._update_button_states()

        self._trim_undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._trim_undo_shortcut.activated.connect(self._undo_trim)

        if initial_file and os.path.isfile(initial_file):
            self._load_file(initial_file)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 3D viewport
        self.plotter = QtInteractor(central)
        self.plotter.set_background("black")
        self.plotter.enable_parallel_projection()
        # QtInteractor starts with no lights, so surfaces render as a flat,
        # unshaded silhouette. Add a ParaView-style 3-light kit so geometry is
        # properly shaded.
        self.plotter.enable_lightkit()

        # Tabbed control panel
        self._tab_widget = QTabWidget()
        self._tab_widget.setMinimumWidth(220)

        # ── Tab 1: Clipping ──
        clip_scroll = QScrollArea()
        clip_scroll.setWidgetResizable(True)
        clip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        clip_widget = QWidget()
        panel = QVBoxLayout(clip_widget)
        clip_scroll.setWidget(clip_widget)
        self._tab_widget.addTab(clip_scroll, "Clipping")

        # --- Load ---
        self.btn_load = QPushButton("Load STL...")
        self.btn_load.clicked.connect(self._on_load)
        panel.addWidget(self.btn_load)

        panel.addWidget(self._separator("Clip Plane"))

        self.btn_add_plane = QPushButton("Add Clip Plane")
        self.btn_add_plane.clicked.connect(self._on_add_plane)
        panel.addWidget(self.btn_add_plane)

        self.btn_confirm_plane = QPushButton("Confirm Plane")
        self.btn_confirm_plane.clicked.connect(self._on_confirm_plane)
        panel.addWidget(self.btn_confirm_plane)

        self.btn_add_constraint = QPushButton("Constrain to Box")
        self.btn_add_constraint.clicked.connect(self._on_add_constraint)
        panel.addWidget(self.btn_add_constraint)

        btn_row = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Clip")
        self.btn_confirm.clicked.connect(self._on_confirm_clip)
        btn_row.addWidget(self.btn_confirm)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_cancel)
        panel.addLayout(btn_row)

        self.btn_flip = QPushButton("Flip Normal")
        self.btn_flip.clicked.connect(self._on_flip_normal)
        panel.addWidget(self.btn_flip)

        # --- Manual Plane Controls (trackpad-friendly) ---
        self._updating_controls = False
        self._plane_controls_box = QGroupBox("Plane Position / Normal")
        pc_layout = QVBoxLayout()

        # Origin X / Y / Z spinboxes
        origin_row = QHBoxLayout()
        self._spin_x = QDoubleSpinBox()
        self._spin_y = QDoubleSpinBox()
        self._spin_z = QDoubleSpinBox()
        for label, spin in [("X", self._spin_x), ("Y", self._spin_y), ("Z", self._spin_z)]:
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            col.addWidget(spin)
            origin_row.addLayout(col)
        pc_layout.addLayout(origin_row)

        # Nudge along normal
        nudge_row = QHBoxLayout()
        nudge_row.addWidget(QLabel("Step:"))
        self._spin_step = QDoubleSpinBox()
        self._spin_step.setRange(0.01, 1000.0)
        self._spin_step.setValue(1.0)
        self._spin_step.setDecimals(2)
        nudge_row.addWidget(self._spin_step)
        btn_nudge_fwd = QPushButton("+N")
        btn_nudge_fwd.setToolTip("Nudge plane forward along normal")
        btn_nudge_fwd.clicked.connect(lambda: self._on_nudge(+1))
        nudge_row.addWidget(btn_nudge_fwd)
        btn_nudge_back = QPushButton("-N")
        btn_nudge_back.setToolTip("Nudge plane backward along normal")
        btn_nudge_back.clicked.connect(lambda: self._on_nudge(-1))
        nudge_row.addWidget(btn_nudge_back)
        pc_layout.addLayout(nudge_row)

        # Normal orientation (spherical coordinates)
        pc_layout.addWidget(QLabel("Tilt (elevation):"))
        self._slider_elev = QSlider(Qt.Horizontal)
        self._slider_elev.setRange(0, 180)      # 0=+Z, 90=XY plane, 180=-Z
        self._slider_elev.setValue(0)
        self._elev_label = QLabel("0°")
        elev_row = QHBoxLayout()
        elev_row.addWidget(self._slider_elev)
        elev_row.addWidget(self._elev_label)
        pc_layout.addLayout(elev_row)

        pc_layout.addWidget(QLabel("Spin (azimuth):"))
        self._slider_azim = QSlider(Qt.Horizontal)
        self._slider_azim.setRange(0, 360)      # rotation in XY plane
        self._slider_azim.setValue(0)
        self._azim_label = QLabel("0°")
        azim_row = QHBoxLayout()
        azim_row.addWidget(self._slider_azim)
        azim_row.addWidget(self._azim_label)
        pc_layout.addLayout(azim_row)

        # Quick axis presets (set both sliders at once)
        preset_row = QHBoxLayout()
        for label, elev, azim in [
            ("+X", 90, 0), ("-X", 90, 180),
            ("+Y", 90, 90), ("-Y", 90, 270),
            ("+Z", 0, 0), ("-Z", 180, 0),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, e=elev, a=azim: self._on_normal_preset_angles(e, a))
            preset_row.addWidget(btn)
        pc_layout.addLayout(preset_row)

        self._plane_controls_box.setLayout(pc_layout)
        panel.addWidget(self._plane_controls_box)
        self._plane_controls_box.setVisible(False)

        # --- Box Constraint Controls (trackpad-friendly) ---
        self._box_controls_box = QGroupBox("Box Constraint")
        bc_layout = QVBoxLayout()

        # Center X / Y / Z  (spinbox + slider per axis)
        center_row = QHBoxLayout()
        self._spin_box_cx = QDoubleSpinBox()
        self._spin_box_cy = QDoubleSpinBox()
        self._spin_box_cz = QDoubleSpinBox()
        self._slider_box_cx = QSlider(Qt.Horizontal)
        self._slider_box_cy = QSlider(Qt.Horizontal)
        self._slider_box_cz = QSlider(Qt.Horizontal)
        for label, spin, slider in [
            ("CX", self._spin_box_cx, self._slider_box_cx),
            ("CY", self._spin_box_cy, self._slider_box_cy),
            ("CZ", self._spin_box_cz, self._slider_box_cz),
        ]:
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            slider.setRange(-1000, 1000)  # default; updated per mesh
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            col.addWidget(spin)
            col.addWidget(slider)
            center_row.addLayout(col)
        bc_layout.addWidget(QLabel("Center:"))
        bc_layout.addLayout(center_row)

        # Rotation Rx / Ry / Rz (degrees)  (spinbox + slider per axis)
        rot_row = QHBoxLayout()
        self._spin_box_rx = QDoubleSpinBox()
        self._spin_box_ry = QDoubleSpinBox()
        self._spin_box_rz = QDoubleSpinBox()
        self._slider_box_rx = QSlider(Qt.Horizontal)
        self._slider_box_ry = QSlider(Qt.Horizontal)
        self._slider_box_rz = QSlider(Qt.Horizontal)
        for label, spin, slider in [
            ("Rx", self._spin_box_rx, self._slider_box_rx),
            ("Ry", self._spin_box_ry, self._slider_box_ry),
            ("Rz", self._spin_box_rz, self._slider_box_rz),
        ]:
            spin.setRange(0, 360)
            spin.setDecimals(1)
            spin.setSingleStep(5.0)
            spin.setWrapping(True)
            slider.setRange(0, 3600)  # 0.0° - 360.0°, step = 0.1°
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            col.addWidget(spin)
            col.addWidget(slider)
            rot_row.addLayout(col)
        bc_layout.addWidget(QLabel("Rotation (deg):"))
        bc_layout.addLayout(rot_row)

        # Size W / H / D (full dimensions along local axes)  (spinbox + slider per axis)
        size_row = QHBoxLayout()
        self._spin_box_w = QDoubleSpinBox()
        self._spin_box_h = QDoubleSpinBox()
        self._spin_box_d = QDoubleSpinBox()
        self._slider_box_w = QSlider(Qt.Horizontal)
        self._slider_box_h = QSlider(Qt.Horizontal)
        self._slider_box_d = QSlider(Qt.Horizontal)
        for label, spin, slider in [
            ("W", self._spin_box_w, self._slider_box_w),
            ("H", self._spin_box_h, self._slider_box_h),
            ("D", self._spin_box_d, self._slider_box_d),
        ]:
            spin.setRange(0.01, 1e6)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            slider.setRange(1, 10000)  # default; updated per mesh
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            col.addWidget(spin)
            col.addWidget(slider)
            size_row.addLayout(col)
        bc_layout.addWidget(QLabel("Size:"))
        bc_layout.addLayout(size_row)

        # Reset button
        btn_reset_box = QPushButton("Reset to Mesh Bounds")
        btn_reset_box.clicked.connect(self._on_reset_box)
        bc_layout.addWidget(btn_reset_box)

        self._box_controls_box.setLayout(bc_layout)
        panel.addWidget(self._box_controls_box)
        self._box_controls_box.setVisible(False)

        # Connect box spinboxes → _on_box_control_change
        for spin in (self._spin_box_cx, self._spin_box_cy, self._spin_box_cz,
                     self._spin_box_rx, self._spin_box_ry, self._spin_box_rz,
                     self._spin_box_w, self._spin_box_h, self._spin_box_d):
            spin.valueChanged.connect(self._on_box_control_change)

        # Connect box sliders → _on_box_slider_change
        for slider in (self._slider_box_cx, self._slider_box_cy, self._slider_box_cz,
                       self._slider_box_rx, self._slider_box_ry, self._slider_box_rz,
                       self._slider_box_w, self._slider_box_h, self._slider_box_d):
            slider.valueChanged.connect(self._on_box_slider_change)

        # Connect origin spinboxes (after guard flag is initialized)
        self._spin_x.valueChanged.connect(self._on_manual_origin_change)
        self._spin_y.valueChanged.connect(self._on_manual_origin_change)
        self._spin_z.valueChanged.connect(self._on_manual_origin_change)

        # Connect normal sliders
        self._slider_elev.valueChanged.connect(self._on_normal_slider_change)
        self._slider_azim.valueChanged.connect(self._on_normal_slider_change)

        panel.addWidget(self._separator("Trim"))

        self._btn_trim = QPushButton("✂ Trim region")
        self._btn_trim.setCheckable(True)
        self._btn_trim.toggled.connect(self._toggle_trim_mode)
        panel.addWidget(self._btn_trim)

        panel.addWidget(self._separator("Patches"))

        self.patch_list = QListWidget()
        panel.addWidget(self.patch_list)

        btn_row2 = QHBoxLayout()
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.clicked.connect(self._on_rename)
        btn_row2.addWidget(self.btn_rename)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        btn_row2.addWidget(self.btn_delete)
        panel.addLayout(btn_row2)

        # ── Geometry Info panel ──
        geo_box = QGroupBox("Geometry Info")
        geo_lay = QVBoxLayout()
        self._lbl_bounds_x = QLabel("X: —")
        self._lbl_bounds_y = QLabel("Y: —")
        self._lbl_bounds_z = QLabel("Z: —")
        geo_lay.addWidget(self._lbl_bounds_x)
        geo_lay.addWidget(self._lbl_bounds_y)
        geo_lay.addWidget(self._lbl_bounds_z)
        self._lbl_wall_faces = QLabel("Wall: — faces")
        geo_lay.addWidget(self._lbl_wall_faces)
        self._lbl_open_profiles = QLabel("Open profiles: —")
        geo_lay.addWidget(self._lbl_open_profiles)
        self._lbl_open_edges = QLabel("Open edges: —")
        geo_lay.addWidget(self._lbl_open_edges)
        self._btn_show_mesh_edges = QPushButton("Show Surface Mesh")
        self._btn_show_mesh_edges.setCheckable(True)
        self._btn_show_mesh_edges.setChecked(False)
        self._btn_show_mesh_edges.clicked.connect(self._on_toggle_mesh_edges)
        geo_lay.addWidget(self._btn_show_mesh_edges)
        self._btn_opaque_wall = QPushButton("Opaque Wall")
        self._btn_opaque_wall.setCheckable(True)
        self._btn_opaque_wall.setChecked(True)
        self._btn_opaque_wall.clicked.connect(self._on_toggle_opaque_wall)
        geo_lay.addWidget(self._btn_opaque_wall)
        self._btn_show_boundary = QPushButton("Show Boundary Edges")
        self._btn_show_boundary.setCheckable(True)
        self._btn_show_boundary.setChecked(False)
        self._btn_show_boundary.clicked.connect(self._on_toggle_boundary_edges)
        geo_lay.addWidget(self._btn_show_boundary)
        self._lbl_non_manifold = QLabel("Non-manifold edges: —")
        geo_lay.addWidget(self._lbl_non_manifold)
        self._lbl_manifold = QLabel("Manifold: —")
        geo_lay.addWidget(self._lbl_manifold)
        self._btn_show_non_manifold = QPushButton("Show Non-Manifold")
        self._btn_show_non_manifold.setCheckable(True)
        self._btn_show_non_manifold.setChecked(False)
        self._btn_show_non_manifold.clicked.connect(self._on_toggle_non_manifold)
        geo_lay.addWidget(self._btn_show_non_manifold)
        geo_box.setLayout(geo_lay)
        panel.addWidget(geo_box)

        panel.addWidget(self._separator("STL Export"))

        # Scale factor input (for STL-only exports on this tab).
        # Applied uniformly to all vertex coordinates on export.
        # Default 1.0 = save raw STL coordinates unchanged; set this yourself
        # (e.g. 0.001 for a mm STL -> metres) when the target needs scaling.
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale factor ×"))
        self._spin_scale = QDoubleSpinBox()
        self._spin_scale.setRange(1e-6, 1e6)
        self._spin_scale.setDecimals(6)
        self._spin_scale.setSingleStep(0.1)
        self._spin_scale.setValue(1.0)
        scale_row.addWidget(self._spin_scale)
        panel.addLayout(scale_row)

        self.btn_export_sep = QPushButton("Export Separate STLs")
        self.btn_export_sep.clicked.connect(self._on_export_separate)
        panel.addWidget(self.btn_export_sep)

        self.btn_export_comb = QPushButton("Export Combined STL")
        self.btn_export_comb.clicked.connect(self._on_export_combined)
        panel.addWidget(self.btn_export_comb)

        # Standalone multi-solid STL save — same file format OpenFOAM consumes
        # (one ``solid <name>`` block per cap + one ``solid wall``), but picks
        # any output path without creating a case directory or writing JSON.
        self.btn_save_of_stl = QPushButton("Save OpenFOAM STL...")
        self.btn_save_of_stl.clicked.connect(self._on_save_openfoam_stl)
        panel.addWidget(self.btn_save_of_stl)

        panel.addWidget(self._separator("Centerline"))

        self.btn_compute_cl = QPushButton("Compute Centerline")
        self.btn_compute_cl.clicked.connect(self._on_compute_centerline)
        panel.addWidget(self.btn_compute_cl)

        self.btn_clear_cl = QPushButton("Clear Centerline")
        self.btn_clear_cl.clicked.connect(self._on_clear_centerline)
        panel.addWidget(self.btn_clear_cl)

        self.btn_save_cl = QPushButton("Save Centerline")
        self.btn_save_cl.clicked.connect(self._on_save_centerline)
        panel.addWidget(self.btn_save_cl)

        panel.addWidget(self._separator("View"))

        # Orthogonal view buttons — 3 rows of axis pairs
        for axis, pos_cb, neg_cb in [
            ("X", self._on_view_pos_x, self._on_view_neg_x),
            ("Y", self._on_view_pos_y, self._on_view_neg_y),
            ("Z", self._on_view_pos_z, self._on_view_neg_z),
        ]:
            row = QHBoxLayout()
            bp = QPushButton(f"+{axis}")
            bp.clicked.connect(pos_cb)
            row.addWidget(bp)
            bn = QPushButton(f"-{axis}")
            bn.clicked.connect(neg_cb)
            row.addWidget(bn)
            panel.addLayout(row)

        self.btn_zoom_fit = QPushButton("Zoom to Fit")
        self.btn_zoom_fit.clicked.connect(self._on_zoom_to_fit)
        panel.addWidget(self.btn_zoom_fit)

        panel.addStretch()

        # ── Tab 2: Clip & Save STL (non-CFD use) ──
        self._build_clip_save_tab()

        # ── Tab 3: Export Settings ──
        self._build_export_settings_tab()

        # ── Tab 4: Mesh Repair ──
        self._build_repair_tab()

        # ── Tab 5: Run OpenFOAM (Beta) ──
        self._build_run_tab()

        # Draggable splitter between viewport and tab panel
        splitter = QSplitter(Qt.Horizontal, central)
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(self._tab_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        layout.addWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load an STL file to begin.")

        self._patch_dpr_picking()

    def _build_export_settings_tab(self):
        """Build Tab 2 with all tunable export parameters from the template."""
        cfg = openfoam_case.load_template()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab2_widget = QWidget()
        tab2 = QVBoxLayout(tab2_widget)
        scroll.setWidget(tab2_widget)
        self._tab_widget.addTab(scroll, "Export Settings")

        # --- Simulation Type ---
        sim_box = QGroupBox("Simulation Type")
        sim_lay = QVBoxLayout()
        self._combo_sim_type = QComboBox()
        self._combo_sim_type.addItem("LES (Smagorinsky)", "les")
        self._combo_sim_type.addItem("RANS (k-omega SST)", "rans")
        self._combo_sim_type.currentIndexChanged.connect(self._on_sim_type_changed)
        sim_lay.addWidget(self._combo_sim_type)
        sim_box.setLayout(sim_lay)
        tab2.addWidget(sim_box)

        # --- Scale ---
        # Uniform scale factor applied to all STL vertex coordinates on export.
        scale_box = QGroupBox("Scale")
        scale_lay = QHBoxLayout()
        scale_lay.addWidget(QLabel("Scale factor ×"))
        self._spin_scale_of = QDoubleSpinBox()
        self._spin_scale_of.setRange(1e-6, 1e6)
        self._spin_scale_of.setDecimals(6)
        self._spin_scale_of.setSingleStep(0.1)
        self._spin_scale_of.setValue(1.0)
        scale_lay.addWidget(self._spin_scale_of)
        scale_box.setLayout(scale_lay)
        tab2.addWidget(scale_box)

        # --- Mesh ---
        mesh_box = QGroupBox("Mesh")
        mesh_lay = QVBoxLayout()
        mesh_cfg = cfg["mesh"]

        self._export_max_cell_size = QDoubleSpinBox()
        self._export_max_cell_size.setRange(1e-6, 10.0)
        self._export_max_cell_size.setDecimals(6)
        self._export_max_cell_size.setSingleStep(0.0001)
        self._export_max_cell_size.setValue(mesh_cfg["maxCellSize"])

        self._export_boundary_cell_size = QDoubleSpinBox()
        self._export_boundary_cell_size.setRange(1e-6, 10.0)
        self._export_boundary_cell_size.setDecimals(6)
        self._export_boundary_cell_size.setSingleStep(0.0001)
        self._export_boundary_cell_size.setValue(mesh_cfg["boundaryCellSize"])

        self._export_wall_cell_size = QDoubleSpinBox()
        self._export_wall_cell_size.setRange(1e-6, 10.0)
        self._export_wall_cell_size.setDecimals(6)
        self._export_wall_cell_size.setSingleStep(0.0001)
        self._export_wall_cell_size.setValue(mesh_cfg["wallCellSize"])

        self._export_n_layers = QSpinBox()
        self._export_n_layers.setRange(0, 20)
        self._export_n_layers.setValue(mesh_cfg["nLayers"])

        self._export_thickness_ratio = QDoubleSpinBox()
        self._export_thickness_ratio.setRange(0.01, 5.0)
        self._export_thickness_ratio.setDecimals(2)
        self._export_thickness_ratio.setSingleStep(0.1)
        self._export_thickness_ratio.setValue(mesh_cfg["thicknessRatio"])

        for label, widget in [
            ("Max cell size:", self._export_max_cell_size),
            ("Boundary cell size:", self._export_boundary_cell_size),
            ("Wall cell size:", self._export_wall_cell_size),
            ("Boundary layers:", self._export_n_layers),
            ("Thickness ratio:", self._export_thickness_ratio),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            mesh_lay.addLayout(row)
        mesh_box.setLayout(mesh_lay)
        tab2.addWidget(mesh_box)

        # --- Solver ---
        solver_box = QGroupBox("Solver")
        solver_lay = QVBoxLayout()
        solver_cfg = cfg["solver"]

        self._export_end_time = QDoubleSpinBox()
        self._export_end_time.setRange(0.001, 1000.0)
        self._export_end_time.setDecimals(3)
        self._export_end_time.setSingleStep(0.1)
        self._export_end_time.setValue(solver_cfg["endTime"])

        self._export_delta_t = QDoubleSpinBox()
        self._export_delta_t.setRange(1e-8, 1.0)
        self._export_delta_t.setDecimals(8)
        self._export_delta_t.setSingleStep(1e-5)
        self._export_delta_t.setValue(solver_cfg["deltaT"])

        self._export_write_interval = QDoubleSpinBox()
        self._export_write_interval.setRange(1e-6, 100.0)
        self._export_write_interval.setDecimals(4)
        self._export_write_interval.setSingleStep(0.01)
        self._export_write_interval.setValue(solver_cfg["writeInterval"])

        self._export_max_co = QDoubleSpinBox()
        self._export_max_co.setRange(0.01, 10.0)
        self._export_max_co.setDecimals(2)
        self._export_max_co.setSingleStep(0.1)
        self._export_max_co.setValue(solver_cfg["maxCo"])

        self._export_max_delta_t = QDoubleSpinBox()
        self._export_max_delta_t.setRange(1e-8, 1.0)
        self._export_max_delta_t.setDecimals(6)
        self._export_max_delta_t.setSingleStep(1e-3)
        self._export_max_delta_t.setValue(solver_cfg["maxDeltaT"])

        for label, widget in [
            ("End time (s):", self._export_end_time),
            ("Delta T (s):", self._export_delta_t),
            ("Write interval:", self._export_write_interval),
            ("Max Courant:", self._export_max_co),
            ("Max Delta T:", self._export_max_delta_t),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            solver_lay.addLayout(row)
        solver_box.setLayout(solver_lay)
        tab2.addWidget(solver_box)

        # --- Fluid ---
        fluid_box = QGroupBox("Fluid")
        fluid_lay = QVBoxLayout()

        self._export_nu = QDoubleSpinBox()
        self._export_nu.setRange(1e-9, 1.0)
        self._export_nu.setDecimals(8)
        self._export_nu.setSingleStep(1e-7)
        self._export_nu.setValue(cfg["fluid"]["nu"])

        row = QHBoxLayout()
        row.addWidget(QLabel("Viscosity nu:"))
        row.addWidget(self._export_nu)
        fluid_lay.addLayout(row)
        fluid_box.setLayout(fluid_lay)
        tab2.addWidget(fluid_box)

        # --- Turbulence ---
        turb_box = QGroupBox("Turbulence")
        turb_lay = QVBoxLayout()

        # LES: Smagorinsky Cs
        self._lbl_cs = QLabel("Cs:")
        self._export_cs = QDoubleSpinBox()
        self._export_cs.setRange(0.01, 1.0)
        self._export_cs.setDecimals(3)
        self._export_cs.setSingleStep(0.01)
        self._export_cs.setValue(cfg["turbulence"]["Cs"])

        row = QHBoxLayout()
        row.addWidget(self._lbl_cs)
        row.addWidget(self._export_cs)
        turb_lay.addLayout(row)

        # RANS: k and omega (hidden by default)
        self._lbl_k = QLabel("k:")
        self._export_k = QDoubleSpinBox()
        self._export_k.setRange(1e-8, 1.0)
        self._export_k.setDecimals(6)
        self._export_k.setSingleStep(1e-4)
        self._export_k.setValue(3.375e-4)

        row = QHBoxLayout()
        row.addWidget(self._lbl_k)
        row.addWidget(self._export_k)
        turb_lay.addLayout(row)

        self._lbl_omega = QLabel("omega:")
        self._export_omega = QDoubleSpinBox()
        self._export_omega.setRange(0.01, 10000.0)
        self._export_omega.setDecimals(2)
        self._export_omega.setSingleStep(1.0)
        self._export_omega.setValue(30.0)

        row = QHBoxLayout()
        row.addWidget(self._lbl_omega)
        row.addWidget(self._export_omega)
        turb_lay.addLayout(row)

        turb_box.setLayout(turb_lay)
        tab2.addWidget(turb_box)

        # Hide RANS fields by default (LES is initial selection)
        self._lbl_k.hide()
        self._export_k.hide()
        self._lbl_omega.hide()
        self._export_omega.hide()

        # --- Inlet ---
        inlet_box = QGroupBox("Inlet")
        inlet_lay = QVBoxLayout()

        self._export_velocity_mag = QDoubleSpinBox()
        self._export_velocity_mag.setRange(0.001, 100.0)
        self._export_velocity_mag.setDecimals(4)
        self._export_velocity_mag.setSingleStep(0.05)
        self._export_velocity_mag.setValue(cfg["inlet"]["velocityMagnitude"])

        row = QHBoxLayout()
        row.addWidget(QLabel("Velocity mag (m/s):"))
        row.addWidget(self._export_velocity_mag)
        inlet_lay.addLayout(row)
        inlet_box.setLayout(inlet_lay)
        tab2.addWidget(inlet_box)

        # --- Parallel ---
        par_box = QGroupBox("Parallel")
        par_lay = QVBoxLayout()

        self._export_n_procs = QSpinBox()
        self._export_n_procs.setRange(1, 1024)
        self._export_n_procs.setValue(cfg["decompose"]["nProcs"])

        row = QHBoxLayout()
        row.addWidget(QLabel("Processors:"))
        row.addWidget(self._export_n_procs)
        par_lay.addLayout(row)
        par_box.setLayout(par_lay)
        tab2.addWidget(par_box)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset Defaults")
        btn_reset.clicked.connect(self._on_reset_export_defaults)
        btn_row.addWidget(btn_reset)

        self.btn_export_foam = QPushButton("Export for OpenFOAM")
        self.btn_export_foam.clicked.connect(self._on_export_openfoam)
        btn_row.addWidget(self.btn_export_foam)
        tab2.addLayout(btn_row)

        tab2.addStretch()

    def _build_clip_save_tab(self):
        """Build the 'Clip & Save STL' tab.

        Reuses the existing Clipping tab's pickers — this tab only displays
        the current clips with a per-clip Open/Closed selector and a Save
        button.  The output is a generic multi-solid ASCII STL with no
        OpenFOAM case scaffolding.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        scroll.setWidget(tab_widget)
        self._tab_widget.addTab(scroll, "Clip & Save STL")

        intro = QLabel(
            "Save the clipped surface as a single-solid STL for downstream\n"
            "tools (e.g. VMTK).  For each clip choose:\n"
            "  • Open (no cap) — leaves a hole at the cut.\n"
            "  • Closed (cap) — merges a cap into the surface (watertight there)."
        )
        intro.setWordWrap(True)
        tab_layout.addWidget(intro)

        self._clip_save_list_box = QGroupBox("Clips")
        self._clip_save_list_layout = QVBoxLayout()
        self._clip_save_list_layout.setSpacing(4)
        self._clip_save_list_box.setLayout(self._clip_save_list_layout)
        tab_layout.addWidget(self._clip_save_list_box)

        # Per-row combo boxes are tracked so we can detach signals on refresh.
        self._clip_save_combos: list = []

        # Independent scale factor for this tab — default 1.0 preserves the
        # original STL units (e.g. keeps mm input as mm on output). All scale
        # spinboxes now default to 1.0 (raw); set a factor explicitly to convert
        # (e.g. 0.001 for a mm STL -> metres for CFD).
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale factor ×"))
        self._spin_scale_save = QDoubleSpinBox()
        self._spin_scale_save.setRange(1e-6, 1e6)
        self._spin_scale_save.setDecimals(6)
        self._spin_scale_save.setSingleStep(0.1)
        self._spin_scale_save.setValue(1.0)
        self._spin_scale_save.setToolTip(
            "Multiplier applied to all vertex coordinates on save.\n"
            "1.0 = preserve original STL units (recommended for VMTK etc.).\n"
            "0.001 = mm → m (CFD/OpenFOAM convention)."
        )
        scale_row.addWidget(self._spin_scale_save)
        tab_layout.addLayout(scale_row)

        self.btn_save_clipped_stl = QPushButton("Save Clipped STL...")
        self.btn_save_clipped_stl.clicked.connect(self._on_save_clipped_stl)
        tab_layout.addWidget(self.btn_save_clipped_stl)

        self._clip_save_status = QLabel("Status: —")
        self._clip_save_status.setWordWrap(True)
        tab_layout.addWidget(self._clip_save_status)

        tab_layout.addStretch()

        # Initial population (engine has no clips yet, so just shows placeholder).
        self._refresh_clip_save_list()

    def _refresh_clip_save_list(self):
        """Rebuild the per-clip rows in the 'Clip & Save STL' tab.

        Safe to call before the tab has been built (early in _build_ui) — it
        no-ops when the layout isn't ready yet.
        """
        if not hasattr(self, "_clip_save_list_layout"):
            return

        # Detach signals before deleting widgets to avoid spurious callbacks.
        for combo in self._clip_save_combos:
            try:
                combo.currentIndexChanged.disconnect()
            except TypeError:
                pass
        self._clip_save_combos.clear()

        while self._clip_save_list_layout.count():
            item = self._clip_save_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self.engine.clips:
            placeholder = QLabel("(no clips — add some on the Clipping tab)")
            placeholder.setStyleSheet("color: #888;")
            self._clip_save_list_layout.addWidget(placeholder)
            self.btn_save_clipped_stl.setEnabled(False)
            return

        self.btn_save_clipped_stl.setEnabled(True)

        for idx, clip_def in enumerate(self.engine.clips):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)

            r, g, b = [int(c * 255) for c in clip_def.color]
            swatch = QLabel("  ")
            swatch.setFixedWidth(14)
            swatch.setFixedHeight(14)
            swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"
            )
            row.addWidget(swatch)

            n = clip_def.cap_mesh.n_cells if clip_def.cap_mesh else 0
            face_info = f"{n:,} cap faces" if clip_def.cap_kind == "closed" else "no cap"
            lbl = QLabel(f"{clip_def.name}  ({face_info})")
            lbl.setMinimumWidth(180)
            row.addWidget(lbl, stretch=1)

            combo = QComboBox()
            combo.addItem("Closed (cap)", "closed")
            combo.addItem("Open (no cap)", "open")
            combo.setCurrentIndex(0 if clip_def.cap_kind == "closed" else 1)
            combo.currentIndexChanged.connect(
                lambda i, _idx=idx: self._on_cap_kind_changed(_idx, i)
            )
            self._clip_save_combos.append(combo)
            row.addWidget(combo)

            self._clip_save_list_layout.addWidget(row_widget)

    def _on_cap_kind_changed(self, idx: int, combo_index: int):
        """Slot for per-clip Open/Closed combo box."""
        kind = "closed" if combo_index == 0 else "open"
        if idx < 0 or idx >= len(self.engine.clips):
            return
        if self.engine.clips[idx].cap_kind == kind:
            return
        self.engine.set_cap_kind(idx, kind)
        # _refresh_patch_list calls _refresh_clip_save_list internally.
        self._refresh_patch_list()
        self._refresh_display()

    def _on_save_clipped_stl(self):
        """File-dialog save of a single-solid STL with current open/closed mix."""
        if not self.engine.clips:
            QMessageBox.warning(self, "No Clips", "Add at least one clip first.")
            return
        default_name = "clipped.stl"
        if self._loaded_filepath:
            stem = os.path.splitext(os.path.basename(self._loaded_filepath))[0]
            default_name = f"{stem}_clipped.stl"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Clipped STL", default_name, "STL files (*.stl)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".stl"):
            filepath += ".stl"
        sf = self._spin_scale_save.value()
        try:
            result = self.engine.export_clipped_surface_stl(filepath, scale_factor=sf)
            opened = [c.name for c in self.engine.clips if c.cap_kind == "open"]
            closed = [c.name for c in self.engine.clips if c.cap_kind == "closed"]
            self._clip_save_status.setText(
                f"Saved: {os.path.basename(filepath)}  "
                f"({result['n_total_cells']:,} faces, "
                f"{result['n_caps_merged']} caps merged, "
                f"{result['n_open_clips']} open)"
            )
            QMessageBox.information(
                self, "Saved",
                f"Single-solid STL saved to:\n{filepath}\n"
                f"Scale factor: ×{sf}\n"
                f"Total faces: {result['n_total_cells']:,}\n\n"
                f"Open (hole) clips: {', '.join(opened) if opened else 'none'}\n"
                f"Closed (merged-cap) clips: {', '.join(closed) if closed else 'none'}",
            )
        except Exception as e:
            self._clip_save_status.setText(f"Error: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    def _build_repair_tab(self):
        """Build Tab 3 with mesh repair operations."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab3_widget = QWidget()
        tab3 = QVBoxLayout(tab3_widget)
        scroll.setWidget(tab3_widget)
        self._tab_widget.addTab(scroll, "Mesh Repair")

        tab3.addWidget(QLabel("Repair modifies original mesh\nand clears all clips."))

        self._btn_repair_clean = QPushButton("Clean Mesh")
        self._btn_repair_clean.setToolTip("Remove duplicate points and degenerate triangles")
        self._btn_repair_clean.clicked.connect(self._on_repair_clean)
        tab3.addWidget(self._btn_repair_clean)

        self._btn_repair_normals = QPushButton("Fix Normals")
        self._btn_repair_normals.setToolTip("Consistent winding order for face normals")
        self._btn_repair_normals.clicked.connect(self._on_repair_normals)
        tab3.addWidget(self._btn_repair_normals)

        self._lbl_repair_status = QLabel("Status: \u2014")
        tab3.addWidget(self._lbl_repair_status)

        tab3.addStretch()

    def _build_run_tab(self):
        """Build Tab 4: Run OpenFOAM (Beta) — Docker solver launcher."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab4_widget = QWidget()
        tab4 = QVBoxLayout(tab4_widget)
        scroll.setWidget(tab4_widget)
        self._tab_widget.addTab(scroll, "Run OpenFOAM (Beta)")

        tab4.addWidget(QLabel("Run Docker pipeline from the app."))

        # Read-only nProcs label (synced from Export Settings)
        self._run_nprocs_label = QLabel("Processors: —")
        tab4.addWidget(self._run_nprocs_label)
        self._export_n_procs.valueChanged.connect(
            lambda v: self._run_nprocs_label.setText(f"Processors: {v}")
        )
        # Set initial value
        self._run_nprocs_label.setText(
            f"Processors: {self._export_n_procs.value()}"
        )

        # Case directory label
        self._run_case_label = QLabel("Case: (none exported)")
        self._run_case_label.setWordWrap(True)
        tab4.addWidget(self._run_case_label)

        # Run / Cancel buttons
        btn_row = QHBoxLayout()
        self._btn_run_solver = QPushButton("Run Solver")
        self._btn_run_solver.clicked.connect(self._on_run_solver)
        btn_row.addWidget(self._btn_run_solver)

        self._btn_cancel_solver = QPushButton("Cancel")
        self._btn_cancel_solver.setEnabled(False)
        self._btn_cancel_solver.clicked.connect(self._on_cancel_solver)
        btn_row.addWidget(self._btn_cancel_solver)
        tab4.addLayout(btn_row)

        # Stage indicator
        self._run_stage_label = QLabel("Stage: \u2014")
        tab4.addWidget(self._run_stage_label)

        # Log viewer
        self._run_log = QPlainTextEdit()
        self._run_log.setReadOnly(True)
        self._run_log.setMaximumBlockCount(50000)
        self._run_log.setFont(QFont("Courier", 10))
        self._run_log.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
        )
        tab4.addWidget(self._run_log)

        # Bottom buttons
        bottom_row = QHBoxLayout()
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.clicked.connect(self._run_log.clear)
        bottom_row.addWidget(btn_clear_log)

        btn_open_folder = QPushButton("Open Case Folder")
        btn_open_folder.clicked.connect(self._on_open_case_folder)
        bottom_row.addWidget(btn_open_folder)
        tab4.addLayout(bottom_row)

    def _on_run_solver(self):
        """Pre-flight checks then launch OpenFOAMWorker."""
        # Check case directory
        if not self._last_case_dir or not os.path.isdir(self._last_case_dir):
            QMessageBox.warning(
                self, "No Case",
                "Export a case first (Export Settings tab).",
            )
            return

        script = os.path.join(self._last_case_dir, "run_docker.sh")
        if not os.path.isfile(script):
            QMessageBox.warning(
                self, "Missing Script",
                f"run_docker.sh not found in:\n{self._last_case_dir}",
            )
            return

        # Quick Docker check
        try:
            subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            QMessageBox.critical(
                self, "Docker Not Available",
                "Cannot reach Docker. Is Docker Desktop running?",
            )
            return

        # Check for existing time directories (results)
        time_dirs = [
            d for d in os.listdir(self._last_case_dir)
            if os.path.isdir(os.path.join(self._last_case_dir, d))
            and d not in ("0", "constant", "system", "__pycache__")
            and not d.startswith(".")
        ]
        # Filter to numeric directory names (OpenFOAM time dirs)
        time_dirs = [d for d in time_dirs if d.replace(".", "", 1).isdigit()
                     and d != "0"]
        if time_dirs:
            reply = QMessageBox.question(
                self, "Existing Results",
                f"Found {len(time_dirs)} time directories. "
                "Running again will overwrite results.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        nprocs = self._export_n_procs.value()
        self._openfoam_worker = OpenFOAMWorker(self._last_case_dir, nprocs)
        self._openfoam_worker.log_line.connect(self._on_solver_log_line)
        self._openfoam_worker.stage_changed.connect(self._on_solver_stage_changed)
        self._openfoam_worker.finished_ok.connect(self._on_solver_finished)
        self._openfoam_worker.failed.connect(self._on_solver_failed)

        self._btn_run_solver.setEnabled(False)
        self._btn_cancel_solver.setEnabled(True)
        self._run_stage_label.setText("Stage: starting...")
        self._run_log.clear()

        self._openfoam_worker.start()

    def _on_cancel_solver(self):
        """Stop Docker containers and terminate the worker."""
        if self._openfoam_worker and self._openfoam_worker.isRunning():
            self._run_stage_label.setText("Stage: cancelling...")
            self._openfoam_worker.cancel()

    def _on_solver_log_line(self, text: str):
        """Append batched log text to the viewer."""
        self._run_log.appendPlainText(text)
        # Auto-scroll to bottom
        sb = self._run_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_solver_stage_changed(self, stage: str):
        """Update the stage label."""
        self._run_stage_label.setText(f"Stage: {stage}")

    def _on_solver_finished(self):
        """Solver completed successfully."""
        self._btn_run_solver.setEnabled(True)
        self._btn_cancel_solver.setEnabled(False)
        self._run_stage_label.setText("Stage: Done")
        self._openfoam_worker = None
        self.status.showMessage("OpenFOAM run completed.")

    def _on_solver_failed(self, msg: str):
        """Solver failed or was cancelled."""
        self._btn_run_solver.setEnabled(True)
        self._btn_cancel_solver.setEnabled(False)
        self._run_stage_label.setText("Stage: FAILED")
        self._openfoam_worker = None
        if msg != "Cancelled by user":
            QMessageBox.critical(self, "Solver Error", msg)
        self.status.showMessage(f"OpenFOAM: {msg}")

    def _on_open_case_folder(self):
        """Open the case directory in the system file manager."""
        if self._last_case_dir and os.path.isdir(self._last_case_dir):
            import platform
            if platform.system() == "Darwin":
                subprocess.Popen(["open", self._last_case_dir])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", self._last_case_dir])
            else:
                subprocess.Popen(["explorer", self._last_case_dir])
        else:
            QMessageBox.information(
                self, "No Case", "No case directory to open.",
            )

    def closeEvent(self, event):
        """Prompt if solver is running before closing."""
        if self._openfoam_worker and self._openfoam_worker.isRunning():
            reply = QMessageBox.question(
                self, "Solver Running",
                "OpenFOAM solver is still running. Stop and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._on_cancel_solver()
                self._openfoam_worker.wait(10000)
            else:
                event.ignore()
                return
        super().closeEvent(event)

    def _on_repair_clean(self):
        msg = self.engine.repair_clean()
        self._lbl_repair_status.setText(msg)
        self._centerline_mesh = None
        self._refresh_patch_list()
        self._refresh_display()
        self._update_button_states()

    def _on_repair_normals(self):
        msg = self.engine.repair_normals()
        self._lbl_repair_status.setText(msg)
        self._centerline_mesh = None
        self._refresh_patch_list()
        self._refresh_display()
        self._update_button_states()

    def _on_sim_type_changed(self, index: int):
        """Switch UI between LES and RANS turbulence parameters."""
        sim_type = self._combo_sim_type.currentData()
        is_rans = sim_type == "rans"

        # Toggle LES vs RANS turbulence widgets
        self._lbl_cs.setVisible(not is_rans)
        self._export_cs.setVisible(not is_rans)
        self._lbl_k.setVisible(is_rans)
        self._export_k.setVisible(is_rans)
        self._lbl_omega.setVisible(is_rans)
        self._export_omega.setVisible(is_rans)

        # Load defaults from the appropriate template
        tmpl_name = "rans_aorta" if is_rans else "les_aorta"
        cfg = openfoam_case.load_template(tmpl_name)
        solver = cfg["solver"]
        self._export_delta_t.setValue(solver["deltaT"])
        self._export_max_co.setValue(solver["maxCo"])
        self._export_max_delta_t.setValue(solver["maxDeltaT"])
        self._export_end_time.setValue(solver["endTime"])

        if is_rans:
            turb = cfg["turbulence"]
            self._export_k.setValue(turb.get("k", 3.375e-4))
            self._export_omega.setValue(turb.get("omega", 30.0))
        else:
            self._export_cs.setValue(cfg["turbulence"].get("Cs", 0.1))

        self.status.showMessage(f"Switched to {cfg['name']} template.")

    def _on_reset_export_defaults(self):
        """Reset all export settings spinboxes to template defaults."""
        sim_type = self._combo_sim_type.currentData()
        tmpl_name = "rans_aorta" if sim_type == "rans" else "les_aorta"
        cfg = openfoam_case.load_template(tmpl_name)
        mesh = cfg["mesh"]
        solver = cfg["solver"]

        self._export_max_cell_size.setValue(mesh["maxCellSize"])
        self._export_boundary_cell_size.setValue(mesh["boundaryCellSize"])
        self._export_wall_cell_size.setValue(mesh["wallCellSize"])
        self._export_n_layers.setValue(mesh["nLayers"])
        self._export_thickness_ratio.setValue(mesh["thicknessRatio"])

        self._export_end_time.setValue(solver["endTime"])
        self._export_delta_t.setValue(solver["deltaT"])
        self._export_write_interval.setValue(solver["writeInterval"])
        self._export_max_co.setValue(solver["maxCo"])
        self._export_max_delta_t.setValue(solver["maxDeltaT"])

        self._export_nu.setValue(cfg["fluid"]["nu"])
        self._export_velocity_mag.setValue(cfg["inlet"]["velocityMagnitude"])
        self._export_n_procs.setValue(cfg["decompose"]["nProcs"])

        if sim_type == "rans":
            turb = cfg["turbulence"]
            self._export_k.setValue(turb.get("k", 3.375e-4))
            self._export_omega.setValue(turb.get("omega", 30.0))
        else:
            self._export_cs.setValue(cfg["turbulence"].get("Cs", 0.1))

        self.status.showMessage(f"Export settings reset to {cfg['name']} defaults.")

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("Open STL...", self._on_load)
        file_menu.addSeparator()
        file_menu.addAction("Export for OpenFOAM", self._on_export_openfoam)
        file_menu.addAction("Export Separate STLs...", self._on_export_separate)
        file_menu.addAction("Export Combined STL...", self._on_export_combined)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

    def _patch_dpr_picking(self):
        """Fix macOS Retina DPR mismatch for VTK widget picking.

        pyvistaqt scales mouse coords by device-pixel-ratio before passing
        them to VTK, but vtkCocoaRenderWindow reports size in logical pixels.
        This makes widget pickers (plane, box) receive physical-pixel coords
        against a logical-pixel viewport — the pick ray misses.  Patch the
        interactor to keep everything in logical-pixel space.
        """
        interactor = self.plotter.interactor

        def _patched_setEventInformation(
            self, x, y, ctrl, shift, key, repeat=0, keysum=None
        ):
            self._Iren.SetEventInformation(
                int(round(x)),
                int(round(self.height() - y - 1)),
                ctrl, shift, key, repeat, keysum,
            )

        def _patched_resizeEvent(self, ev):
            w = self.width()
            h = self.height()
            if self._RenderWindow is None:
                return
            self._RenderWindow.SetDPI(72)
            vtk.vtkRenderWindow.SetSize(self._RenderWindow, w, h)
            self._Iren.SetSize(w, h)
            self._Iren.ConfigureEvent()
            self.update()

        interactor._setEventInformation = types.MethodType(
            _patched_setEventInformation, interactor
        )
        interactor.resizeEvent = types.MethodType(
            _patched_resizeEvent, interactor
        )

        # Force a resize so VTK picks up the corrected dimensions now
        from PyQt5.QtGui import QResizeEvent
        from PyQt5.QtCore import QSize
        interactor.resizeEvent(
            QResizeEvent(
                QSize(interactor.width(), interactor.height()),
                QSize(interactor.width(), interactor.height()),
            )
        )

    @staticmethod
    def _separator(label: str) -> QLabel:
        lbl = QLabel(f"— {label} —")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: gray; margin-top: 8px;")
        return lbl

    def _update_button_states(self):
        has_mesh = self.engine.original_mesh is not None
        has_clips = len(self.engine.clips) > 0
        plane_active = self._plane_widget_active

        self.btn_add_plane.setEnabled(has_mesh and not plane_active)
        self.btn_confirm_plane.setEnabled(plane_active and not self._plane_confirmed)
        self.btn_add_constraint.setEnabled(self._plane_confirmed and not self._constraint_box_active)
        self.btn_confirm.setEnabled(self._plane_confirmed)
        self.btn_cancel.setEnabled(plane_active)
        self.btn_flip.setEnabled(plane_active and not self._plane_confirmed)
        self.btn_rename.setEnabled(has_clips)
        self.btn_delete.setEnabled(has_clips)
        self.btn_export_foam.setEnabled(has_clips)
        self.btn_export_sep.setEnabled(has_clips)
        self.btn_export_comb.setEnabled(has_clips)
        self.btn_save_of_stl.setEnabled(has_clips)

        # Centerline buttons
        has_inlet = any("inlet" in c.name.lower() for c in self.engine.clips)
        has_outlet = any("outlet" in c.name.lower() for c in self.engine.clips)
        computing = self._centerline_worker is not None and self._centerline_worker.isRunning()
        self.btn_compute_cl.setEnabled(has_inlet and has_outlet and not computing)
        self.btn_clear_cl.setEnabled(self._centerline_mesh is not None)
        self.btn_save_cl.setEnabled(self._centerline_mesh is not None)

        # Repair buttons
        self._btn_repair_clean.setEnabled(has_mesh)
        self._btn_repair_normals.setEnabled(has_mesh)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _on_load(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open STL", "", "STL Files (*.stl);;All Files (*)"
        )
        if filepath:
            self._load_file(filepath)

    def _load_file(self, filepath: str):
        try:
            mesh = self.engine.load_stl(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._loaded_filepath = filepath
        self._cancel_clip_widgets()
        self._centerline_mesh = None
        self._centerline_worker = None
        self._lbl_repair_status.setText("Status: \u2014")
        self._refresh_display()

        fname = os.path.basename(filepath)
        n = mesh.n_cells
        self.status.showMessage(f"Loaded: {fname} | {n:,} faces | 0 clips defined")
        self._update_button_states()

    # ------------------------------------------------------------------
    # Clip plane & constraint box widgets
    # ------------------------------------------------------------------

    def _on_add_plane(self):
        """Start clip workflow: add interactive plane widget."""
        if self.engine.original_mesh is None:
            return
        mesh = self.engine.get_wall_mesh()
        self._plane_widget_active = True
        self._constraint_box_active = False
        self._plane_confirmed = False
        self._current_box_planes_data = None
        self._current_plane_origin = None
        self._current_plane_normal = None
        # Reset box state from any previous constraint
        self.plotter.clear_box_widgets()
        if self._box_actor is not None:
            self.plotter.remove_actor(self._box_actor, render=False)
            self._box_actor = None
        self._box_center = None
        self._box_rotation_deg = None
        self._box_half_extents = None
        self._box_initial_center = None
        self._box_initial_half_extents = None
        self._box_controls_box.setVisible(False)

        self.plotter.add_plane_widget(
            self._plane_callback,
            normal='z',
            origin=mesh.center,
            color=PREVIEW_COLOR,
        )

        # Initialize manual controls from mesh geometry
        center = np.array(mesh.center, dtype=float)
        self._current_plane_origin = center.copy()
        self._current_plane_normal = np.array([0.0, 0.0, 1.0])
        bounds = np.array(mesh.bounds).reshape(3, 2)
        diag = np.linalg.norm(bounds.ptp(axis=1))
        self._spin_step.setValue(round(max(diag * 0.01, 0.1), 2))
        self._show_plane_controls(True)

        self._update_button_states()

    def _plane_callback(self, normal, origin):
        """Called when user moves/rotates the plane widget."""
        if self._updating_controls:
            return
        self._current_plane_normal = np.asarray(normal, dtype=float)
        self._current_plane_origin = np.asarray(origin, dtype=float)
        self._sync_controls_from_state()
        self._update_preview()

    def _on_confirm_plane(self):
        """Lock the plane position — remove interactive widget, show static disc."""
        if self._current_plane_origin is None:
            return
        self._plane_confirmed = True
        self.plotter.clear_plane_widgets()
        self._add_static_plane_visual()
        self._show_plane_controls(False)
        self._update_button_states()

    def _add_static_plane_visual(self):
        """Render a static disc at the current plane position (non-interactive)."""
        if self._static_plane_actor is not None:
            self.plotter.remove_actor(self._static_plane_actor, render=False)
        wall = self.engine.get_wall_mesh()
        if wall is None:
            return
        bounds = np.array(wall.bounds).reshape(3, 2)
        radius = np.linalg.norm(bounds.ptp(axis=1)) * 0.5
        disc = pv.Disc(center=self._current_plane_origin,
                       normal=self._current_plane_normal,
                       inner=0.0, outer=radius)
        self._static_plane_actor = self.plotter.add_mesh(
            disc, color=PREVIEW_COLOR, opacity=0.15,
            name="_static_plane", render=True, reset_camera=False,
        )

    def _on_add_constraint(self):
        """Add optional constraint box to limit cut to a region of interest.

        Creates both a VTK box widget (mouse drag) and spinbox/slider controls,
        kept in bidirectional sync.
        """
        if not self._plane_widget_active or self._current_plane_origin is None:
            return
        mesh = self.engine.get_wall_mesh()
        self._constraint_box_active = True

        # Compute initial box state — centered at current plane origin
        center = np.array(self._current_plane_origin, dtype=float)
        bounds = np.array(mesh.bounds).reshape(3, 2)
        extents = bounds.ptp(axis=1)  # [dx, dy, dz]
        half_extents = extents * 0.3 / 2.0  # factor=0.3 matching old widget

        self._box_center = center.copy()
        self._box_rotation_deg = np.array([0.0, 0.0, 0.0])
        self._box_half_extents = half_extents.copy()
        self._box_initial_center = center.copy()
        self._box_initial_half_extents = half_extents.copy()

        # Initialize slider ranges from mesh geometry
        self._init_box_slider_ranges(mesh)

        # Set spinbox + slider values (guarded against triggering callbacks)
        self._updating_box_controls = True
        self._spin_box_cx.setValue(float(center[0]))
        self._spin_box_cy.setValue(float(center[1]))
        self._spin_box_cz.setValue(float(center[2]))
        self._spin_box_rx.setValue(0.0)
        self._spin_box_ry.setValue(0.0)
        self._spin_box_rz.setValue(0.0)
        self._spin_box_w.setValue(float(half_extents[0] * 2))
        self._spin_box_h.setValue(float(half_extents[1] * 2))
        self._spin_box_d.setValue(float(half_extents[2] * 2))
        self._updating_box_controls = False
        self._sync_box_sliders_from_spinboxes()

        # Add VTK box widget for mouse interaction
        box_bounds = [
            center[0] - half_extents[0], center[0] + half_extents[0],
            center[1] - half_extents[1], center[1] + half_extents[1],
            center[2] - half_extents[2], center[2] + half_extents[2],
        ]
        self.plotter.add_box_widget(
            self._constraint_box_callback,
            bounds=box_bounds,
            factor=1.0,
            rotation_enabled=True,
            color=(0.2, 0.8, 0.4),
            use_planes=True,
        )

        # Compute initial 6 planes from current state
        self._box_controls_box.setVisible(True)
        self._update_constraint_box()
        self._update_button_states()

    def _constraint_box_callback(self, vtk_planes):
        """Called when user drags the VTK box widget handles.

        Extracts 6 planes from the vtkPlanes object, decomposes them into
        center/rotation/half_extents, and syncs the spinboxes + sliders.
        """
        if self._updating_box_controls:
            return

        # Extract 6 planes from vtkPlanes
        planes_data = []
        for i in range(vtk_planes.GetNumberOfPlanes()):
            plane = vtk_planes.GetPlane(i)
            n = np.array(plane.GetNormal())
            p = np.array(plane.GetOrigin())
            planes_data.append((n.copy(), p.copy()))

        self._current_box_planes_data = planes_data

        # Decompose planes into center, rotation, half_extents
        center, rot_deg, half_extents = self._decompose_box_planes(planes_data)
        if center is None:
            self._update_preview()
            return

        self._box_center = center
        self._box_rotation_deg = rot_deg
        self._box_half_extents = half_extents

        # Update spinboxes + sliders (guarded)
        self._updating_box_controls = True
        self._spin_box_cx.setValue(float(center[0]))
        self._spin_box_cy.setValue(float(center[1]))
        self._spin_box_cz.setValue(float(center[2]))
        self._spin_box_rx.setValue(float(rot_deg[0]))
        self._spin_box_ry.setValue(float(rot_deg[1]))
        self._spin_box_rz.setValue(float(rot_deg[2]))
        self._spin_box_w.setValue(float(half_extents[0] * 2))
        self._spin_box_h.setValue(float(half_extents[1] * 2))
        self._spin_box_d.setValue(float(half_extents[2] * 2))
        self._updating_box_controls = False
        self._sync_box_sliders_from_spinboxes()

        self._update_preview()

    def _on_box_control_change(self):
        """Read all 9 spinboxes and update box state + planes + VTK widget."""
        if self._updating_box_controls:
            return
        self._box_center = np.array([
            self._spin_box_cx.value(),
            self._spin_box_cy.value(),
            self._spin_box_cz.value(),
        ])
        self._box_rotation_deg = np.array([
            self._spin_box_rx.value(),
            self._spin_box_ry.value(),
            self._spin_box_rz.value(),
        ])
        w = self._spin_box_w.value()
        h = self._spin_box_h.value()
        d = self._spin_box_d.value()
        self._box_half_extents = np.array([w / 2.0, h / 2.0, d / 2.0])
        self._sync_box_sliders_from_spinboxes()
        self._update_constraint_box()
        self._push_transform_to_box_widget()

    @staticmethod
    def _euler_rotation_matrix(rx_deg, ry_deg, rz_deg):
        """Build rotation matrix from Euler angles (Rz * Ry * Rx order)."""
        rx = np.radians(rx_deg)
        ry = np.radians(ry_deg)
        rz = np.radians(rz_deg)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _rotation_matrix_to_euler(R):
        """Extract Euler angles (rx, ry, rz) in degrees from R = Rz * Ry * Rx.

        Uses the convention:
            R[2,0] = -sin(ry)
            R[2,1]/cos(ry) = sin(rx), R[2,2]/cos(ry) = cos(rx)
            R[1,0]/cos(ry) = sin(rz), R[0,0]/cos(ry) = cos(rz)
        """
        sy = -R[2, 0]
        sy = np.clip(sy, -1.0, 1.0)
        ry = np.arcsin(sy)
        cy = np.cos(ry)
        if abs(cy) > 1e-6:
            rx = np.arctan2(R[2, 1] / cy, R[2, 2] / cy)
            rz = np.arctan2(R[1, 0] / cy, R[0, 0] / cy)
        else:
            # Gimbal lock: ry ≈ ±90°, set rz=0 and solve rx
            rx = np.arctan2(-R[1, 2], R[1, 1])
            rz = 0.0
        return np.degrees(rx) % 360, np.degrees(ry) % 360, np.degrees(rz) % 360

    @staticmethod
    def _decompose_box_planes(planes_data):
        """Decompose 6 (normal, point) tuples into (center, rotation_deg, half_extents).

        Pairs opposite faces (normals that sum to ~zero), extracts 3 axis
        directions, half-extents, and center, then converts the rotation
        matrix to Euler angles.
        """
        if len(planes_data) != 6:
            return None, None, None

        normals = [np.asarray(n, dtype=float) for n, _ in planes_data]
        points = [np.asarray(p, dtype=float) for _, p in planes_data]

        # Pair opposite faces: find pairs whose normals sum to ~zero
        used = [False] * 6
        pairs = []
        for i in range(6):
            if used[i]:
                continue
            for j in range(i + 1, 6):
                if used[j]:
                    continue
                if np.linalg.norm(normals[i] + normals[j]) < 0.3:
                    pairs.append((i, j))
                    used[i] = used[j] = True
                    break

        if len(pairs) != 3:
            return None, None, None

        axes = []
        half_exts = []
        centers = []
        for i, j in pairs:
            # Axis direction = normal of the "positive" face
            axis = normals[i] / np.linalg.norm(normals[i])
            # Half-extent = half the distance between opposite face points projected onto axis
            diff = points[i] - points[j]
            he = abs(np.dot(axis, diff)) / 2.0
            # Center contribution from this pair
            mid = (points[i] + points[j]) / 2.0
            axes.append(axis)
            half_exts.append(he)
            centers.append(mid)

        center = np.mean(centers, axis=0)

        # Sort axes to canonical ordering: axis[0] closest to global X,
        # axis[1] closest to Y, axis[2] closest to Z — so W/H/D spinboxes
        # match the expected local-axis convention.
        order = sorted(range(3), key=lambda k: np.argmax(np.abs(axes[k])))
        axes = [axes[i] for i in order]
        half_exts = [half_exts[i] for i in order]
        half_extents = np.array(half_exts)

        # Build rotation matrix from 3 axes (ensure right-handed)
        R = np.column_stack(axes)
        if np.linalg.det(R) < 0:
            R[:, 2] = -R[:, 2]

        rx, ry, rz = STLClipperApp._rotation_matrix_to_euler(R)
        return center, np.array([rx, ry, rz]), half_extents

    def _on_box_slider_change(self):
        """When a slider moves, push its value to the corresponding spinbox.

        The spinbox valueChanged signal then triggers _on_box_control_change.
        """
        if self._updating_box_controls:
            return
        self._updating_box_controls = True
        # Center sliders: integer value / 100.0 → float position
        self._spin_box_cx.setValue(self._slider_box_cx.value() / 100.0)
        self._spin_box_cy.setValue(self._slider_box_cy.value() / 100.0)
        self._spin_box_cz.setValue(self._slider_box_cz.value() / 100.0)
        # Rotation sliders: integer / 10.0 → degrees
        self._spin_box_rx.setValue(self._slider_box_rx.value() / 10.0)
        self._spin_box_ry.setValue(self._slider_box_ry.value() / 10.0)
        self._spin_box_rz.setValue(self._slider_box_rz.value() / 10.0)
        # Size sliders: integer / 100.0 → float dimension
        self._spin_box_w.setValue(self._slider_box_w.value() / 100.0)
        self._spin_box_h.setValue(self._slider_box_h.value() / 100.0)
        self._spin_box_d.setValue(self._slider_box_d.value() / 100.0)
        self._updating_box_controls = False
        # Trigger full update (since spinbox signals were blocked by guard)
        self._on_box_control_change()

    def _sync_box_sliders_from_spinboxes(self):
        """Push current spinbox values to slider positions (guarded)."""
        self._updating_box_controls = True
        self._slider_box_cx.setValue(int(round(self._spin_box_cx.value() * 100)))
        self._slider_box_cy.setValue(int(round(self._spin_box_cy.value() * 100)))
        self._slider_box_cz.setValue(int(round(self._spin_box_cz.value() * 100)))
        self._slider_box_rx.setValue(int(round(self._spin_box_rx.value() * 10)))
        self._slider_box_ry.setValue(int(round(self._spin_box_ry.value() * 10)))
        self._slider_box_rz.setValue(int(round(self._spin_box_rz.value() * 10)))
        self._slider_box_w.setValue(int(round(self._spin_box_w.value() * 100)))
        self._slider_box_h.setValue(int(round(self._spin_box_h.value() * 100)))
        self._slider_box_d.setValue(int(round(self._spin_box_d.value() * 100)))
        self._updating_box_controls = False

    def _init_box_slider_ranges(self, mesh):
        """Set slider integer ranges based on mesh bounds.

        Center sliders: mesh_min - margin .. mesh_max + margin  (×100 for 0.01 step)
        Size sliders:   0.01 .. 2× mesh extent per axis  (×100)
        Rotation sliders: fixed 0-3600 (0.0°-360.0°)
        """
        bounds = np.array(mesh.bounds).reshape(3, 2)
        extents = bounds.ptp(axis=1)
        margin = extents * 0.5  # 50% margin

        for slider, bmin, bmax, m in [
            (self._slider_box_cx, bounds[0, 0], bounds[0, 1], margin[0]),
            (self._slider_box_cy, bounds[1, 0], bounds[1, 1], margin[1]),
            (self._slider_box_cz, bounds[2, 0], bounds[2, 1], margin[2]),
        ]:
            slider.setRange(int((bmin - m) * 100), int((bmax + m) * 100))

        for slider, ext in [
            (self._slider_box_w, extents[0]),
            (self._slider_box_h, extents[1]),
            (self._slider_box_d, extents[2]),
        ]:
            slider.setRange(1, int(ext * 2 * 100))  # min 0.01, max 2× extent

        # Rotation sliders: always 0-3600
        for slider in (self._slider_box_rx, self._slider_box_ry, self._slider_box_rz):
            slider.setRange(0, 3600)

    def _push_transform_to_box_widget(self):
        """Push current center/rotation/size to the VTK box widget."""
        if not self.plotter.box_widgets:
            return
        if self._box_initial_center is None or self._box_initial_half_extents is None:
            return
        widget = self.plotter.box_widgets[-1]
        c0 = self._box_initial_center
        h0 = self._box_initial_half_extents
        c = self._box_center
        h = self._box_half_extents
        R = self._euler_rotation_matrix(*self._box_rotation_deg)

        t = vtk.vtkTransform()
        t.PostMultiply()
        # 1. Undo initial placement center
        t.Translate(-c0[0], -c0[1], -c0[2])
        # 2. Scale from initial to desired size
        sx = h[0] / h0[0] if h0[0] > 1e-12 else 1.0
        sy = h[1] / h0[1] if h0[1] > 1e-12 else 1.0
        sz = h[2] / h0[2] if h0[2] > 1e-12 else 1.0
        t.Scale(sx, sy, sz)
        # 3. Rotate
        rx, ry, rz = self._box_rotation_deg
        t.RotateX(float(rx))
        t.RotateY(float(ry))
        t.RotateZ(float(rz))
        # 4. Move to desired center
        t.Translate(c[0], c[1], c[2])

        widget.SetTransform(t)
        self.plotter.render()

    def _update_constraint_box(self):
        """Recompute 6 planes from current box state and refresh preview.

        The VTK box widget provides the visual — no wireframe actor needed.
        """
        if self._box_center is None:
            return

        center = self._box_center
        R = self._euler_rotation_matrix(*self._box_rotation_deg)

        # 6 planes with outward-pointing normals
        axes = [R[:, 0], R[:, 1], R[:, 2]]
        planes = []
        for i, axis in enumerate(axes):
            h = self._box_half_extents[i]
            planes.append((axis.copy(), (center + axis * h).copy()))
            planes.append((-axis.copy(), (center - axis * h).copy()))
        self._current_box_planes_data = planes

        # Green sphere marker at box center
        radius = np.min(self._box_half_extents) * 0.1
        self.plotter.add_mesh(
            pv.Sphere(center=center, radius=radius),
            color="green", name="box_center_marker",
        )

        self._update_preview()

    def _on_reset_box(self):
        """Reset box controls to initial values (mesh center, no rotation, default size)."""
        mesh = self.engine.get_wall_mesh()
        if mesh is None:
            return
        center = np.array(mesh.center, dtype=float)
        bounds = np.array(mesh.bounds).reshape(3, 2)
        extents = bounds.ptp(axis=1)
        half_extents = extents * 0.3 / 2.0

        self._updating_box_controls = True
        self._spin_box_cx.setValue(float(center[0]))
        self._spin_box_cy.setValue(float(center[1]))
        self._spin_box_cz.setValue(float(center[2]))
        self._spin_box_rx.setValue(0.0)
        self._spin_box_ry.setValue(0.0)
        self._spin_box_rz.setValue(0.0)
        self._spin_box_w.setValue(float(half_extents[0] * 2))
        self._spin_box_h.setValue(float(half_extents[1] * 2))
        self._spin_box_d.setValue(float(half_extents[2] * 2))
        self._updating_box_controls = False
        self._sync_box_sliders_from_spinboxes()

        self._box_center = center
        self._box_rotation_deg = np.array([0.0, 0.0, 0.0])
        self._box_half_extents = half_extents
        self._box_initial_center = center.copy()
        self._box_initial_half_extents = half_extents.copy()

        # Re-place VTK widget at reset bounds
        self.plotter.remove_actor("box_center_marker")
        self.plotter.clear_box_widgets()
        box_bounds = [
            center[0] - half_extents[0], center[0] + half_extents[0],
            center[1] - half_extents[1], center[1] + half_extents[1],
            center[2] - half_extents[2], center[2] + half_extents[2],
        ]
        self.plotter.add_box_widget(
            self._constraint_box_callback,
            bounds=box_bounds,
            factor=1.0,
            rotation_enabled=True,
            color=(0.2, 0.8, 0.4),
            use_planes=True,
        )
        self._update_constraint_box()

    def _update_preview(self):
        """Show a yellow slice preview (clipped to box) and a green normal arrow."""
        wall = self.engine.get_wall_mesh()
        if wall is None or wall.n_cells == 0:
            return

        # Remove previous preview actors
        if self._preview_actor is not None:
            self.plotter.remove_actor(self._preview_actor, render=False)
            self._preview_actor = None
        if self._arrow_actor is not None:
            self.plotter.remove_actor(self._arrow_actor, render=False)
            self._arrow_actor = None

        try:
            sliced = wall.slice(
                normal=self._current_plane_normal,
                origin=self._current_plane_origin,
            )
            # Filter preview to only show loops inside the box region
            if self._current_box_planes_data and sliced.n_cells > 0:
                stripper = vtk.vtkStripper()
                stripper.SetInputData(sliced)
                stripper.JoinContiguousSegmentsOn()
                stripper.Update()
                sliced = STLClipperEngine._filter_loops_by_box(
                    pv.wrap(stripper.GetOutput()), self._current_box_planes_data
                )

            if sliced.n_cells > 0:
                self._preview_actor = self.plotter.add_mesh(
                    sliced, color=PREVIEW_COLOR, line_width=4,
                    name="_clip_preview", render=False,
                    reset_camera=False,
                )
        except Exception:
            pass

        # Normal direction arrow (green = keep side)
        arrow_length = np.linalg.norm(
            np.array(wall.bounds).reshape(3, 2).ptp(axis=1)
        ) * 0.15
        arrow = pv.Arrow(
            start=self._current_plane_origin,
            direction=self._current_plane_normal,
            scale=arrow_length,
        )
        self._arrow_actor = self.plotter.add_mesh(
            arrow, color="green", name="_normal_arrow", render=False,
            reset_camera=False,
        )
        self.plotter.render()

    def _on_flip_normal(self):
        if self._current_plane_normal is not None:
            self._current_plane_normal = -self._current_plane_normal
            self._sync_normal_sliders_from_state()
            self._update_plane_widget_position()
            self._update_preview()

    def _on_cancel(self):
        self._cancel_clip_widgets()
        self._refresh_display()
        self._update_button_states()

    def _cancel_clip_widgets(self):
        """Remove all clip widgets and reset state."""
        self._plane_widget_active = False
        self._constraint_box_active = False
        self._plane_confirmed = False
        self._current_box_planes_data = None
        self._updating_box_controls = False  # defensive reset
        self.plotter.clear_plane_widgets()
        self.plotter.clear_box_widgets()
        # Remove wireframe box actor (fallback)
        if self._box_actor is not None:
            self.plotter.remove_actor(self._box_actor, render=False)
            self._box_actor = None
        self.plotter.remove_actor("box_center_marker")
        self._box_center = None
        self._box_rotation_deg = None
        self._box_half_extents = None
        self._box_initial_center = None
        self._box_initial_half_extents = None
        self._box_controls_box.setVisible(False)
        if self._static_plane_actor is not None:
            self.plotter.remove_actor(self._static_plane_actor, render=False)
            self._static_plane_actor = None
        if self._preview_actor is not None:
            self.plotter.remove_actor(self._preview_actor, render=False)
            self._preview_actor = None
        if self._arrow_actor is not None:
            self.plotter.remove_actor(self._arrow_actor, render=False)
            self._arrow_actor = None
        self._show_plane_controls(False)

    # ------------------------------------------------------------------
    # Manual plane controls (trackpad-friendly)
    # ------------------------------------------------------------------

    def _on_manual_origin_change(self):
        """Update plane position from manual spinbox input."""
        if self._updating_controls:
            return
        origin = np.array([
            self._spin_x.value(),
            self._spin_y.value(),
            self._spin_z.value(),
        ])
        self._current_plane_origin = origin
        self._update_plane_widget_position()
        self._update_preview()

    def _on_normal_slider_change(self):
        """Compute plane normal from elevation/azimuth sliders."""
        if self._updating_controls:
            return
        elev = np.radians(self._slider_elev.value())
        azim = np.radians(self._slider_azim.value())
        self._current_plane_normal = np.array([
            np.sin(elev) * np.cos(azim),
            np.sin(elev) * np.sin(azim),
            np.cos(elev),
        ])
        self._elev_label.setText(f"{self._slider_elev.value()}°")
        self._azim_label.setText(f"{self._slider_azim.value()}°")
        self._update_plane_widget_position()
        self._update_preview()

    def _on_normal_preset_angles(self, elev, azim):
        """Set sliders to preset axis angles (triggers _on_normal_slider_change)."""
        self._updating_controls = True
        self._slider_elev.setValue(elev)
        self._updating_controls = False
        self._slider_azim.setValue(azim)  # this triggers the slider change

    def _sync_normal_sliders_from_state(self):
        """Convert current normal vector back to elevation/azimuth and update sliders."""
        if self._current_plane_normal is None:
            return
        n = self._current_plane_normal / np.linalg.norm(self._current_plane_normal)
        elev = int(round(np.degrees(np.arccos(np.clip(n[2], -1, 1)))))
        azim = int(round(np.degrees(np.arctan2(n[1], n[0])))) % 360
        self._updating_controls = True
        self._slider_elev.setValue(elev)
        self._slider_azim.setValue(azim)
        self._elev_label.setText(f"{elev}°")
        self._azim_label.setText(f"{azim}°")
        self._updating_controls = False

    def _on_nudge(self, direction):
        """Move plane along its current normal by step amount."""
        if self._current_plane_origin is None or self._current_plane_normal is None:
            return
        step = self._spin_step.value() * direction
        n = self._current_plane_normal / np.linalg.norm(self._current_plane_normal)
        self._current_plane_origin = self._current_plane_origin + n * step
        self._sync_controls_from_state()
        self._update_plane_widget_position()
        self._update_preview()

    def _update_plane_widget_position(self):
        """Push current origin/normal to the VTK plane widget (if active)."""
        if not self.plotter.plane_widgets:
            return
        widget = self.plotter.plane_widgets[-1]
        widget.SetOrigin(*self._current_plane_origin)
        widget.SetNormal(*self._current_plane_normal)
        widget.UpdatePlacement()
        self.plotter.render()

    def _sync_controls_from_state(self):
        """Update spinbox values from internal plane state (guarded against recursion)."""
        if self._current_plane_origin is None:
            return
        self._updating_controls = True
        self._spin_x.setValue(float(self._current_plane_origin[0]))
        self._spin_y.setValue(float(self._current_plane_origin[1]))
        self._spin_z.setValue(float(self._current_plane_origin[2]))
        self._updating_controls = False
        self._sync_normal_sliders_from_state()

    def _show_plane_controls(self, visible: bool):
        """Show/hide the manual plane controls panel."""
        self._plane_controls_box.setVisible(visible)
        if visible and self._current_plane_origin is not None:
            self._sync_controls_from_state()
            self._sync_normal_sliders_from_state()

    # ------------------------------------------------------------------
    # Confirm / store clip
    # ------------------------------------------------------------------

    def _on_confirm_clip(self):
        if self._current_plane_origin is None:
            QMessageBox.information(
                self, "Position Plane",
                "Move the plane widget to set the clip position first.",
            )
            return

        # Suggest a default name
        existing_names = {c.name for c in self.engine.clips}
        if "inlet" not in existing_names:
            default = "inlet"
        else:
            idx = 1
            while f"outlet_{idx}" in existing_names:
                idx += 1
            default = f"outlet_{idx}"

        name, ok = QInputDialog.getText(
            self, "Patch Name", "Name for this boundary patch:", text=default,
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Reject duplicate names
        if name in existing_names or name == "wall":
            QMessageBox.warning(self, "Duplicate Name", f"'{name}' is already used. Choose another.")
            return

        origin = self._current_plane_origin.copy()
        normal = self._current_plane_normal.copy()
        box_planes_data = None
        if self._current_box_planes_data:
            box_planes_data = [(n.copy(), p.copy()) for n, p in self._current_box_planes_data]

        try:
            self.engine.add_clip(name, origin, normal, box_planes_data)
        except RuntimeError as e:
            QMessageBox.warning(self, "Clip Error", str(e))
            return

        self._cancel_clip_widgets()
        self._refresh_display()
        self._refresh_patch_list()
        self._update_status()
        self._update_button_states()

    # ------------------------------------------------------------------
    # Patch management
    # ------------------------------------------------------------------

    def _on_rename(self):
        row = self.patch_list.currentRow()
        if row < 0 or row >= len(self.engine.clips):
            return
        old_name = self.engine.clips[row].name
        new_name, ok = QInputDialog.getText(
            self, "Rename Patch", "New name:", text=old_name,
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        existing = {c.name for i, c in enumerate(self.engine.clips) if i != row}
        if new_name in existing or new_name == "wall":
            QMessageBox.warning(self, "Duplicate Name", f"'{new_name}' is already used.")
            return
        self.engine.rename_clip(row, new_name)
        self._centerline_mesh = None  # invalidate — inlet/outlet classification may have changed
        self._refresh_patch_list()
        self._refresh_display()
        self._update_status()

    def _on_delete(self):
        row = self.patch_list.currentRow()
        if row < 0 or row >= len(self.engine.clips):
            return
        name = self.engine.clips[row].name
        reply = QMessageBox.question(
            self, "Delete Clip", f"Remove clip '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.engine.remove_clip(row)
            self._centerline_mesh = None  # invalidate — clip set changed
            self._refresh_display()
            self._refresh_patch_list()
            self._update_status()
            self._update_button_states()

    # ------------------------------------------------------------------
    # Trim region (freehand lasso)
    # ------------------------------------------------------------------

    def _trim_safe(self, fn):
        """Wrap a VTK observer callback so exceptions are logged, not thrown
        back into VTK (which can crash the app)."""
        def _cb(_obj=None, _evt=None):
            try:
                fn()
            except Exception:
                logger.exception("trim callback failed")
        return _cb

    def _toggle_trim_mode(self, checked):
        vtk_iren = self.plotter.iren.interactor
        if checked:
            self._trim_points = []
            self._trim_drawing = False
            # Capture the lasso with a do-nothing style and observe the mouse
            # events ON that active style. Going through pyvista's add_observer
            # misroutes LeftButtonReleaseEvent to the previous (now-inactive)
            # style (pyvista #4976), so the stroke never completes.
            self._trim_saved_style = vtk_iren.GetInteractorStyle()
            self._trim_style = vtk.vtkInteractorStyleUser()
            vtk_iren.SetInteractorStyle(self._trim_style)
            s = self._trim_style
            self._trim_obs = [
                s.AddObserver("LeftButtonPressEvent", self._trim_safe(self._on_trim_press)),
                s.AddObserver("MouseMoveEvent", self._trim_safe(self._on_trim_move)),
                s.AddObserver("LeftButtonReleaseEvent", self._trim_safe(self._on_trim_release)),
            ]
            logger.info("trim mode ON: %d observers on active style", len(self._trim_obs))
            self.status.showMessage(
                "Trim mode: drag to lasso a region to delete. Toggle off to exit."
            )
        else:
            style = getattr(self, "_trim_style", None)
            if style is not None:
                for obs in getattr(self, "_trim_obs", []):
                    style.RemoveObserver(obs)
            self._trim_obs = []
            if getattr(self, "_trim_saved_style", None) is not None:
                vtk_iren.SetInteractorStyle(self._trim_saved_style)
            self._end_lasso_overlay()
            logger.info("trim mode OFF")
            self.status.showMessage("Trim mode off.")

    def _on_trim_press(self):
        self._trim_drawing = True
        self._trim_points = [self.plotter.iren.get_event_position()]
        self._start_lasso_overlay()
        self._update_lasso_overlay()
        logger.info("trim press at %s", self._trim_points[0])

    def _on_trim_move(self):
        if getattr(self, "_trim_drawing", False):
            self._trim_points.append(self.plotter.iren.get_event_position())
            self._update_lasso_overlay()

    def _on_trim_release(self):
        self._trim_drawing = False
        self._end_lasso_overlay()
        points = list(self._trim_points)
        self._trim_points = []
        logger.info("trim release: %d points", len(points))
        if len(points) < 3:
            self.status.showMessage("Trim: stroke too short — draw a closed shape.")
            return
        self._apply_trim(points)

    # --- Lasso outline overlay (2D screen-space polyline while drawing) -------

    def _start_lasso_overlay(self):
        """Create a fresh yellow 2D polyline actor for the in-progress lasso."""
        self._end_lasso_overlay()
        self._lasso_pts = vtk.vtkPoints()
        self._lasso_cells = vtk.vtkCellArray()
        self._lasso_poly = vtk.vtkPolyData()
        self._lasso_poly.SetPoints(self._lasso_pts)
        self._lasso_poly.SetLines(self._lasso_cells)
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._lasso_poly)
        mapper.SetTransformCoordinate(coord)
        self._lasso_actor = vtk.vtkActor2D()
        self._lasso_actor.SetMapper(mapper)
        self._lasso_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        self._lasso_actor.GetProperty().SetLineWidth(2.0)
        self.plotter.renderer.AddActor2D(self._lasso_actor)

    def _update_lasso_overlay(self):
        """Rebuild the polyline from the captured display points and redraw."""
        if getattr(self, "_lasso_actor", None) is None:
            return
        pts = self._trim_points
        self._lasso_pts.Reset()
        self._lasso_cells.Reset()
        for x, y in pts:
            self._lasso_pts.InsertNextPoint(float(x), float(y), 0.0)
        n = len(pts)
        if n >= 2:
            self._lasso_cells.InsertNextCell(n + 1)
            for i in range(n):
                self._lasso_cells.InsertCellPoint(i)
            self._lasso_cells.InsertCellPoint(0)  # close the loop
        self._lasso_poly.Modified()
        self.plotter.render()

    def _end_lasso_overlay(self):
        """Remove the lasso overlay actor if present."""
        actor = getattr(self, "_lasso_actor", None)
        if actor is not None:
            try:
                self.plotter.renderer.RemoveActor2D(actor)
            except Exception:
                pass
            self._lasso_actor = None
            self.plotter.render()

    def _apply_trim(self, display_points):
        cam = self.plotter.camera
        size = self.plotter.render_window.GetSize()
        width, height = int(size[0]), int(size[1])
        aspect = width / height if height else 1.0
        vtk_m = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
        matrix = np.array([[vtk_m.GetElement(i, j) for j in range(4)] for i in range(4)])
        logger.info("trim apply: %d pts, viewport=%s", len(display_points), (width, height))
        try:
            result = self.engine.trim_by_screen_polygon(display_points, matrix, (width, height))
        except ValueError as exc:
            self.status.showMessage(str(exc))
            logger.info("trim refused: %s", exc)
            return
        if result is None:
            self.status.showMessage("Trim: nothing selected.")
            logger.info("trim: nothing selected (0 cells inside polygon)")
            return
        self._refresh_display()
        self.status.showMessage(f"Trimmed region. Mesh now {result.n_cells} cells. Ctrl+Z to undo.")
        logger.info("trim done: mesh now %d cells", result.n_cells)

    def _undo_trim(self):
        if self.engine.undo_trim():
            self._refresh_display()
            self.status.showMessage("Undid last trim.")
        else:
            self.status.showMessage("Nothing to undo.")

    def _refresh_patch_list(self):
        self.patch_list.clear()
        for clip_def in self.engine.clips:
            n = clip_def.cap_mesh.n_cells if clip_def.cap_mesh else 0
            r, g, b = [int(c * 255) for c in clip_def.color]
            item = QListWidgetItem(f"{clip_def.name}  ({n:,} faces)")
            item.setForeground(Qt.black)
            item.setBackground(QColor(r, g, b, 60))
            self.patch_list.addItem(item)
        # Keep the 'Clip & Save STL' tab's list in sync — same source of truth.
        self._refresh_clip_save_list()

    # ------------------------------------------------------------------
    # Centerline
    # ------------------------------------------------------------------

    def _on_compute_centerline(self):
        """Build capped surface + cap-centroid seeds, launch VMTK."""
        clips = self.engine.clips
        inlet_clips = [c for c in clips if "inlet" in c.name.lower()]
        outlet_clips = [c for c in clips if "outlet" in c.name.lower()]

        if not inlet_clips or not outlet_clips:
            QMessageBox.warning(self, "Centerline Error",
                                "Need at least one clip named 'inlet' and one named 'outlet'.")
            return

        wall = self.engine.get_wall_mesh()
        if wall is None or wall.n_cells == 0:
            return

        # Cap the open wall mesh so VMTK's Voronoi diagram doesn't degenerate
        capped = wall.copy()
        for c in clips:
            if c.cap_mesh is not None and c.cap_mesh.n_cells > 0:
                capped = capped.merge(c.cap_mesh)

        # Use cap centroids as seeds (guaranteed on surface); fall back to origin
        source_pts = []
        for c in inlet_clips:
            pt = c.cap_mesh.center if (c.cap_mesh is not None and c.cap_mesh.n_cells > 0) else c.origin.tolist()
            source_pts.extend(list(pt))
        target_pts = []
        for c in outlet_clips:
            pt = c.cap_mesh.center if (c.cap_mesh is not None and c.cap_mesh.n_cells > 0) else c.origin.tolist()
            target_pts.extend(list(pt))
        logger.info("Seed points (cap centroids): source=%s, target=%s", source_pts, target_pts)

        worker = CenterlineWorker(capped, source_pts, target_pts)
        worker.result_ready.connect(self._on_centerline_finished)
        worker.failed.connect(self._on_centerline_failed)
        worker.finished.connect(self._cleanup_centerline_worker)  # QThread built-in
        self._centerline_worker = worker
        worker.start()
        self.status.showMessage("Computing centerline...")
        self._update_button_states()

    def _on_centerline_finished(self, result):
        """Slot: VMTK computation succeeded (result_ready signal)."""
        logger.info("Centerline finished — result: n_points=%s, n_cells=%s",
                     result.n_points if result is not None else None,
                     result.n_cells if result is not None else None)
        if result is None or result.n_points == 0:
            QMessageBox.warning(self, "Centerline Warning",
                                "VMTK returned an empty centerline — seed points may be unreachable.")
            self.status.showMessage("Centerline computation returned empty result.")
            return
        logger.info("Storing centerline mesh (%d points, %d cells)",
                     result.n_points, result.n_cells)
        self._centerline_mesh = result
        self._refresh_display()
        self._update_button_states()
        self.status.showMessage("Centerline computed successfully.")

    def _on_centerline_failed(self, error_msg):
        """Slot: VMTK computation failed."""
        logger.error("Centerline failed: %s", error_msg)
        QMessageBox.warning(self, "Centerline Error", error_msg)
        self._update_button_states()

    def _cleanup_centerline_worker(self):
        """Slot: QThread.finished — safe to release the worker now."""
        self._centerline_worker = None

    def _on_save_centerline(self):
        """Export the computed centerline to the case directory."""
        if self._centerline_mesh is None:
            return
        case_dir = self._get_or_create_case_dir()
        if not case_dir:
            return
        filepath = os.path.join(case_dir, "centerline.vtp")
        try:
            self._centerline_mesh.save(filepath)
            QMessageBox.information(
                self, "Export Complete",
                f"Centerline saved to:\n{filepath}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_clear_centerline(self):
        """Remove the computed centerline from the display."""
        self._centerline_mesh = None
        self._refresh_display()
        self._update_button_states()
        self.status.showMessage("Centerline cleared.")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _refresh_display(self):
        self.plotter.clear()

        wall = self.engine.get_wall_mesh()
        if wall is None or wall.n_cells == 0:
            self.plotter.render()
            return

        # Wall mesh — optionally opaque, optionally with surface mesh edges
        show_mesh = self._btn_show_mesh_edges.isChecked()
        wall_opacity = 1.0 if self._btn_opaque_wall.isChecked() else 0.4
        self.plotter.add_mesh(
            wall, color=WALL_COLOR, opacity=wall_opacity,
            show_edges=show_mesh, edge_color="black", line_width=0.5,
            specular=0.15, specular_power=20.0, ambient=0.15, diffuse=0.9,
            name="wall",
        )

        # Cap patches — name-based colors with white edges for visibility
        for clip_def in self.engine.clips:
            if clip_def.cap_mesh and clip_def.cap_mesh.n_cells > 0:
                self.plotter.add_mesh(
                    clip_def.cap_mesh, color=clip_def.color, opacity=1.0,
                    show_edges=True, edge_color="white", line_width=2,
                    name=f"cap_{clip_def.name}",
                )

        # Centerline — yellow tube
        has_cl = self._centerline_mesh is not None and self._centerline_mesh.n_points > 0
        logger.debug("Refresh display: _centerline_mesh exists=%s, n_points=%s, n_cells=%s",
                      self._centerline_mesh is not None,
                      self._centerline_mesh.n_points if self._centerline_mesh is not None else 0,
                      self._centerline_mesh.n_cells if self._centerline_mesh is not None else 0)
        if has_cl:
            tube = self._centerline_mesh.tube(radius=0.3)
            logger.debug("Tube generated: n_points=%s, n_cells=%s",
                          tube.n_points if tube is not None else None,
                          tube.n_cells if tube is not None else None)
            if tube is not None and tube.n_points > 0:
                self.plotter.add_mesh(
                    tube, color="yellow", opacity=1.0,
                    name="centerline",
                )
                logger.info("Centerline tube added to plotter")
            else:
                logger.warning("Tube generation produced empty mesh — centerline not rendered")

        self.plotter.reset_camera()
        self.plotter.render()

        # Update wall face count label
        n_wall = wall.n_cells if wall else 0
        self._lbl_wall_faces.setText(f"Wall: {n_wall:,} faces")

        # Update bounding box info — show raw min..max per axis + extent
        if self.engine.original_mesh is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self.engine.original_mesh.bounds
            dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
            self._lbl_bounds_x.setText(f"X: {xmin:.4g} .. {xmax:.4g}  [{dx:.4g}]")
            self._lbl_bounds_y.setText(f"Y: {ymin:.4g} .. {ymax:.4g}  [{dy:.4g}]")
            self._lbl_bounds_z.setText(f"Z: {zmin:.4g} .. {zmax:.4g}  [{dz:.4g}]")
        else:
            self._lbl_bounds_x.setText("X: —")
            self._lbl_bounds_y.setText("Y: —")
            self._lbl_bounds_z.setText("Z: —")

        # Update geometry quality indicators
        quality = self.engine.geometry_quality()
        self._boundary_mesh = quality["boundary_mesh"]
        n_profiles = quality["open_profiles"]
        n_edges = quality["open_edges"]
        n_clips = len(self.engine.clips)

        if n_profiles == 0 and n_clips == 0:
            self._lbl_open_profiles.setText("Open profiles: —")
            self._lbl_open_profiles.setStyleSheet("")
        elif n_profiles == n_clips:
            self._lbl_open_profiles.setText(
                f"Open profiles: {n_profiles} (expected: {n_clips}) \u2713"
            )
            self._lbl_open_profiles.setStyleSheet("color: green;")
        else:
            self._lbl_open_profiles.setText(
                f"Open profiles: {n_profiles} (expected: {n_clips}) \u2717"
            )
            self._lbl_open_profiles.setStyleSheet("color: red;")

        self._lbl_open_edges.setText(f"Open edges: {n_edges}")

        # Update non-manifold indicators
        self._non_manifold_mesh = quality["non_manifold_mesh"]
        n_nm = quality["non_manifold_edges"]
        is_mf = quality["is_manifold"]

        if n_nm == 0:
            self._lbl_non_manifold.setText(f"Non-manifold edges: {n_nm} \u2713")
            self._lbl_non_manifold.setStyleSheet("color: green;")
        else:
            self._lbl_non_manifold.setText(f"Non-manifold edges: {n_nm} \u2717")
            self._lbl_non_manifold.setStyleSheet("color: red;")

        if is_mf is None:
            self._lbl_manifold.setText("Manifold: \u2014 (no clips)")
            self._lbl_manifold.setStyleSheet("color: gray;")
        elif is_mf:
            self._lbl_manifold.setText("Manifold: \u2713")
            self._lbl_manifold.setStyleSheet("color: green;")
        else:
            self._lbl_manifold.setText("Manifold: \u2717")
            self._lbl_manifold.setStyleSheet("color: red;")

        # Re-render boundary edges if toggle is on
        if self._btn_show_boundary.isChecked() and self._boundary_mesh is not None and self._boundary_mesh.n_cells > 0:
            self.plotter.add_mesh(
                self._boundary_mesh, color="red", line_width=4,
                name="boundary_edges",
            )

        # Re-render non-manifold edges if toggle is on
        if self._btn_show_non_manifold.isChecked() and self._non_manifold_mesh is not None:
            if self._non_manifold_mesh.n_cells > 0:
                self.plotter.add_mesh(
                    self._non_manifold_mesh, color="magenta", line_width=4,
                    name="non_manifold_edges",
                )

        self.plotter.render()

    def _on_toggle_mesh_edges(self):
        """Toggle surface mesh edge visualization on the wall."""
        self._refresh_display()

    def _on_toggle_opaque_wall(self):
        """Toggle wall opacity between transparent (0.4) and opaque (1.0)."""
        self._refresh_display()

    def _on_toggle_boundary_edges(self):
        """Toggle red boundary edge visualization in the 3D viewport."""
        if self._btn_show_boundary.isChecked():
            if self._boundary_mesh is not None and self._boundary_mesh.n_cells > 0:
                self.plotter.add_mesh(
                    self._boundary_mesh, color="red", line_width=4,
                    name="boundary_edges",
                )
                self.plotter.render()
        else:
            self.plotter.remove_actor("boundary_edges")
            self.plotter.render()

    def _on_toggle_non_manifold(self):
        """Toggle magenta non-manifold edge visualization in the 3D viewport."""
        if self._btn_show_non_manifold.isChecked():
            if self._non_manifold_mesh is not None and self._non_manifold_mesh.n_cells > 0:
                self.plotter.add_mesh(
                    self._non_manifold_mesh, color="magenta", line_width=4,
                    name="non_manifold_edges",
                )
                self.plotter.render()
        else:
            self.plotter.remove_actor("non_manifold_edges")
            self.plotter.render()

    # ------------------------------------------------------------------
    # View controls
    # ------------------------------------------------------------------

    def _on_view_pos_x(self):
        """Camera looks from +X toward origin (right side view)."""
        self.plotter.view_yz(negative=True)

    def _on_view_neg_x(self):
        """Camera looks from -X toward origin (left side view)."""
        self.plotter.view_yz(negative=False)

    def _on_view_pos_y(self):
        """Camera looks from +Y toward origin (back view)."""
        self.plotter.view_xz(negative=True)

    def _on_view_neg_y(self):
        """Camera looks from -Y toward origin (front view)."""
        self.plotter.view_xz(negative=False)

    def _on_view_pos_z(self):
        """Camera looks from +Z toward origin (top view)."""
        self.plotter.view_xy(negative=False)

    def _on_view_neg_z(self):
        """Camera looks from -Z toward origin (bottom view)."""
        self.plotter.view_xy(negative=True)

    def _on_zoom_to_fit(self):
        """Reset camera to fit all visible actors."""
        self.plotter.reset_camera()

    def _update_status(self):
        if self.engine.original_mesh is None:
            return
        n = self.engine.original_mesh.n_cells
        nc = len(self.engine.clips)
        names = ", ".join(c.name for c in self.engine.clips) if nc else "none"
        self.status.showMessage(f"Original: {n:,} faces | {nc} clips ({names})")

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _get_or_create_case_dir(self) -> Optional[str]:
        """Prompt user for a case directory name and create it next to the source STL.

        Returns the case directory path, or None if the user cancels.
        """
        from pathlib import Path

        if self._loaded_filepath is None:
            return None

        stl_path = Path(self._loaded_filepath)
        default_name = stl_path.stem

        name, ok = QInputDialog.getText(
            self, "Case Directory Name",
            "Name for the case directory:",
            text=default_name,
        )
        if not ok or not name.strip():
            return None

        case_dir = stl_path.parent / name.strip()
        case_dir.mkdir(parents=True, exist_ok=True)
        return str(case_dir)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_separate(self):
        if not self.engine.clips:
            return
        case_dir = self._get_or_create_case_dir()
        if not case_dir:
            return
        sf = self._spin_scale.value()
        sep_dir = os.path.join(case_dir, "separate")
        try:
            self.engine.export_separate_stl(sep_dir, scale_factor=sf)
            planes_path = os.path.join(case_dir, "clip_planes.json")
            self.engine.export_clip_planes(planes_path)
            files = [f"{c.name}.stl" for c in self.engine.clips] + ["wall.stl"]
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(files)} files to:\n{sep_dir}\n"
                f"Scale factor: ×{sf}\n\n"
                + "\n".join(files)
                + f"\n\nClip planes JSON:\n{planes_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_combined(self):
        if not self.engine.clips:
            return
        case_dir = self._get_or_create_case_dir()
        if not case_dir:
            return
        sf = self._spin_scale.value()
        filepath = os.path.join(case_dir, "boundary.stl")
        try:
            self.engine.export_combined_stl(filepath, scale_factor=sf)
            planes_path = os.path.join(case_dir, "clip_planes.json")
            self.engine.export_clip_planes(planes_path)
            QMessageBox.information(
                self, "Export Complete",
                f"Combined STL saved to:\n{filepath}\n"
                f"Scale factor: ×{sf}\n\n"
                f"Clip planes JSON:\n{planes_path}\n\n"
                f"Patches: {', '.join(c.name for c in self.engine.clips)}, wall",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_save_openfoam_stl(self):
        if not self.engine.clips:
            return
        default_name = "boundary.stl"
        if self._loaded_filepath:
            stem = os.path.splitext(os.path.basename(self._loaded_filepath))[0]
            default_name = f"{stem}_of.stl"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save OpenFOAM multi-solid STL", default_name, "STL files (*.stl)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".stl"):
            filepath += ".stl"
        sf = self._spin_scale.value()
        try:
            self.engine.export_combined_stl(filepath, scale_factor=sf)
            patches = ", ".join(c.name for c in self.engine.clips) + ", wall"
            QMessageBox.information(
                self, "Saved",
                f"Multi-solid STL saved to:\n{filepath}\n"
                f"Scale factor: \u00d7{sf}\n\n"
                f"Solids: {patches}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_export_openfoam(self):
        if not self.engine.clips or self._loaded_filepath is None:
            return
        case_dir = self._get_or_create_case_dir()
        if not case_dir:
            return
        # Use scale from Export Settings tab (OF-specific)
        sf = self._spin_scale_of.value()
        stl_filename = os.path.basename(self._loaded_filepath)

        # Read all spinbox values into template_params dict
        simulation_type = self._combo_sim_type.currentData()
        turb_params = {"Cs": self._export_cs.value()}
        if simulation_type == "rans":
            turb_params["k"] = self._export_k.value()
            turb_params["omega"] = self._export_omega.value()

        template_params = {
            "mesh": {
                "maxCellSize": self._export_max_cell_size.value(),
                "boundaryCellSize": self._export_boundary_cell_size.value(),
                "wallCellSize": self._export_wall_cell_size.value(),
                "nLayers": self._export_n_layers.value(),
                "thicknessRatio": self._export_thickness_ratio.value(),
            },
            "solver": {
                "endTime": self._export_end_time.value(),
                "deltaT": self._export_delta_t.value(),
                "writeInterval": self._export_write_interval.value(),
                "maxCo": self._export_max_co.value(),
                "maxDeltaT": self._export_max_delta_t.value(),
            },
            "fluid": {
                "nu": self._export_nu.value(),
            },
            "turbulence": turb_params,
            "inlet": {
                "velocityMagnitude": self._export_velocity_mag.value(),
            },
            "decompose": {
                "nProcs": self._export_n_procs.value(),
            },
        }

        try:
            result = self.engine.export_openfoam_case(
                case_dir, stl_filename, self._centerline_mesh,
                scale_factor=sf,
                template_params=template_params,
                simulation_type=simulation_type,
            )
            self._last_case_dir = case_dir
            self._run_case_label.setText(f"Case: {case_dir}")
            patches = ", ".join(c.name for c in self.engine.clips) + ", wall"
            # Summarise generated files by category
            bc_files = [k for k in ("p", "U", "nut", "k", "omega") if k in result]
            system_files = [
                k for k in result
                if k not in ("stl_path", "planes_path", "p", "U", "nut",
                             "transportProperties", "turbulenceProperties",
                             "volumetricFlowRate.csv", "env.sh", "Allrun", "Allclean",
                             "run_docker.sh", "diagnose_patches.py", "meshDict",
                             ".foam", "visualize.py", "plot_residuals.py")
            ]
            QMessageBox.information(
                self, "OpenFOAM Case Export Complete",
                f"Case directory: {case_dir}\n"
                f"Scale factor: ×{sf}\n\n"
                f"Patches: {patches}\n\n"
                f"Boundary conditions (0/): {', '.join(bc_files)}\n"
                f"System dictionaries: {', '.join(system_files)}\n"
                f"Scripts: Allrun, Allclean, env.sh, run_docker.sh\n"
                f"ParaView: visualize.py, .foam file\n"
                f"Monitoring: plot_residuals.py\n\n"
                f"Docker (recommended):\n"
                f"  ./run_docker.sh\n\n"
                f"Native OpenFOAM:\n"
                f"  source env.sh && ./Allrun\n\n"
                f"View: pvpython visualize.py\n"
                f"Monitor: python plot_residuals.py  (or --headless)",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = QApplication.instance() or QApplication(sys.argv)

    initial_file = sys.argv[1] if len(sys.argv) > 1 else None
    window = STLClipperApp(initial_file=initial_file)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
