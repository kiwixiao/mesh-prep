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
from typing import Optional

import numpy as np

from . import openfoam_case
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QFont, QKeySequence
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
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
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

PATCH_ID = "patch_id"   # cell-data array on current_mesh; 0 = wall, 1..N = named patch


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
            # Native (pure numpy/scipy/VTK) engine is the default — works in the
            # pip / Apple-Silicon install. vmtk is only a fallback: used when the
            # native path fails AND vmtk happens to be installed (conda).
            try:
                from mesh_prep.centerline_native import compute_centerlines
                result = compute_centerlines(
                    self.surface_mesh, self.source_points, self.target_points)
            except Exception as native_err:
                try:
                    from mesh_prep.centerline import compute_centerlines as _vmtk
                except ImportError:
                    raise native_err
                result = _vmtk(
                    self.surface_mesh, self.source_points, self.target_points)
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


def _project_to_display(points, view_matrix, viewport):
    """Project world points (N,3) through a 4x4 world->clip matrix to display
    (x, y) in logical pixels (VTK bottom-left origin)."""
    pts = np.asarray(points, dtype=float)
    matrix = np.asarray(view_matrix, dtype=float).reshape(4, 4)
    width, height = viewport
    n = pts.shape[0]
    homog = np.hstack([pts, np.ones((n, 1))])
    clip = homog @ matrix.T
    w = clip[:, 3].copy()
    w[w == 0] = 1e-12
    ndc = clip[:, :3] / w[:, None]
    disp_x = (ndc[:, 0] * 0.5 + 0.5) * width
    disp_y = (ndc[:, 1] * 0.5 + 0.5) * height
    return disp_x, disp_y


def plane_from_two_points(a, b, view_dir):
    """Cut plane through points a and b, parallel to view_dir (so it appears edge-on
    from the current camera and slices along the line of sight). Returns
    (origin, normal) with a unit normal and origin at the midpoint, or None if the
    inputs are degenerate (a == b, or b - a parallel to view_dir → zero cross)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(view_dir, dtype=float)
    normal = np.cross(d, b - a)
    n = float(np.linalg.norm(normal))
    if n <= 1e-12:
        return None
    return (a + b) / 2.0, normal / n


class STLClipperEngine:
    """
    Core mesh clipping logic — no Qt dependency.

    Workflow (single compounding current_mesh):
        1. load_stl(path)
        2. clip_and_name / cut / delete / fill / smooth / repair  [repeat, undoable]
        3. export_combined_stl(path) or export_separate_stl(dir)  — one solid per patch_id
    """

    def __init__(self):
        self.original_mesh: Optional[pv.PolyData] = None
        self.patch_names: dict[int, str] = {}             # patch_id -> name (0 = wall)
        self._next_patch_id: int = 1
        self._patch_normals: dict[int, tuple] = {}        # patch_id -> outward normal (clip-created)
        self._trim_history: deque = deque(maxlen=10)
        self._adj_for_mesh = None
        self._adj_ok = False
        self._feature_curves: list = []   # polylines from cuts (sub-feature A)
        # Flood-select (sub-feature B) cached barrier edge-adjacency
        self._flood_adj_for_mesh = None
        self._flood_adj_n_curves = -1
        self._flood_adj = None
        self._flood_ok = False

    @property
    def current_mesh(self):
        """The single working surface. Aliases original_mesh during the migration;
        Task 9 renames the field and drops the alias."""
        return self.original_mesh

    @current_mesh.setter
    def current_mesh(self, mesh):
        self.original_mesh = mesh

    def load_stl(self, filepath: str) -> pv.PolyData:
        mesh = pv.read(filepath)
        if not isinstance(mesh, pv.PolyData):
            raise ValueError(f"Expected PolyData, got {type(mesh).__name__}")
        # Single working surface: triangulated, all-wall labels (0).
        mesh = mesh.triangulate()
        mesh.cell_data[PATCH_ID] = np.zeros(mesh.n_cells, dtype=np.int64)
        self.original_mesh = mesh
        self._trim_history.clear()
        self._feature_curves.clear()
        self.patch_names = {}
        self._next_patch_id = 1
        self._patch_normals = {}
        return mesh

    @staticmethod
    def _carry_labels(result: pv.PolyData, source: pv.PolyData, default_id: int = 0) -> pv.PolyData:
        """Ensure result carries a PATCH_ID cell array. If a filter preserved it,
        keep it; otherwise remap each result cell to the nearest source cell's label."""
        if result is None or result.n_cells == 0:
            return result
        if PATCH_ID in result.cell_data and len(result.cell_data[PATCH_ID]) == result.n_cells:
            return result
        if source is None or PATCH_ID not in source.cell_data or source.n_cells == 0:
            result.cell_data[PATCH_ID] = np.full(result.n_cells, default_id, dtype=np.int64)
            return result
        src_centers = source.cell_centers().points
        src_labels = np.asarray(source.cell_data[PATCH_ID])
        res_centers = result.cell_centers().points
        from scipy.spatial import cKDTree
        _, idx = cKDTree(src_centers).query(res_centers)
        result.cell_data[PATCH_ID] = src_labels[idx].astype(np.int64)
        return result

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

    def _ensure_labels(self) -> None:
        """Guarantee current_mesh carries a valid PATCH_ID array (all-wall default)."""
        m = self.current_mesh
        if m is not None and (PATCH_ID not in m.cell_data
                              or len(m.cell_data[PATCH_ID]) != m.n_cells):
            m.cell_data[PATCH_ID] = np.zeros(m.n_cells, dtype=np.int64)

    def _new_patch_id(self, name: str) -> int:
        pid = self._next_patch_id
        self.patch_names[pid] = name
        self._next_patch_id += 1
        return pid

    @staticmethod
    def _merge_labeled(base: pv.PolyData, cap: pv.PolyData, pid: int) -> pv.PolyData:
        """Merge cap into base keeping PATCH_ID attached to the right faces.

        Never assumes anything about merged cell order (pyvista's merge appends
        (other, base) — the opposite of intuition). Instead both inputs get
        type-identical int64 label arrays BEFORE merging, which VTK then
        preserves per-cell. (The array was only ever dropped when types differed,
        e.g. the plane-clipper's output vs a fresh numpy array.)"""
        base.cell_data[PATCH_ID] = np.asarray(base.cell_data[PATCH_ID], dtype=np.int64)
        cap.cell_data[PATCH_ID] = np.full(cap.n_cells, pid, dtype=np.int64)
        merged = base.merge(cap, merge_points=True)
        if PATCH_ID not in merged.cell_data or len(merged.cell_data[PATCH_ID]) != merged.n_cells:
            # Order-independent fallback: map by nearest cell centroid from a
            # plain append union, which preserves both label arrays.
            union = base.merge(cap, merge_points=False)
            merged = STLClipperEngine._carry_labels(merged, union)
        return merged

    def clip_and_name(self, name: str, origin, normal, box_planes_data: list = None):
        """Trim current_mesh by a plane (optionally box-scoped), cap the opening,
        and label the cap as a new named patch — one compounding step.
        Returns the new current_mesh, or None on a no-op (miss / empty cap)."""
        if self.current_mesh is None:
            return None
        self._ensure_labels()
        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)
        try:
            if box_planes_data:
                trimmed = self.clip_with_box(self.current_mesh, box_planes_data, origin, normal)
            else:
                trimmed = self.clip_with_plane(self.current_mesh, origin, normal)
        except RuntimeError:
            return None
        trimmed = self._carry_labels(trimmed, self.current_mesh)
        cap = self._generate_cap(self.current_mesh, origin, normal,
                                 box_planes_data if box_planes_data else None)
        if cap is None or cap.n_cells == 0:
            return None
        cap = cap.triangulate()
        # Weld guarantee: the cap rim comes from vtkCutter (slice) but the
        # trimmed wall rim from vtkClipPolyData — the same plane/edge
        # intersections through different float paths. On some geometries the
        # coordinates differ in the last bits, so merge-by-coordinate never
        # welds and the "capped" clip is silently open. Snap cap vertices onto
        # the trimmed wall's boundary vertices within a tiny tolerance (far
        # below rim spacing) so the weld is exact by construction.
        wall_rim = trimmed.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        if wall_rim.n_points and cap.n_points:
            b = np.asarray(trimmed.bounds, dtype=float)
            diag = float(np.linalg.norm(b[1::2] - b[0::2]))
            tol = 1e-6 * diag if diag > 0 else 1e-6
            from scipy.spatial import cKDTree
            d, j = cKDTree(np.asarray(wall_rim.points)).query(
                np.asarray(cap.points))
            hit = d < tol
            if hit.any():
                pts = np.asarray(cap.points).copy()
                pts[hit] = np.asarray(wall_rim.points)[j[hit]]
                cap.points = pts
        cap = self._orient_cap_to_base(cap, trimmed)
        self._push_history()
        pid = self._new_patch_id(name)
        mag = float(np.linalg.norm(normal))
        if mag > 0:
            # Outward-pointing (out of the kept domain), matching the old -clip.normal
            self._patch_normals[pid] = tuple(-normal / mag)
        # Re-triangulate: plane/box clipping emits quads where triangles were cut,
        # which would break the all-triangle current_mesh invariant that
        # check_normals / flood / grow / sharp-select fast paths rely on.
        # vtkTriangleFilter carries cell data, so each sub-triangle keeps its label.
        self.original_mesh = self._merge_labeled(trimmed, cap, pid).triangulate()
        return self.current_mesh

    def rename_patch(self, pid: int, new_name: str) -> bool:
        if pid in self.patch_names:
            self._push_history()                 # renames are undoable like any op
            self.patch_names[pid] = new_name
            return True
        return False

    def remove_patch(self, pid: int) -> bool:
        """Un-name a patch: relabel its faces back to wall (0). Geometry is kept
        (this is the single-mesh model — use undo to restore geometry)."""
        if self.current_mesh is None or pid == 0 or pid not in self.patch_names:
            return False
        self._ensure_labels()
        self._push_history()
        ids = np.asarray(self.current_mesh.cell_data[PATCH_ID]).copy()
        ids[ids == pid] = 0
        self.current_mesh.cell_data[PATCH_ID] = ids
        self.patch_names.pop(pid, None)
        self._patch_normals.pop(pid, None)
        return True

    def split_patch(self, pid: int):
        """Split a named patch into its disconnected face components (e.g. a clip
        cap across a bifurcation). Each component becomes its own patch named
        ``{name}_1..N``; the parent name is retired; children inherit the parent's
        outward normal. Relabel-only (no geometry change), one undo step.
        Returns the list of new patch ids, or None if the patch has fewer than
        two components (or pid is invalid)."""
        if self.current_mesh is None or pid == 0 or pid not in self.patch_names:
            return None
        self._ensure_labels()
        ids = np.asarray(self.current_mesh.cell_data[PATCH_ID])
        cell_idx = np.nonzero(ids == pid)[0]
        if len(cell_idx) == 0:
            return None
        sub = self.current_mesh.extract_cells(cell_idx)
        conn = sub.connectivity('all')
        rid = np.asarray(conn.cell_data['RegionId'])
        regions = np.unique(rid)
        if len(regions) < 2:
            return None
        self._push_history()
        name = self.patch_names[pid]
        normal = self._patch_normals.get(pid)
        new_ids = ids.copy()
        new_pids = []
        for k, r in enumerate(regions, start=1):
            child = self._new_patch_id(f"{name}_{k}")
            new_ids[cell_idx[rid == r]] = child
            if normal is not None:
                self._patch_normals[child] = normal
            new_pids.append(child)
        self.current_mesh.cell_data[PATCH_ID] = new_ids
        self.patch_names.pop(pid, None)
        self._patch_normals.pop(pid, None)
        return new_pids

    def faces_on_edges(self, edges) -> list:
        """Cell ids of current_mesh faces that use both endpoints of any edge in
        `edges` (a PolyData of line cells, e.g. one non-manifold group)."""
        m = self.current_mesh
        if m is None or edges is None or edges.n_cells == 0:
            return []
        b = np.asarray(m.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        tol = 1e-6 * diag if diag > 0 else 1e-6
        pts = np.asarray(edges.points)
        pid = np.empty(edges.n_points, dtype=np.int64)
        ok = np.zeros(edges.n_points, dtype=bool)
        for i in range(edges.n_points):
            j = int(m.find_closest_point(pts[i]))
            pid[i] = j
            ok[i] = float(np.linalg.norm(pts[i] - m.points[j])) <= tol
        out = set()
        for seg in edges.lines.reshape(-1, 3):
            i0, i1 = int(seg[1]), int(seg[2])
            if ok[i0] and ok[i1]:
                c0 = {int(c) for c in m.point_cell_ids(int(pid[i0]))}
                c1 = {int(c) for c in m.point_cell_ids(int(pid[i1]))}
                out |= (c0 & c1)
        return sorted(out)

    @staticmethod
    def _retriangulate_planar_cap(cap, rim):
        """Rebuild a PLANAR cap's triangulation with near-isotropic triangles.

        vtkDelaunay2D's edge constraint is unreliable on real rims (concave or
        multi-disc cross-sections make it fill the convex hull instead). This
        does it the robust way: unconstrained planar Delaunay over the rim
        points PLUS interior seed points laid out on a hex grid at rim-edge
        spacing, then keep only triangles whose center lies ON the original
        cap — concavities and separate discs filter themselves out, and the
        rim points are reused verbatim so the merge glues exactly.

        Returns the new cap PolyData, or None when it declines (non-planar cap,
        degenerate rim, or Delaunay dropped points)."""
        cap_pts = np.asarray(cap.points, dtype=float)
        centroid = cap_pts.mean(axis=0)
        cov = np.cov((cap_pts - centroid).T)
        w, v = np.linalg.eigh(cov)
        n, u1, u2 = v[:, 0], v[:, 1], v[:, 2]            # normal + in-plane basis
        b = np.asarray(cap.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        if diag <= 0:
            return None
        thickness = float(np.ptp((cap_pts - centroid) @ n))
        if thickness > 1e-3 * diag:                      # only flat caps
            return None

        rim_pts = np.asarray(rim.points, dtype=float)
        seg = rim.lines.reshape(-1, 3)[:, 1:]
        spacing = float(np.median(np.linalg.norm(
            rim_pts[seg[:, 0]] - rim_pts[seg[:, 1]], axis=1)))
        if spacing <= 0:
            return None
        # Bound the interior grid: a very fine rim on a large cap would demand
        # millions of candidates (observed: a 10-minute hang). Coarsening the
        # spacing keeps the work bounded at a slightly coarser interior.
        min_spacing = diag / 300.0
        if spacing < min_spacing:
            spacing = min_spacing

        # Hex-grid interior candidates over the cap's in-plane bounding box.
        rp2 = np.column_stack([(rim_pts - centroid) @ u1, (rim_pts - centroid) @ u2])
        lo, hi = rp2.min(axis=0), rp2.max(axis=0)
        xs = np.arange(lo[0], hi[0] + spacing, spacing)
        row_h = spacing * np.sqrt(3.0) / 2.0
        ys = np.arange(lo[1], hi[1] + row_h, row_h)
        gx, gy = np.meshgrid(xs, ys)
        gx[1::2, :] += spacing / 2.0                     # hex offset rows
        cand2 = np.column_stack([gx.ravel(), gy.ravel()])
        cand3 = centroid + cand2[:, :1] * u1 + cand2[:, 1:2] * u2
        # Keep candidates ON the cap (inside test against the actual surface)
        # and clear of the rim, so boundary triangles stay well-shaped.
        from scipy.spatial import cKDTree
        if len(cand3):
            _, cp = cap.find_closest_cell(cand3, return_closest_point=True)
            on_cap = np.linalg.norm(cand3 - cp, axis=1) < 1e-6 * diag
            # Clearance > 0.707*edge keeps every interior point outside the
            # diametral circle of any rim segment, so the Delaunay always
            # contains the rim edges (no coverage gaps in narrow notches).
            clear = cKDTree(rim_pts).query(cand3)[0] > 0.75 * spacing
            interior = cand3[on_cap & clear]
        else:
            interior = np.empty((0, 3))

        pts3 = np.vstack([rim_pts, interior])
        flat = np.column_stack([(pts3 - centroid) @ u1, (pts3 - centroid) @ u2,
                                np.zeros(len(pts3))])
        try:
            tri2d = pv.PolyData(flat).delaunay_2d()
        except Exception:
            return None
        if tri2d.n_points != len(pts3):                  # Delaunay dropped points
            return None
        faces = tri2d.faces.reshape(-1, 4)
        if not bool((faces[:, 0] == 3).all()):
            return None
        # Back to 3D with the ORIGINAL coordinates (rim points verbatim), then
        # drop hull triangles: keep only those whose center is on the cap.
        new_cap = pv.PolyData(pts3, faces.ravel())
        centers = new_cap.cell_centers().points
        _, cp = cap.find_closest_cell(centers, return_closest_point=True)
        keep = np.nonzero(np.linalg.norm(centers - cp, axis=1) < 1e-6 * diag)[0]
        if len(keep) == 0:
            return None
        return new_cap.extract_cells(keep).extract_surface().triangulate()

    def remesh_patch(self, pid: int):
        """Replace a cap patch's triangulation with a near-isotropic planar one
        built on its rim (well-shaped triangles instead of a sliver fan).
        Handles concave and multi-disc cross-sections (a clip plane through
        several vessel limbs). Relabels with the same patch id/name; one undo
        step. Returns the new current_mesh, or None if the patch is missing,
        non-planar, or retriangulation fails validation."""
        if self.current_mesh is None or pid == 0 or pid not in self.patch_names:
            return None
        self._ensure_labels()
        ids = np.asarray(self.current_mesh.cell_data[PATCH_ID])
        cell_idx = np.nonzero(ids == pid)[0]
        if len(cell_idx) == 0:
            return None
        cap = self.current_mesh.extract_cells(cell_idx).extract_surface()
        rim = cap.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        if rim.n_cells == 0:
            return None
        new_cap = self._retriangulate_planar_cap(cap, rim)
        if new_cap is None or new_cap.n_cells == 0:
            return None
        # Retriangulation must cover EXACTLY the same surface — same rim, same
        # area. If the Delaunay constraint failed on a concave rim, VTK falls
        # back to the convex hull, which would ADD surface beyond the original
        # boundary. Reject rather than extend the mesh.
        old_area = float(cap.area)
        if old_area > 0 and abs(float(new_cap.area) - old_area) > 0.01 * old_area:
            return None
        # Signatures of loops that already exist (pre-existing openings must
        # not be touched; only gaps CREATED by the retriangulation get sealed).
        b = np.asarray(self.current_mesh.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        tol = 1e-6 * diag if diag > 0 else 1e-6
        before = [(g.n_cells, np.asarray(g.points).mean(axis=0))
                  for g in self.detect_open_profiles()]
        # Assemble the full result FIRST; commit only if it validates.
        keep = np.nonzero(ids != pid)[0]
        base = self.current_mesh.extract_cells(keep).extract_surface()
        base = self._carry_labels(base, self.current_mesh)
        new_cap = self._orient_cap_to_base(new_cap, base)
        result = self._merge_labeled(base, new_cap, pid)
        # A rim segment can still be missed in a degenerate spot, leaving a
        # thin gap the area gate cannot see. Seal CLOSED new loops with a fan
        # (labeled as this patch). A non-closed chain means overlapping or
        # non-manifold geometry would result — decline instead of committing.
        edges = result.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        for loop in self._split_edge_groups(edges):
            cl = np.asarray(loop.points).mean(axis=0)
            if any(nc == loop.n_cells and float(np.linalg.norm(cl - bc)) <= tol
                   for nc, bc in before):
                continue                                 # pre-existing opening
            seg = loop.lines.reshape(-1, 3)[:, 1:] if loop.lines.size else None
            closed = (seg is not None
                      and np.all(np.bincount(seg.ravel()) == 2))
            if not closed:
                return None                              # would corrupt — decline
            gap = self._fan_fill(loop)
            if gap is None:
                return None
            result = self._merge_labeled(result, gap, pid)
        self._push_history()
        self.original_mesh = result
        return self.current_mesh

    def patches_by_id(self) -> dict:
        """{patch_id: PolyData} grouping current_mesh faces by PATCH_ID."""
        out = {}
        if self.current_mesh is None:
            return out
        self._ensure_labels()
        ids = np.asarray(self.current_mesh.cell_data[PATCH_ID])
        for pid in np.unique(ids):
            cells = np.nonzero(ids == pid)[0]
            out[int(pid)] = self.current_mesh.extract_cells(cells).extract_surface()
        return out

    def patch_name_for(self, pid: int) -> str:
        return "wall" if pid == 0 else self.patch_names.get(pid, f"patch_{pid}")

    def trim_by_screen_polygon(self, polygon_xy, view_matrix, viewport):
        """Permanently delete cells of original_mesh whose centroid projects
        inside the freehand outline polygon_xy (through-model). Mutates the base
        mesh and re-applies clips. Returns the new current_mesh, or None on a no-op.

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
        centers = self.original_mesh.cell_centers().points          # (N, 3)
        disp_x, disp_y = _project_to_display(centers, view_matrix, viewport)
        inside = _points_in_polygon(disp_x, disp_y, poly)
        n_inside = int(inside.sum())
        if n_inside == 0:
            return None
        if n_inside == centers.shape[0]:
            raise ValueError("Trim would delete the entire mesh")
        self._push_history()
        keep_ids = np.where(~inside)[0]
        self.original_mesh = self.original_mesh.extract_cells(keep_ids).extract_surface()
        return self.current_mesh

    def select_cells_in_polygon(self, polygon_xy, view_matrix, viewport,
                                view_direction, front_only=True):
        """Cell ids the screen polygon covers — a cell is selected if ANY of its
        vertices projects inside polygon_xy, so triangles partially under the lasso
        are captured (not just those whose centroid is inside). If front_only, also
        require the cell normal to face the camera. view_direction is the world
        vector the camera looks along (into the screen)."""
        if self.original_mesh is None:
            return []
        poly = np.asarray(polygon_xy, dtype=float)
        if poly.shape[0] < 3:
            return []
        mesh = self.original_mesh
        faces = mesh.faces
        if faces.size == 4 * mesh.n_cells:                 # all-triangle fast path
            tri = faces.reshape(-1, 4)[:, 1:]
            disp_x, disp_y = _project_to_display(mesh.points, view_matrix, viewport)
            pt_inside = _points_in_polygon(disp_x, disp_y, poly)
            inside = pt_inside[tri].any(axis=1)
        else:                                              # non-triangle: centroid test
            centers = mesh.cell_centers().points
            disp_x, disp_y = _project_to_display(centers, view_matrix, viewport)
            inside = _points_in_polygon(disp_x, disp_y, poly)
        if front_only:
            vd = np.asarray(view_direction, dtype=float)
            facing = (np.asarray(mesh.cell_normals) @ vd) < 0.0
            inside = inside & facing
        return np.where(inside)[0].tolist()

    def _ensure_adjacency(self):
        """Build (once per mesh) CSR adjacency for fast neighbor lookups so grow and
        smooth avoid per-element pyvista queries (~1 ms each). Rebuilt automatically
        whenever original_mesh is replaced (identity check). Falls back to no fast
        path (`_adj_ok = False`) on non-triangle meshes."""
        mesh = self.original_mesh
        if mesh is None:
            self._adj_for_mesh = None
            self._adj_ok = False
            return
        if self._adj_for_mesh is mesh:
            return
        faces = mesh.faces
        if faces.size != 4 * mesh.n_cells:
            self._adj_for_mesh = mesh
            self._adj_ok = False
            return
        faces = faces.reshape(-1, 4)
        if not bool((faces[:, 0] == 3).all()):
            self._adj_for_mesh = mesh
            self._adj_ok = False
            return
        tri = faces[:, 1:].astype(np.int64)
        n_cells, n_points = mesh.n_cells, mesh.n_points
        # point -> incident cells (CSR)
        cell_rep = np.repeat(np.arange(n_cells, dtype=np.int64), 3)
        pt_flat = tri.ravel()
        order = np.argsort(pt_flat, kind="stable")
        cells_by_point = cell_rep[order]
        counts = np.bincount(pt_flat, minlength=n_points)
        cell_starts = np.zeros(n_points + 1, dtype=np.int64)
        np.cumsum(counts, out=cell_starts[1:])
        # point -> edge-neighbor points (CSR; unique both-direction edges so a point
        # shared by two triangles is not listed twice — keeps the Laplacian mean
        # identical to pyvista's point_neighbors)
        edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        edges = np.vstack([edges, edges[:, ::-1]])
        edges = np.unique(edges, axis=0)
        nbr_by_point = edges[:, 1].copy()
        ecounts = np.bincount(edges[:, 0], minlength=n_points)
        nbr_starts = np.zeros(n_points + 1, dtype=np.int64)
        np.cumsum(ecounts, out=nbr_starts[1:])
        self._adj_tri = tri
        self._adj_cells_by_point = cells_by_point
        self._adj_cell_starts = cell_starts
        self._adj_nbr_by_point = nbr_by_point
        self._adj_nbr_starts = nbr_starts
        self._adj_for_mesh = mesh
        self._adj_ok = True

    def grow_cells(self, cell_ids, rings=1):
        """Dilate a cell selection by `rings` point-connected face neighbors."""
        if self.original_mesh is None:
            return sorted({int(c) for c in cell_ids})
        self._ensure_adjacency()
        n_cells = self.original_mesh.n_cells
        current = {int(c) for c in cell_ids if 0 <= int(c) < n_cells}
        if not self._adj_ok:
            mesh = self.original_mesh
            for _ in range(int(rings)):
                nxt = set(current)
                for cid in current:
                    nxt.update(int(c) for c in mesh.cell_neighbors(cid, connections="points"))
                current = nxt
            return sorted(current)
        tri = self._adj_tri
        cells_by_point = self._adj_cells_by_point
        cell_starts = self._adj_cell_starts
        for _ in range(int(rings)):
            if not current:
                break
            cur = np.fromiter(current, dtype=np.int64, count=len(current))
            pts = np.unique(tri[cur])
            chunks = [cells_by_point[cell_starts[p]:cell_starts[p + 1]] for p in pts]
            if chunks:
                current.update(int(c) for c in np.unique(np.concatenate(chunks)))
        return sorted(current)

    def smooth_cells(self, cell_ids, iterations=5, relaxation=0.5):
        """Constrained Laplacian smoothing of the points of the selected cells.
        Points not in any selected cell stay fixed (the patch blends into the rest).
        Mutates original_mesh, pushes undo history, returns the new current_mesh."""
        if self.original_mesh is None:
            return None
        mesh = self.original_mesh
        ids = [int(c) for c in cell_ids if 0 <= int(c) < mesh.n_cells]
        if not ids:
            return None
        self._ensure_adjacency()
        if self._adj_ok:
            movable = np.unique(self._adj_tri[np.asarray(ids, dtype=np.int64)])
            nbr = self._adj_nbr_by_point
            starts = self._adj_nbr_starts
            deg = (starts[movable + 1] - starts[movable]).astype(np.int64)
            keep = deg > 0
            mov = movable[keep]
            dg = deg[keep]
            if mov.size == 0:
                return None
            # reshape the ragged neighbor lists into a fixed (M, max_deg) index
            # matrix + validity mask, vectorized via the cumsum "ranges" trick,
            # so every smoothing iteration is a single array op.
            total = int(dg.sum())
            n_movable, max_deg = mov.size, int(dg.max())
            row = np.repeat(np.arange(n_movable), dg)
            off = np.arange(total) - np.repeat(np.cumsum(dg) - dg, dg)
            pos = np.repeat(starts[mov], dg) + off
            idx = np.zeros((n_movable, max_deg), dtype=np.int64)
            idx[row, off] = nbr[pos]
            valid = np.zeros((n_movable, max_deg), dtype=bool)
            valid[row, off] = True
            inv_deg = (1.0 / dg)[:, None]
            pts = mesh.points.copy()
            self._push_history()
            for _ in range(int(iterations)):
                gathered = pts[idx]
                gathered[~valid] = 0.0
                mean_nb = gathered.sum(axis=1) * inv_deg
                pts[mov] = (1.0 - relaxation) * pts[mov] + relaxation * mean_nb
        else:
            movable = set()
            for cid in ids:
                movable.update(int(p) for p in mesh.get_cell(cid).point_ids)
            movable = sorted(movable)
            if not movable:
                return None
            neighbors = {p: np.asarray(list(mesh.point_neighbors(p)), dtype=np.int64) for p in movable}
            pts = mesh.points.copy()
            self._push_history()
            for _ in range(int(iterations)):
                new_pts = pts.copy()
                for p in movable:
                    nb = neighbors[p]
                    if len(nb):
                        new_pts[p] = (1.0 - relaxation) * pts[p] + relaxation * pts[nb].mean(axis=0)
                pts = new_pts
        smoothed = mesh.copy()
        smoothed.points = pts
        self.original_mesh = smoothed
        return self.current_mesh

    def _barrier_edge_keys(self, mesh):
        """Edge keys (min*n_points + max) for every feature-curve segment, mapped to
        current mesh vertices. A segment contributes a wall only if BOTH endpoints
        map to a vertex within 1e-6 * bbox-diagonal, so feature-curve points whose
        region was deleted drop out instead of snapping to a wrong vertex."""
        n_points = mesh.n_points
        b = np.asarray(mesh.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        tol = 1e-6 * diag if diag > 0 else 1e-6
        barrier = set()
        for curve in self._feature_curves:
            if curve is None or curve.n_cells == 0:
                continue
            pts = np.asarray(curve.points)
            pid = np.empty(curve.n_points, dtype=np.int64)
            ok = np.zeros(curve.n_points, dtype=bool)
            for i in range(curve.n_points):
                j = int(mesh.find_closest_point(pts[i]))
                pid[i] = j
                ok[i] = float(np.linalg.norm(pts[i] - mesh.points[j])) <= tol
            for seg in curve.lines.reshape(-1, 3):
                i0, i1 = int(seg[1]), int(seg[2])
                if ok[i0] and ok[i1]:
                    u, v = int(pid[i0]), int(pid[i1])
                    if u != v:
                        if u > v:
                            u, v = v, u
                        barrier.add(u * n_points + v)
        return barrier

    def _ensure_flood_adjacency(self):
        """Build (once per mesh identity + feature-curve count) cell edge-adjacency
        that excludes feature-curve edges, so a flood never crosses a cut. Sets
        _flood_ok = False on a non-triangle mesh (no fast path). Mirrors the
        _ensure_adjacency cache pattern."""
        mesh = self.original_mesh
        if mesh is None:
            self._flood_adj_for_mesh = None
            self._flood_adj_n_curves = -1
            self._flood_adj = None
            self._flood_ok = False
            return
        if (self._flood_adj_for_mesh is mesh
                and self._flood_adj_n_curves == len(self._feature_curves)):
            return
        self._flood_adj_for_mesh = mesh
        self._flood_adj_n_curves = len(self._feature_curves)
        faces = mesh.faces
        if faces.size != 4 * mesh.n_cells or not bool(
                (faces.reshape(-1, 4)[:, 0] == 3).all()):
            self._flood_adj = None
            self._flood_ok = False
            return
        n_cells, n_points = mesh.n_cells, mesh.n_points
        tri = faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        barrier = self._barrier_edge_keys(mesh)
        e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        e.sort(axis=1)
        keys = e[:, 0] * n_points + e[:, 1]
        cell_of = np.tile(np.arange(n_cells, dtype=np.int64), 3)   # tile, NOT repeat
        order = np.argsort(keys, kind="stable")
        keys_s = keys[order]
        cells_s = cell_of[order]
        adj = [[] for _ in range(n_cells)]
        n = len(keys_s)
        i = 0
        while i < n:
            j = i
            while j < n and keys_s[j] == keys_s[i]:
                j += 1
            if (j - i) == 2 and int(keys_s[i]) not in barrier:
                c0 = int(cells_s[i])
                c1 = int(cells_s[i + 1])
                adj[c0].append(c1)
                adj[c1].append(c0)
            i = j
        self._flood_adj = adj
        self._flood_ok = True

    def flood_select(self, seed_cell):
        """Connected surface region containing seed_cell, with feature-curve edges as
        walls (the flood never crosses a cut). Returns a sorted list of cell ids; []
        if the seed is out of range or there is no mesh; [seed_cell] on a non-triangle
        mesh."""
        mesh = self.original_mesh
        if mesh is None:
            return []
        seed = int(seed_cell)
        if seed < 0 or seed >= mesh.n_cells:
            return []
        self._ensure_flood_adjacency()
        if not self._flood_ok:
            return [seed]
        adj = self._flood_adj
        seen = {seed}
        stack = [seed]
        while stack:
            c = stack.pop()
            for nb in adj[c]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return sorted(seen)

    def select_to_sharp_edges(self, seed_cells, angle_deg: float = 30.0):
        """Flood-select from seed faces, stopping at sharp edges.

        Grows across edge-adjacent (manifold) neighbors whose dihedral angle —
        the angle between the two faces' normals — stays within `angle_deg`.
        A flat end cap on a curved pipe is bounded by its ~90° rim, so one seed
        face selects exactly the cap. Feature-curve edges are walls too (same
        rule as flood_select), so a cut still bounds the selection even where
        the surface is geometrically smooth. Assumes consistent winding (run
        Fix Normals first if flipped faces stop the flood early).

        Returns a sorted list of cell ids; [] if no mesh/seeds; the valid seeds
        unchanged on a non-triangle mesh."""
        mesh = self.original_mesh
        if mesh is None:
            return []
        n_cells = mesh.n_cells
        seeds = sorted({int(c) for c in seed_cells if 0 <= int(c) < n_cells})
        if not seeds:
            return []
        faces = mesh.faces
        if faces.size != 4 * n_cells or not bool(
                (faces.reshape(-1, 4)[:, 0] == 3).all()):
            return seeds
        tri = faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        n_points = mesh.n_points
        pts = np.asarray(mesh.points, dtype=float)

        # Per-face unit normals (winding order). Degenerate faces -> zero normal,
        # which fails every dihedral test, isolating them — the safe behavior.
        v0, v1, v2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        mag = np.linalg.norm(fn, axis=1)
        fn = fn / np.maximum(mag, 1e-30)[:, None]

        # Manifold edge -> its two incident faces (runs of exactly 2 equal keys).
        e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        e.sort(axis=1)
        keys = e[:, 0] * n_points + e[:, 1]
        cell_of = np.tile(np.arange(n_cells, dtype=np.int64), 3)
        order = np.argsort(keys, kind="stable")
        ks, cs = keys[order], cell_of[order]
        is_start = np.r_[True, ks[1:] != ks[:-1]]
        run_id = np.cumsum(is_start) - 1
        counts = np.bincount(run_id)
        starts = np.nonzero(is_start)[0]
        two = starts[counts == 2]
        a, b, ekey = cs[two], cs[two + 1], ks[two]

        cos_thresh = float(np.cos(np.radians(angle_deg)))
        ok = np.einsum('ij,ij->i', fn[a], fn[b]) >= cos_thresh
        barrier = self._barrier_edge_keys(mesh)
        if barrier:
            ok &= ~np.isin(ekey, np.fromiter(barrier, dtype=np.int64))
        a, b = a[ok], b[ok]

        # CSR adjacency over traversable edges, then BFS from the seeds.
        src = np.concatenate([a, b])
        dst = np.concatenate([b, a])
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        deg = np.bincount(src, minlength=n_cells)
        adj_starts = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(deg, out=adj_starts[1:])
        seen = np.zeros(n_cells, dtype=bool)
        seen[seeds] = True
        stack = list(seeds)
        while stack:
            c = stack.pop()
            for nb in dst[adj_starts[c]:adj_starts[c + 1]]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
        return sorted(int(i) for i in np.nonzero(seen)[0])

    def assign_patch_from_cells(self, cell_ids, name: str):
        """Label the given faces as a new named patch (e.g. an auto-selected end
        cap -> 'outlet1'). Pure relabel — geometry untouched, one undo step.
        Records the selection's area-weighted mean normal as the patch's outward
        normal (used by the OpenFOAM export). Returns the new patch id, or None
        on a no-op (no mesh / empty selection / blank name)."""
        if self.original_mesh is None or not name or not name.strip():
            return None
        n = self.original_mesh.n_cells
        ids = sorted({int(c) for c in cell_ids if 0 <= int(c) < n})
        if not ids:
            return None
        self._ensure_labels()
        self._push_history()
        pid = self._new_patch_id(name.strip())
        labels = np.asarray(self.original_mesh.cell_data[PATCH_ID], dtype=np.int64)
        labels[ids] = pid
        self.original_mesh.cell_data[PATCH_ID] = labels
        # Relabeling can take the last faces of another named patch (e.g. a
        # through-model lasso that swept a hidden cap). A name with zero faces
        # would still be exported into controlDict but never into the STL —
        # a guaranteed solver failure — so purge emptied patches now.
        present = set(int(v) for v in np.unique(labels))
        for old_pid in [p for p in self.patch_names if p != pid and p not in present]:
            del self.patch_names[old_pid]
            self._patch_normals.pop(old_pid, None)
        faces = self.original_mesh.faces
        if faces.size == 4 * n and bool((faces.reshape(-1, 4)[:, 0] == 3).all()):
            tri = faces.reshape(-1, 4)[:, 1:].astype(np.int64)[ids]
            pts = np.asarray(self.original_mesh.points, dtype=float)
            fn = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                          pts[tri[:, 2]] - pts[tri[:, 0]])   # 2*area-weighted
            mean = fn.sum(axis=0)
            mag = float(np.linalg.norm(mean))
            total = float(np.linalg.norm(fn, axis=1).sum())
            # Record only a MEANINGFUL mean direction. Opposing faces (both end
            # caps selected at once) cancel to numerical noise; storing that
            # noise would silently become the inlet velocity direction in 0/U.
            if total > 0 and mag > 1e-6 * total:
                self._patch_normals[pid] = tuple(mean / mag)
        return pid

    def _split_edge_groups(self, edges):
        """Split an edge PolyData into connected groups; one geometry per group
        (line cells preserved) for highlighting. [] if empty/None."""
        if edges is None or edges.n_cells == 0:
            return []
        conn = edges.connectivity('all')
        rid = np.asarray(conn.cell_data['RegionId'])
        return [conn.extract_cells(np.nonzero(rid == r)[0]).extract_surface()
                for r in np.unique(rid)]

    def detect_open_profiles(self):
        """List of boundary-edge loops (open profiles / holes), one pv.PolyData per
        connected loop. [] if watertight or no mesh."""
        m = self.current_mesh if self.current_mesh is not None else self.original_mesh
        if m is None:
            return []
        edges = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                        manifold_edges=False, non_manifold_edges=False)
        return self._split_edge_groups(edges)

    def detect_nonmanifold_edges(self):
        """List of non-manifold edge groups (edges shared by >2 faces), one
        pv.PolyData per connected group. [] if none or no mesh."""
        m = self.current_mesh if self.current_mesh is not None else self.original_mesh
        if m is None:
            return []
        edges = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                        manifold_edges=False, non_manifold_edges=True)
        return self._split_edge_groups(edges)

    def detect_pieces(self):
        """List of connected components; each entry is a sorted list of cell ids into
        current_mesh. [] if no mesh. Single watertight body -> one entry."""
        m = self.current_mesh if self.current_mesh is not None else self.original_mesh
        if m is None:
            return []
        conn = m.connectivity('all')
        rid = np.asarray(conn.cell_data['RegionId'])
        return [sorted(int(c) for c in np.nonzero(rid == r)[0]) for r in np.unique(rid)]

    def named_patches(self):
        """(patch_id, name, face_count) for each named patch present on current_mesh."""
        if self.current_mesh is None or PATCH_ID not in self.current_mesh.cell_data:
            return []
        ids = np.asarray(self.current_mesh.cell_data[PATCH_ID])
        return [(pid, name, int(np.count_nonzero(ids == pid)))
                for pid, name in sorted(self.patch_names.items())]

    @staticmethod
    def _fan_fill(loop):
        """Seal a boundary loop with a triangle fan from its centroid.

        Robust for small non-planar rims (e.g. around a pinch) where contour
        triangulation covers the loop only partially: every rim edge gets
        exactly one fan triangle, so the loop seals by construction. Quality is
        lower than a Delaunay cap (use for small defect rims, then Smooth)."""
        if loop is None or loop.n_cells == 0:
            return None
        lines = loop.lines
        if lines.size != 3 * loop.n_cells:               # not simple 2-pt segments
            return None
        pts = np.asarray(loop.points, dtype=float)
        seg = lines.reshape(-1, 3)[:, 1:]
        ci = len(pts)                                    # centroid index
        new_pts = np.vstack([pts, pts.mean(axis=0)])
        faces = np.column_stack([
            np.full(len(seg), 3), seg[:, 0], seg[:, 1],
            np.full(len(seg), ci)]).astype(np.int64).ravel()
        return pv.PolyData(new_pts, faces)

    @staticmethod
    def _triangulate_loop(profile_edges):
        """Triangulate a boundary-edge loop into a cap PolyData (contour
        triangulation, Delaunay fallback). None if it cannot be triangulated."""
        if profile_edges is None or profile_edges.n_cells == 0:
            return None
        strip = vtk.vtkStripper()
        strip.SetInputData(profile_edges)
        strip.Update()
        tri = vtk.vtkContourTriangulator()
        tri.SetInputData(strip.GetOutput())
        tri.Update()
        cap = pv.wrap(tri.GetOutput())
        if cap is None or cap.n_cells == 0:
            cap = pv.PolyData(profile_edges.points).delaunay_2d()     # fallback
        if cap is None or cap.n_cells == 0:
            return None
        return cap.triangulate()

    @staticmethod
    def _orient_cap_to_base(cap, base):
        """Flip the cap's winding if it disagrees with the surrounding surface.

        The base's boundary directed edges (a->b, as wound in their single
        incident face) define the orientation a conforming cap must have: the
        cap face sharing edge (a,b) must traverse b->a. Cap and base share rim
        coordinates exactly (loop points verbatim, or snap-welded), so edges
        are matched by coordinates. Majority vote over all matched rim edges;
        no matches leaves the cap unchanged. Returns the (possibly flipped)
        cap. Fixes caps being merged wound INTO the domain, which broke the
        normals health check after every clip/fill and silently reversed the
        inlet velocity derived from fill-created patches."""
        if cap is None or cap.n_cells == 0 or base is None or base.n_cells == 0:
            return cap
        bf = base.faces
        if bf.size != 4 * base.n_cells:
            # Clip output still holds quads at this point; triangulation
            # preserves winding, so the boundary directed edges are unchanged.
            base = base.triangulate()
            bf = base.faces
        cf = cap.faces
        if (bf.size != 4 * base.n_cells or cf.size != 4 * cap.n_cells):
            return cap
        btri = bf.reshape(-1, 4)[:, 1:].astype(np.int64)
        n_points = base.n_points
        de = np.vstack([btri[:, [0, 1]], btri[:, [1, 2]], btri[:, [2, 0]]])
        und = de.min(axis=1) * n_points + de.max(axis=1)
        uu, uc = np.unique(und, return_counts=True)
        boundary_und = set(uu[uc == 1].tolist())
        bpts = np.asarray(base.points, dtype=float)
        bdir = set()
        for (a, b), u in zip(de, und):
            if int(u) in boundary_und:
                bdir.add((bpts[a].tobytes(), bpts[b].tobytes()))
        if not bdir:
            return cap
        ctri = cf.reshape(-1, 4)[:, 1:].astype(np.int64)
        cpts = np.asarray(cap.points, dtype=float)
        ckeys = [cpts[i].tobytes() for i in range(cap.n_points)]
        same = opposite = 0
        for t in ctri:
            for i in range(3):
                u, v = ckeys[t[i]], ckeys[t[(i + 1) % 3]]
                if (u, v) in bdir:
                    same += 1          # cap traverses a->b like the base: wrong
                elif (v, u) in bdir:
                    opposite += 1      # cap traverses b->a: conforming
        if same > opposite:
            flipped = cf.reshape(-1, 4).copy()
            flipped[:, 1:] = flipped[:, 1:][:, ::-1]
            out = pv.PolyData(cpts, flipped.ravel())
            for k in cap.cell_data:
                out.cell_data[k] = np.asarray(cap.cell_data[k])
            return out
        return cap

    def fill_profile(self, profile_edges, name=None):
        """Triangulate an open profile's boundary loop into a cap and merge it
        into current_mesh (closing the hole).

        If `name` is given the cap becomes a new named patch; if `name` is None
        (or blank) the cap merges into the wall (patch_id 0) — a quick fill that
        needs no name. Returns the new current_mesh, or None if the edges are
        empty or cannot be triangulated."""
        if self.current_mesh is None or profile_edges is None or profile_edges.n_cells == 0:
            return None
        cap = self._triangulate_loop(profile_edges)
        if cap is None:
            return None
        cap = self._orient_cap_to_base(cap, self.current_mesh)
        self._ensure_labels()
        self._push_history()
        pid = self._new_patch_id(name) if name and name.strip() else 0
        if pid != 0:
            # Record the outward normal (cap is now wound like the surrounding
            # surface) so the OpenFOAM export derives the inlet velocity from a
            # real direction instead of the previously arbitrary cap winding.
            tri = cap.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
            pts = np.asarray(cap.points, dtype=float)
            fn = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                          pts[tri[:, 2]] - pts[tri[:, 0]])
            mean = fn.sum(axis=0)
            mag = float(np.linalg.norm(mean))
            total = float(np.linalg.norm(fn, axis=1).sum())
            if total > 0 and mag > 1e-6 * total:
                self._patch_normals[pid] = tuple(mean / mag)
        self.original_mesh = self._merge_labeled(self.current_mesh, cap, pid)
        return self.current_mesh

    def extrude_profile(self, profile_edges, length, direction=None):
        """Flow extension: extrude an open rim straight along the cut-plane
        normal so the vessel continues naturally (vmtk's flow-extension idea).

        direction defaults to the rim's best-fit plane normal (for a flat cut
        this IS the cut normal), signed to point away from the existing
        surface. The rim vertices are copied outward in rings (~rim edge
        length apart, so side triangles stay well-shaped); side walls are
        wound to match the surrounding surface, so winding stays consistent
        without a global re-wind. The far end becomes a NEW open profile —
        Fill it as a named patch to cap the extension. One undo step.

        Returns a status message, or None on a no-op (no mesh / bad loop /
        non-positive length / rim not found on the surface)."""
        mesh = self.current_mesh
        if (mesh is None or profile_edges is None or profile_edges.n_cells == 0
                or not np.isfinite(length) or length <= 0):
            return None
        faces = mesh.faces
        if faces.size != 4 * mesh.n_cells or not bool(
                (faces.reshape(-1, 4)[:, 0] == 3).all()):
            return None
        lines = profile_edges.lines
        if lines.size != 3 * profile_edges.n_cells:
            return None

        # Map rim points to mesh vertex ids (same tolerance rule as faces_on_edges).
        b = np.asarray(mesh.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        tol = 1e-6 * diag if diag > 0 else 1e-6
        lpts = np.asarray(profile_edges.points)
        pid = np.empty(profile_edges.n_points, dtype=np.int64)
        ok = np.zeros(profile_edges.n_points, dtype=bool)
        for i in range(profile_edges.n_points):
            j = int(mesh.find_closest_point(lpts[i]))
            pid[i] = j
            ok[i] = float(np.linalg.norm(lpts[i] - mesh.points[j])) <= tol
        seg_keys = set()
        n_points = mesh.n_points
        for seg in lines.reshape(-1, 3):
            i0, i1 = int(seg[1]), int(seg[2])
            if ok[i0] and ok[i1]:
                a_, b_ = int(pid[i0]), int(pid[i1])
                seg_keys.add(min(a_, b_) * n_points + max(a_, b_))
        if not seg_keys:
            return None

        # Directed boundary edges of THIS rim, as wound in their one incident
        # face (a->b). The extension face sharing edge (a,b) must traverse b->a,
        # which keeps the whole extension consistent with the surface winding.
        tri = faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        de = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        und = de.min(axis=1) * n_points + de.max(axis=1)
        uu, uc = np.unique(und, return_counts=True)
        boundary_und = set(uu[uc == 1].tolist())
        rim_dir = [(int(a_), int(b_)) for (a_, b_), u in zip(de, und)
                   if int(u) in boundary_und and int(u) in seg_keys]
        if not rim_dir:
            return None

        # Extrusion direction: best-fit plane normal of the rim (PCA), signed to
        # point AWAY from the faces attached to the rim (out of the vessel).
        rim_ids = sorted({v for e in rim_dir for v in e})
        rim_pts = np.asarray(mesh.points, dtype=float)[rim_ids]
        centroid = rim_pts.mean(axis=0)
        if direction is not None:
            d = np.asarray(direction, dtype=float)
        else:
            cov = np.cov((rim_pts - centroid).T)
            w, v = np.linalg.eigh(cov)
            d = v[:, 0]                                   # smallest-variance axis
            attached = self.faces_on_edges(profile_edges)
            if attached:
                inward = mesh.cell_centers().points[attached].mean(axis=0)
                if float(np.dot(d, centroid - inward)) < 0:
                    d = -d
        mag = float(np.linalg.norm(d))
        if mag == 0:
            return None
        d = d / mag

        # Ring spacing ~ rim edge length keeps side triangles near-isotropic.
        seg_len = [float(np.linalg.norm(mesh.points[a_] - mesh.points[b_]))
                   for a_, b_ in rim_dir]
        med = float(np.median(seg_len))
        n_rings = int(np.clip(round(length / med) if med > 0 else 1, 1, 100))

        # The extension inherits each rim face's winding. On an already
        # inconsistent surface every ring replicates the conflict, so warn the
        # user to repair first (checked BEFORE mutating).
        pre_flips = self._count_winding_flips(mesh) or 0

        local = {v: i for i, v in enumerate(rim_ids)}
        nv = len(rim_ids)
        base = np.asarray(mesh.points, dtype=float)[rim_ids]
        rings = [base + d * (length * k / n_rings) for k in range(n_rings + 1)]
        pts = np.vstack(rings)
        tris = []
        for k in range(n_rings):
            bot, top = k * nv, (k + 1) * nv
            for a_, b_ in rim_dir:
                la, lb = local[a_] , local[b_]
                tris.append([3, lb + bot, la + bot, la + top])   # (b, a, a')
                tris.append([3, lb + bot, la + top, lb + top])   # (b, a', b')
        ext = pv.PolyData(pts, np.asarray(tris, dtype=np.int64).ravel())

        self._ensure_labels()
        self._push_history()
        self.original_mesh = self._merge_labeled(self.current_mesh, ext, 0)
        msg = (f"Extended {len(rim_dir)}-edge rim by {length:g} along "
               f"({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}) — {n_rings} ring(s), "
               f"{2 * n_rings * len(rim_dir)} faces added. The new end is an "
               f"open profile: Fill it as a named patch to cap it.")
        if pre_flips:
            msg += (f" Note: the surface already had {pre_flips} flipped edge(s) "
                    f"— consider Ctrl+Z, Fix Normals, then extrude again.")
        return msg

    def unfilled_open_profiles(self):
        """Open profiles not yet filled. Filling merges the cap into current_mesh
        (the hole closes), so every remaining open profile is by definition unfilled."""
        return self.detect_open_profiles()

    def drawable_feature_curves(self):
        """Feature curves that still lie on the CLOSED interior of the surface, i.e.
        whose edges are still shared by two faces. Once a cut's region is deleted its
        edges become open-boundary (it is now an open profile) or vanish entirely, so
        it is excluded here — no stale/floating cyan feature curve is drawn."""
        m = self.original_mesh
        if m is None or not self._feature_curves:
            return []
        faces = m.faces
        if faces.size != 4 * m.n_cells or not bool((faces.reshape(-1, 4)[:, 0] == 3).all()):
            return list(self._feature_curves)          # non-triangle: can't check, draw as-is
        n_points = m.n_points
        tri = faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        e.sort(axis=1)
        keys = e[:, 0] * n_points + e[:, 1]
        uk, cnt = np.unique(keys, return_counts=True)
        share = dict(zip(uk.tolist(), cnt.tolist()))
        b = np.asarray(m.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        tol = 1e-6 * diag if diag > 0 else 1e-6
        out = []
        for cur in self._feature_curves:
            if cur is None or cur.n_cells == 0:
                continue
            pts = np.asarray(cur.points)
            pid = np.empty(cur.n_points, dtype=np.int64)
            ok = np.zeros(cur.n_points, dtype=bool)
            for i in range(cur.n_points):
                j = int(m.find_closest_point(pts[i]))
                pid[i] = j
                ok[i] = float(np.linalg.norm(pts[i] - m.points[j])) <= tol
            # Per-SEGMENT filter: keep exactly the pieces still on interior
            # surface. An all-or-nothing threshold left whole curves floating
            # over trimmed-away regions (or hid valid remainders).
            keep = []
            for ci, seg in enumerate(cur.lines.reshape(-1, 3)):
                i0, i1 = int(seg[1]), int(seg[2])
                if ok[i0] and ok[i1]:
                    u, v = int(pid[i0]), int(pid[i1])
                    if u != v:
                        a, bb = (u, v) if u < v else (v, u)
                        if share.get(a * n_points + bb, 0) == 2:
                            keep.append(ci)
            if not keep:
                continue
            if len(keep) == cur.n_cells:
                out.append(cur)
            else:
                out.append(cur.extract_cells(keep).extract_surface())
        return out

    def delete_cells(self, cell_ids):
        """Permanently delete the given cells from original_mesh (shared undo).
        Returns the new current_mesh, or None on a no-op (no mesh, empty/stale
        selection, or a selection covering the whole mesh)."""
        if self.original_mesh is None:
            return None
        n = self.original_mesh.n_cells
        ids = {int(c) for c in cell_ids if 0 <= int(c) < n}
        if not ids or len(ids) >= n:
            return None
        self._push_history()
        keep_ids = np.array([i for i in range(n) if i not in ids], dtype=np.int64)
        self.original_mesh = self.original_mesh.extract_cells(keep_ids).extract_surface()
        return self.current_mesh

    def remesh_region(self, cell_ids) -> Optional[str]:
        """Re-triangulate a selected region: delete the selected faces, cap each
        NEWLY created rim loop with a fresh triangulation (merged back as wall),
        then re-wind normals. One undoable step.

        This is the local fix for a non-manifold junction: select its attached
        faces (tree click), Grow a ring or two, Remesh — the defect's faces are
        replaced by a clean disc over the rim, so the junction is gone and the
        surface stays closed. Pre-existing openings (inlet/outlet, open scans)
        are matched by geometry and left open — but if the selection touches an
        opening's rim, the merged loop counts as new and gets capped (Ctrl+Z).

        Returns a status message, or None on a no-op (no mesh / empty selection /
        selection covering the whole mesh)."""
        if self.original_mesh is None:
            return None
        n = self.original_mesh.n_cells
        ids = {int(c) for c in cell_ids if 0 <= int(c) < n}
        if not ids or len(ids) >= n:
            return None
        self._ensure_labels()

        # Signatures of loops that already exist (identical geometry survives the
        # deletion untouched, so centroid + edge count match exactly within tol).
        b = np.asarray(self.original_mesh.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        tol = 1e-6 * diag if diag > 0 else 1e-6
        before = [(g.n_cells, np.asarray(g.points).mean(axis=0))
                  for g in self.detect_open_profiles()]

        def _is_preexisting(loop):
            c = np.asarray(loop.points).mean(axis=0)
            return any(nc == loop.n_cells and float(np.linalg.norm(c - bc)) <= tol
                       for nc, bc in before)

        self._push_history()
        keep = np.array([i for i in range(n) if i not in ids], dtype=np.int64)
        result = self.original_mesh.extract_cells(keep).extract_surface()
        result.cell_data[PATCH_ID] = np.asarray(result.cell_data[PATCH_ID], dtype=np.int64)

        def _new_loops(mesh):
            edges = mesh.extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                manifold_edges=False, non_manifold_edges=False)
            return [g for g in self._split_edge_groups(edges)
                    if not _is_preexisting(g)]

        # Fan-fill each new rim loop. NOT the contour triangulator used by
        # fill_profile: on the small non-planar rims left by a defect deletion,
        # it can pair the rim points into different EDGES than the rim's own
        # (ambiguous projection), so the cap boundary never glues to the rim and
        # a residual hole survives the merge. The centroid fan reuses the rim's
        # exact edges — one triangle per rim edge — so it seals by construction
        # (verified on real scan data; Smooth afterwards if shape matters).
        filled = 0
        for loop in _new_loops(result):
            cap = self._fan_fill(loop)
            if cap is None:
                continue
            result = self._merge_labeled(result, cap, 0)   # new faces are wall
            filled += 1
        failed = len(_new_loops(result))

        # Re-wind so the fresh caps agree with the surrounding winding (and point
        # outward when the result is watertight).
        result = result.compute_normals(
            cell_normals=False, point_normals=True, split_vertices=False,
            consistent_normals=True, auto_orient_normals=self._mesh_is_closed(result))
        self.original_mesh = self._carry_labels(result.triangulate(), self.current_mesh)
        added = self.original_mesh.n_cells - len(keep)
        msg = (f"Remeshed region: {len(ids)} faces removed, {added} rebuilt "
               f"({filled} rim loop(s) re-triangulated)")
        if failed:
            msg += f"; {failed} small loop(s) remain open (use Fill Pinholes)"
        return msg

    def cut_by_plane(self, origin, normal):
        """Split the surface along the plane but keep it one connected, still-closed
        surface (the cut becomes an internal feature curve, not an open boundary).
        Records the cut curve; mutates original_mesh; pushes shared undo. Returns the
        new _wall_mesh, or None if the plane misses the surface (a no-op)."""
        if self.original_mesh is None:
            return None
        a, b = self.original_mesh.clip(normal, origin=origin, return_clipped=True)
        if a.n_cells == 0 or b.n_cells == 0:      # plane missed -> nothing to cut
            return None
        cut_curve = a.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        self._push_history()
        # _carry_labels: the two halves' label arrays can be dropped by merge on
        # array-type mismatch; remap from the pre-cut mesh (still current here).
        # triangulate(): clipping emits quads — restore the all-triangle invariant.
        self.original_mesh = self._carry_labels(
            a.merge(b, merge_points=True).triangulate(), self.current_mesh)
        self._feature_curves.append(cut_curve)
        return self.current_mesh

    def cut_by_box(self, box_planes_data):
        """Split the surface along an oriented box's faces, keeping it one closed
        surface; record the cut curve. box_planes_data: list of (normal, point).
        Returns new current_mesh, or None if the box does not intersect the surface."""
        if self.original_mesh is None or not box_planes_data:
            return None
        planes = vtk.vtkPlanes()
        pts = vtk.vtkPoints()
        norms = vtk.vtkDoubleArray()
        norms.SetNumberOfComponents(3)
        for normal, point in box_planes_data:
            pts.InsertNextPoint(*point)
            norms.InsertNextTuple3(*normal)
        planes.SetPoints(pts)
        planes.SetNormals(norms)
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.original_mesh)
        clipper.SetClipFunction(planes)
        clipper.GenerateClippedOutputOn()
        clipper.Update()
        a = pv.wrap(clipper.GetOutput())
        b = pv.wrap(clipper.GetClippedOutput())
        # vtkClipPolyData may produce UnstructuredGrid when the implicit function
        # introduces non-triangular cells; extract the surface as PolyData first.
        if hasattr(a, 'extract_surface'):
            a = a.extract_surface()
        if hasattr(b, 'extract_surface'):
            b = b.extract_surface()
        if a.n_cells == 0 or b.n_cells == 0:
            return None
        cut_curve = a.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        self._push_history()
        # _carry_labels: the two halves' label arrays can be dropped by merge on
        # array-type mismatch; remap from the pre-cut mesh (still current here).
        # triangulate(): clipping emits quads — restore the all-triangle invariant.
        self.original_mesh = self._carry_labels(
            a.merge(b, merge_points=True).triangulate(), self.current_mesh)
        self._feature_curves.append(cut_curve)
        return self.current_mesh

    def _push_history(self):
        """Snapshot the full edit state (mesh with labels, curves, patch registry)
        for shared Ctrl+Z undo."""
        self._trim_history.append((self.current_mesh.copy(), list(self._feature_curves),
                                   dict(self.patch_names), self._next_patch_id,
                                   dict(self._patch_normals)))

    def undo_trim(self) -> bool:
        """Restore the state from before the most recent operation.
        Returns True if a state was restored, False if there is no history."""
        if not self._trim_history:
            return False
        mesh, curves, names, next_id, normals = self._trim_history.pop()
        self.original_mesh = mesh
        self._feature_curves = curves
        self.patch_names = names
        self._next_patch_id = next_id
        self._patch_normals = normals
        return True

    def get_wall_mesh(self) -> Optional[pv.PolyData]:
        return self.current_mesh

    def check_normals(self) -> dict:
        """Normal-orientation health of current_mesh.

        consistent : neighboring triangles agree on winding (a directed edge
                     appearing twice means two faces disagree). None if no mesh
                     or non-triangle faces.
        flipped_edges : number of directed edges with winding conflicts.
        outward    : for a CLOSED consistent surface, True if the winding points
                     outward (signed volume > 0). None when open/inconsistent.
        """
        m = self.current_mesh
        flipped = self._count_winding_flips(m)
        if flipped is None:
            return {"consistent": None, "flipped_edges": 0, "outward": None}
        consistent = flipped == 0
        outward = None
        if consistent and self._mesh_is_closed(m):        # watertight -> signed volume
            tri = m.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
            p = m.points
            vol6 = float(np.einsum(
                'ij,ij->i', p[tri[:, 0]],
                np.cross(p[tri[:, 1]], p[tri[:, 2]])).sum())
            outward = bool(vol6 > 0)
        return {"consistent": consistent, "flipped_edges": flipped, "outward": outward}

    def geometry_quality(self) -> dict:
        """Return geometry quality metrics for the current mesh (caps included —
        they are part of current_mesh in the single-mesh model)."""
        wall = self.current_mesh
        if wall is None or wall.n_cells == 0:
            return {"open_edges": 0, "open_profiles": 0, "boundary_mesh": None,
                    "non_manifold_edges": 0, "is_manifold": None,
                    "non_manifold_mesh": None}

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

        non_manifold = wall.extract_feature_edges(
            boundary_edges=False, feature_edges=False,
            manifold_edges=False, non_manifold_edges=True,
        )
        n_nm = non_manifold.n_cells
        # Manifoldness only meaningful once openings are named/capped
        is_mf = wall.is_manifold if self.patch_names else None

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
        """Replace current_mesh with the repaired version. Snapshot-first so
        repair is undoable like every other operation; patch labels ride the
        mesh (re-attached by nearest-face mapping if a filter dropped them)."""
        self._push_history()
        repaired = repaired_mesh.triangulate()
        self.original_mesh = self._carry_labels(repaired, self.current_mesh)

    def repair_clean(self) -> str:
        """Remove duplicate points and degenerate triangles from original mesh."""
        if self.original_mesh is None:
            return "No mesh loaded."
        before = self.original_mesh.n_cells
        repaired = self.original_mesh.clean()
        after = repaired.n_cells
        self._apply_repair(repaired)
        return f"Cleaned: {before} \u2192 {after} faces ({before - after} removed)"

    @staticmethod
    def _mesh_is_closed(m) -> bool:
        """True if the surface has no boundary (open) edges — i.e. watertight."""
        if m is None or m.n_cells == 0:
            return False
        boundary = m.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        return boundary.n_cells == 0

    def _is_closed(self) -> bool:
        return self._mesh_is_closed(self.current_mesh)

    @staticmethod
    def _count_winding_flips(m):
        """Number of MANIFOLD-edge winding conflicts in a triangulated surface,
        or None if the mesh is empty or has non-triangle faces.

        A directed edge appearing twice in the SAME orientation means its two
        incident faces disagree on winding. Only edges with exactly two incident
        faces are counted: a non-manifold edge (3+ faces) always yields a
        same-direction duplicate regardless of winding, so it is a structural
        defect (reported separately), not a fixable flip."""
        if m is None or m.n_cells == 0:
            return None
        faces = m.faces
        if faces.size != 4 * m.n_cells or not bool((faces.reshape(-1, 4)[:, 0] == 3).all()):
            return None
        tri = faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        n = m.n_points
        de = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        keys = de[:, 0] * n + de[:, 1]
        und = de.min(axis=1) * n + de.max(axis=1)
        und_unique, und_cnt = np.unique(und, return_counts=True)
        manifold_und = set(und_unique[und_cnt == 2].tolist())
        dir_unique, dir_cnt = np.unique(keys, return_counts=True)
        flipped = 0
        for k in dir_unique[dir_cnt > 1].tolist():
            a, b = k // n, k % n
            if (min(a, b) * n + max(a, b)) in manifold_und:
                flipped += 1
        return flipped

    def repair_normals(self, max_nonmanifold_faces: int = 200) -> str:
        """Fix face-normal winding, automatically clearing a small non-manifold
        block if one prevents consistency.

        First re-wind for consistency (auto-oriented OUTWARD when the surface is
        closed — auto-orient is undefined on open surfaces). If re-winding cannot
        reach a consistent winding, the cause is almost always a few non-manifold
        edges (3+ faces on one edge) that make the surface locally non-orientable:
        vtk will not propagate winding across them, stranding the flipped faces
        behind them. When only a SMALL number of faces sit on those edges
        (<= max_nonmanifold_faces), remove them and re-wind — that reconnects the
        surface into orientable pieces and clears the flips. This opens small
        holes (close them with Fill Pinholes or Make Watertight). A larger
        non-manifold defect is left to Make Watertight (MeshFix). One undoable
        step."""
        if self.original_mesh is None:
            return "No mesh loaded."

        def _rewind(mesh):
            return mesh.compute_normals(
                cell_normals=False, point_normals=True, split_vertices=False,
                consistent_normals=True, auto_orient_normals=self._mesh_is_closed(mesh))

        base = self.current_mesh
        candidate = _rewind(base)
        if self._count_winding_flips(candidate) == 0:
            self._apply_repair(candidate)
            return ("Normals fixed (consistent winding, oriented outward)"
                    if self._mesh_is_closed(candidate)
                    else "Normals fixed (consistent winding)")

        # Re-winding alone did not converge -> find the non-manifold block.
        residual = self._count_winding_flips(candidate) or 0
        groups = self.detect_nonmanifold_edges()
        nm_faces = sorted({c for g in groups for c in self.faces_on_edges(g)})
        if not nm_faces:
            self._apply_repair(candidate)
            return (f"Re-wound normals, but {residual} edge(s) remain flipped and no "
                    f"non-manifold edges were found — inspect this region manually.")
        if len(nm_faces) > max_nonmanifold_faces:
            self._apply_repair(candidate)
            return (f"Re-wound normals, but {residual} edge(s) remain flipped due to "
                    f"{len(groups)} non-manifold junction(s) spanning {len(nm_faces)} "
                    f"faces — too many to auto-remove. Use Make Watertight (MeshFix).")

        # Small non-manifold defect: strip the offending faces, then re-wind.
        n = base.n_cells
        strip = set(nm_faces)
        keep = np.array([i for i in range(n) if i not in strip], dtype=np.int64)
        stripped = base.extract_cells(keep).extract_surface()
        self._apply_repair(_rewind(stripped))
        after = self._count_winding_flips(self.current_mesh)
        tail = "winding now consistent" if after == 0 else f"{after} edge(s) still flipped"
        return (f"Normals fixed — removed {len(nm_faces)} non-manifold face(s) at "
                f"{len(groups)} junction(s) that blocked re-winding ({tail}); this opens "
                f"small holes — close them with Fill Pinholes or Make Watertight.")

    def repair_fill_pinholes(self, max_radius: float) -> str:
        """Fill small holes (boundary loops) up to max_radius via vtkFillHolesFilter.
        Intended for tiny scan defects — named openings should be larger than the
        radius so they stay open. Undoable; labels re-carried (new fill triangles
        inherit the nearest face's patch)."""
        if self.original_mesh is None:
            return "No mesh loaded."
        before = len(self.detect_open_profiles())
        if before == 0:
            return "No open profiles — nothing to fill."
        f = vtk.vtkFillHolesFilter()
        f.SetInputData(self.current_mesh)
        f.SetHoleSize(float(max_radius))
        f.Update()
        out = pv.wrap(f.GetOutput())
        if out is None or out.n_cells == 0:
            return "Fill pinholes produced nothing — no change."
        out = out.triangulate()
        # New fill triangles can come out with arbitrary winding; re-consist.
        out = out.compute_normals(cell_normals=False, point_normals=True,
                                  split_vertices=False, consistent_normals=True,
                                  auto_orient_normals=False)
        self._apply_repair(out)
        after = len(self.detect_open_profiles())
        return f"Pinholes filled: open profiles {before} → {after}"

    def repair_decimate(self, reduction: float) -> str:
        """Reduce triangle density by ~`reduction` (0–1 fraction of faces to
        remove) via quadric decimation. Undoable; patch labels are re-carried
        onto the decimated mesh by nearest-face mapping."""
        if self.original_mesh is None:
            return "No mesh loaded."
        reduction = min(max(float(reduction), 0.05), 0.95)
        before = self.current_mesh.n_cells
        try:
            decimated = self.current_mesh.triangulate().decimate(reduction)
        except Exception as e:
            return f"Decimation failed: {e}"
        if decimated is None or decimated.n_cells == 0:
            return "Decimation produced an empty mesh — no change."
        self._apply_repair(decimated)
        after = self.current_mesh.n_cells
        return (f"Decimated: {before:,} → {after:,} faces "
                f"({100.0 * (before - after) / before:.0f}% removed)")

    def remesh_surface(self, target_faces: int = None) -> str:
        """Whole-surface ISOTROPIC remesh (ACVD clustering via pyacvd).

        Re-tessellates current_mesh into near-equilateral triangles of uniform
        size — the standard CFD preparation step after repair. New vertices are
        placed ON the original surface (centroidal Voronoi clustering, not
        smoothing), so geometry is preserved to a fraction of the edge length.
        Patch labels are re-carried by nearest face — boundaries between
        patches become approximate, so run this BEFORE clipping/naming when
        crisp patch borders matter. Cut feature curves no longer lie on mesh
        edges afterwards and stop being drawn. One undo step.

        Best on watertight surfaces; open boundaries may be re-tessellated
        raggedly (the message warns when openings exist)."""
        if self.original_mesh is None:
            return "No mesh loaded."
        try:
            import pyacvd
        except ImportError:
            return ("pyacvd is not installed — run: pip install pyacvd "
                    "(note: not usable in the legacy conda env — its bundled "
                    "OpenMP clashes with vmtk's)")
        m = self.current_mesh.triangulate()
        before = m.n_cells
        target = int(target_faces) if target_faces else before
        if target < 100:
            return "Target too small — need at least 100 faces."
        n_open = len(self.detect_open_profiles())
        # ACVD clusters VERTICES; a closed triangulated surface has roughly
        # twice as many faces as points, so aim for target/2 clusters. The
        # input needs comfortably more points than clusters — subdivide first
        # when upsampling.
        n_clusters = max(50, target // 2)
        try:
            clus = pyacvd.Clustering(m)
            guard = 0
            while clus.mesh.n_points < 3 * n_clusters and guard < 3:
                clus.subdivide(2)
                guard += 1
            clus.cluster(n_clusters)
            out = clus.create_mesh()
        except Exception as e:
            return f"Remesh failed: {type(e).__name__}: {e}"
        if out is None or out.n_cells == 0:
            return "Remesh produced an empty mesh — no change."
        self._push_history()
        self.original_mesh = self._carry_labels(out.triangulate(), self.current_mesh)
        # A patch smaller than the new triangle size can lose all its faces in
        # the relabeling — purge such names so exports stay consistent.
        labels = np.asarray(self.original_mesh.cell_data[PATCH_ID])
        present = set(int(v) for v in np.unique(labels))
        for old_pid in [p for p in self.patch_names if p not in present]:
            del self.patch_names[old_pid]
            self._patch_normals.pop(old_pid, None)
        after = self.original_mesh.n_cells
        msg = f"Remeshed surface: {before:,} → {after:,} faces (isotropic)"
        if n_open:
            msg += (f". Note: {n_open} open profile(s) — open boundaries "
                    f"re-tessellate raggedly; cap or fill them first for a "
                    f"clean rim.")
        return msg

    def repair_make_watertight(self) -> str:
        """MeshFix (pymeshfix): close ALL holes, remove self-intersections and
        non-manifold geometry, keep the largest component. Use BEFORE clipping —
        it will also fill named inlet/outlet openings. Undoable."""
        if self.original_mesh is None:
            return "No mesh loaded."
        try:
            import pymeshfix
        except ImportError:
            return ("pymeshfix is not installed — run: "
                    "conda run -n mesh-prep pip install pymeshfix")
        before = self._health_stats()
        try:
            mf = pymeshfix.MeshFix(self.current_mesh.triangulate())
            mf.repair()
            out = mf.mesh
        except Exception as e:
            return f"MeshFix failed: {e}"
        if out is None or out.n_cells == 0:
            return "MeshFix produced an empty mesh — no change."
        self._apply_repair(out)
        after = self._health_stats()
        return ("MeshFix — "
                f"faces {before['cells']}→{after['cells']}, "
                f"open {before['open']}→{after['open']}, "
                f"non-manifold {before['nm']}→{after['nm']}, "
                f"pieces {before['pieces']}→{after['pieces']}")

    def _health_stats(self):
        """Mesh health snapshot for the auto-repair summary."""
        m = self.original_mesh
        nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=True).n_cells
        op = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=False).n_cells
        return {"cells": m.n_cells, "nm": nm, "open": op, "pieces": len(self.detect_pieces())}

    def auto_repair(self):
        """Clean + fix-normals in one pass; return a before->after health summary.
        Best-effort (pyvista has no true non-manifold repair). Replaces original_mesh
        and clears clips via _apply_repair, like the existing repair buttons."""
        if self.original_mesh is None:
            return "No mesh loaded."
        before = self._health_stats()
        m = self.original_mesh.clean()
        m = m.compute_normals(cell_normals=False, point_normals=True,
                              split_vertices=False, consistent_normals=True,
                              auto_orient_normals=self._is_closed())
        self._apply_repair(m)
        after = self._health_stats()
        return ("Auto-repair — "
                f"faces {before['cells']}->{after['cells']}, "
                f"non-manifold {before['nm']}->{after['nm']}, "
                f"open edges {before['open']}->{after['open']}, "
                f"pieces {before['pieces']}->{after['pieces']}")

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
        """Write a single ASCII STL with one solid block per patch label
        (named patches first, wall last — solid names become pMesh patch names)."""
        patches = self.patches_by_id()
        blocks = []
        for pid in sorted(patches, key=lambda p: (p == 0, p)):   # named first, wall last
            blocks.append(self._polydata_to_ascii_stl_block(
                patches[pid], self.patch_name_for(pid), scale_factor))
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
        if self.current_mesh is None or self.current_mesh.n_cells == 0:
            raise RuntimeError("No mesh to export.")

        # current_mesh already includes every cap — it IS the combined surface.
        combined = self.current_mesh
        block = self._polydata_to_ascii_stl_block(combined, solid_name, scale_factor)
        with open(filepath, "w") as f:
            f.write(block)

        ids = np.asarray(combined.cell_data[PATCH_ID]) if PATCH_ID in combined.cell_data else None
        n_wall = int(np.count_nonzero(ids == 0)) if ids is not None else int(combined.n_cells)
        return {
            "filepath": filepath,
            "n_wall_cells": n_wall,
            "n_caps_merged": len(self.patch_names),
            "n_open_clips": 0,
            "n_total_cells": int(combined.n_cells),
        }

    def export_clip_planes(self, filepath: str):
        """Save the named-patch registry as JSON: name, patch_id, outward normal
        (exact clip-plane normal for clip-created patches, null for filled ones)
        and the cap centroid as center."""
        patches = self.patches_by_id()
        entries = []
        for pid, name in sorted(self.patch_names.items()):
            cap = patches.get(pid)
            center = (list(float(v) for v in cap.center)
                      if cap is not None and cap.n_cells > 0 else None)
            entries.append({
                "name": name,
                "patch_id": pid,
                "outward_normal": list(self._patch_normals[pid])
                if pid in self._patch_normals else None,
                "center": center,
            })
        with open(filepath, "w") as f:
            json.dump({"patches": entries}, f, indent=2)

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

        # 2. Collect patch metadata from the label registry
        stl_stem = stl_filename.rsplit(".", 1)[0] if "." in stl_filename else stl_filename
        patch_names = list(self.patch_names.values()) + ["wall"]
        patches = self.patches_by_id()
        inlet_normals = {}
        for pid, name in self.patch_names.items():
            if openfoam_case.classify_patch(name) != "inlet":
                continue
            if pid in self._patch_normals:      # clip-created: exact outward normal
                inlet_normals[name] = tuple(self._patch_normals[pid])
            else:                                # fill-created: mean cap normal
                cap = patches.get(pid)
                if cap is not None and cap.n_cells > 0:
                    capn = cap.compute_normals(cell_normals=True, point_normals=False)
                    nvec = np.asarray(capn.cell_normals).mean(axis=0)
                    mag = float(np.linalg.norm(nvec))
                    if mag > 0:
                        inlet_normals[name] = tuple(nvec / mag)

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
                    n for n in self.patch_names.values()
                    if openfoam_case.classify_patch(n) == "outlet"
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
                "name": name,
                "origin": tuple(float(v) for v in (
                    patches[pid].center if pid in patches and patches[pid].n_cells > 0
                    else (0.0, 0.0, 0.0))),
                "normal": tuple(float(v) for v in self._patch_normals.get(pid, (0.0, 0.0, 1.0))),
            }
            for pid, name in sorted(self.patch_names.items())
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
        """Write one STL file per patch label into output_dir (wall included)."""
        os.makedirs(output_dir, exist_ok=True)
        for pid, mesh in self.patches_by_id().items():
            name = self.patch_name_for(pid)
            path = os.path.join(output_dir, f"{name}.stl")
            with open(path, "w") as f:
                f.write(self._polydata_to_ascii_stl_block(mesh, name, scale_factor))


class _SelectLassoStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera style for Select mode. Shift+left-drag draws a lasso
    (camera movement suppressed for that drag, so a patch can be encircled from
    anywhere, including empty space); a plain left-drag rotates the view normally.
    Follows pyvista's subclass + AddObserver + conditional-forward pattern."""

    def __init__(self, app):
        self._app = app
        self._lasso_active = False
        self._last_click_time = 0.0
        self._last_click_pos = None
        self._right_press_pos = None
        self.AddObserver("LeftButtonPressEvent", self._on_press)
        self.AddObserver("MouseMoveEvent", self._on_move)
        self.AddObserver("LeftButtonReleaseEvent", self._on_release)
        self.AddObserver("RightButtonPressEvent", self._on_right_press)
        self.AddObserver("RightButtonReleaseEvent", self._on_right_release)

    def _on_right_press(self, _obj=None, _evt=None):
        try:
            if self._lasso_active:
                return           # mid-lasso: a modal menu would eat the left release
            self._right_press_pos = self._app.plotter.iren.get_event_position()
            self.OnRightButtonDown()                       # keep right-drag zoom
        except Exception:
            logger.exception("select right-press handler failed")

    def _on_right_release(self, _obj=None, _evt=None):
        try:
            if self._lasso_active:
                return
            self.OnRightButtonUp()
            pp = self._right_press_pos
            self._right_press_pos = None
            rp = self._app.plotter.iren.get_event_position()
            if pp is None or abs(rp[0] - pp[0]) > 3 or abs(rp[1] - pp[1]) > 3:
                return                                    # a zoom drag, not a click
            self._app._show_select_context_menu()
        except Exception:
            logger.exception("select right-release handler failed")

    def _on_press(self, _obj=None, _evt=None):
        try:
            # Read the live OS modifier state via Qt rather than VTK's
            # GetShiftKey(), which can drop/lag the modifier on macOS and made
            # Shift+drag intermittently rotate instead of lasso.
            shift = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
            if shift:                                      # Shift+drag -> lasso
                self._app._select_lasso_press()
                self._lasso_active = True
                return                                     # suppress camera (don't forward)
            self._lasso_active = False
            self._press_pos = self._app.plotter.iren.get_event_position()
            self.OnLeftButtonDown()                        # plain drag -> rotate
        except Exception:
            logger.exception("select press handler failed")

    def _on_move(self, _obj=None, _evt=None):
        try:
            if self._lasso_active:
                self._app._select_lasso_move()
            else:
                self.OnMouseMove()
        except Exception:
            logger.exception("select move handler failed")

    def _on_release(self, _obj=None, _evt=None):
        try:
            if self._lasso_active:
                self._lasso_active = False
                self._app._select_lasso_release()
                return
            self.OnLeftButtonUp()
            pp = getattr(self, "_press_pos", None)
            rp = self._app.plotter.iren.get_event_position()
            if pp is None or abs(rp[0] - pp[0]) > 3 or abs(rp[1] - pp[1]) > 3:
                return                                    # a rotate/drag, not a click
            now = time.time()
            last_t = self._last_click_time
            last_p = self._last_click_pos
            if (last_p is not None and (now - last_t) <= 0.4
                    and abs(rp[0] - last_p[0]) <= 8 and abs(rp[1] - last_p[1]) <= 8):
                self._last_click_time = 0.0               # consume; no triple-click chain
                self._app._select_double_click(rp)
            else:
                self._last_click_time = now
                self._last_click_pos = rp
                self._app._select_click_pick(rp)
        except Exception:
            logger.exception("select release handler failed")


class _TwoPointStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera style for the two-point plane tool. Plain drag orbits/zooms;
    Shift+click places a plane-defining point (camera suppressed for that click).
    Mirrors _SelectLassoStyle's subclass + AddObserver + conditional-forward pattern."""

    def __init__(self, app):
        self._app = app
        self.AddObserver("LeftButtonPressEvent", self._on_press)
        self.AddObserver("MouseMoveEvent", self._on_move)
        self.AddObserver("LeftButtonReleaseEvent", self._on_release)

    def _on_press(self, _obj=None, _evt=None):
        try:
            if bool(QApplication.keyboardModifiers() & Qt.ShiftModifier):
                self._app._twopt_click(self._app.plotter.iren.get_event_position())
                return                                   # suppress camera on the click
            self.OnLeftButtonDown()                      # plain press -> orbit
        except Exception:
            logger.exception("two-point press handler failed")

    def _on_move(self, _obj=None, _evt=None):
        try:
            self.OnMouseMove()
        except Exception:
            logger.exception("two-point move handler failed")

    def _on_release(self, _obj=None, _evt=None):
        try:
            self.OnLeftButtonUp()
        except Exception:
            logger.exception("two-point release handler failed")


class STLClipperApp(QMainWindow):
    """Qt GUI with embedded PyVista viewport and control panel."""

    def __init__(self, initial_file: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("STL Boundary Patch Clipper")
        self.resize(1400, 900)

        self.engine = STLClipperEngine()
        self._loaded_filepath = None
        self.setAcceptDrops(True)   # drag & drop an .stl anywhere on the window
        self._plane_widget_active = False       # plane widget on screen
        self._constraint_box_active = False     # optional constraint box on screen
        self._current_plane_origin = None       # from plane widget callback
        self._current_plane_normal = None       # from plane widget callback
        self._twopt_active = False
        self._twopt_first = None
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

        # Selection state
        self._selection: set = set()          # active selected cell ids (original_mesh)
        self._cap_actor_names: set = set()    # cap actors currently drawn (for staleness)
        self._select_mode = False

        self._build_ui()
        self._build_menu()
        self._update_button_states()

        self._trim_undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._trim_undo_shortcut.activated.connect(self._undo_trim)

        self._select_esc_shortcut = QShortcut(QKeySequence("Escape"), self)
        self._select_esc_shortcut.activated.connect(self._on_escape_selection)

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
        # pyvistaqt's interactor has its own file-drop handler that add_mesh()es
        # the file directly (default light-blue, bypassing the engine entirely).
        # Disable it so viewport drops bubble up to the window's dropEvent and
        # load through engine.load_stl like everywhere else.
        self.plotter.interactor.setAcceptDrops(False)
        self.plotter.set_background("black")
        self.plotter.enable_parallel_projection()
        # Interactive axes gizmo in the corner: click a face/arrow to snap to that
        # orthographic view (replaces the old +X/-X..+Z/-Z panel buttons).
        self.plotter.add_camera_orientation_widget()

        # Floating "Fit view" button overlaid in the viewport's top-left corner —
        # recenter/zoom to the geometry without reaching to the side panel.
        self._btn_fit_overlay = QPushButton("⤢ Fit", self.plotter.interactor)
        self._btn_fit_overlay.setToolTip("Recenter and zoom to fit the geometry")
        self._btn_fit_overlay.setCursor(Qt.PointingHandCursor)
        self._btn_fit_overlay.setStyleSheet(
            "QPushButton { background: rgba(45,45,45,190); color: white; "
            "border: 1px solid #999; border-radius: 4px; padding: 3px 9px; } "
            "QPushButton:hover { background: rgba(80,80,80,220); }")
        self._btn_fit_overlay.clicked.connect(self._on_zoom_to_fit)
        self._btn_fit_overlay.adjustSize()
        self._btn_fit_overlay.move(10, 10)
        self._btn_fit_overlay.raise_()
        self._btn_fit_overlay.show()

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

        self.btn_two_point_plane = QPushButton("◪ 2-Point Plane")
        self.btn_two_point_plane.setCheckable(True)   # stays pressed while capture mode is armed
        self.btn_two_point_plane.setToolTip(
            "Shift+click two points in the viewer to define a cut plane along your line of sight")
        self.btn_two_point_plane.clicked.connect(self._on_two_point_plane)
        panel.addWidget(self.btn_two_point_plane)

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

        self.btn_cut = QPushButton("✂ Cut")
        self.btn_cut.clicked.connect(self._on_cut)
        btn_row.addWidget(self.btn_cut)

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

        self._btn_select = QPushButton("🖈 Select faces")
        self._btn_select.setCheckable(True)
        self._btn_select.toggled.connect(self._toggle_select_mode)
        panel.addWidget(self._btn_select)

        self._btn_grow = QPushButton("➕ Grow")
        self._btn_grow.clicked.connect(self._on_grow_selection)
        panel.addWidget(self._btn_grow)

        self._btn_smooth = QPushButton("✨ Smooth (×5)")
        self._btn_smooth.clicked.connect(self._on_smooth_selection)
        panel.addWidget(self._btn_smooth)

        self._btn_remesh_sel = QPushButton("🔧 Remesh region")
        self._btn_remesh_sel.setToolTip(
            "Delete the selected faces and fill the rim with a FLAT patch — "
            "for removing defects (non-manifold junctions), NOT for improving "
            "a curved area (it flattens the shape there). To refine while "
            "keeping the shape, use Smooth or Remesh Surface (isotropic).")
        self._btn_remesh_sel.clicked.connect(self._on_remesh_selection)
        panel.addWidget(self._btn_remesh_sel)

        self._btn_delete = QPushButton("🗑 Delete faces")
        self._btn_delete.clicked.connect(self._on_delete_selection)
        panel.addWidget(self._btn_delete)

        self._btn_fill = QPushButton("🩹 Fill profile → patch")
        self._btn_fill.setToolTip("Fill the open profile selected in the Objects tree into a named patch")
        self._btn_fill.clicked.connect(self._on_fill_profile)
        panel.addWidget(self._btn_fill)

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
        self._lbl_wall_faces = QLabel("Wall: — faces")
        geo_lay.addWidget(self._lbl_wall_faces)
        self._lbl_open_profiles = QLabel("Open profiles: —")
        self._lbl_open_profiles.setWordWrap(True)
        self._lbl_open_profiles.setMinimumWidth(1)   # wrap, never widen the panel
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
        self._lbl_non_manifold.setWordWrap(True)
        self._lbl_non_manifold.setMinimumWidth(1)   # wrap, never widen the panel
        geo_lay.addWidget(self._lbl_non_manifold)
        self._lbl_manifold = QLabel("Manifold: —")
        self._lbl_manifold.setWordWrap(True)
        self._lbl_manifold.setMinimumWidth(1)   # wrap, never widen the panel
        geo_lay.addWidget(self._lbl_manifold)
        self._lbl_normals = QLabel("Normals: —")   # shown in the LEFT Geometry box
        self._lbl_normals.setWordWrap(True)        # long messages wrap to new lines…
        self._lbl_normals.setMinimumWidth(1)       # …instead of widening the panel
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


        panel.addStretch()

        # ── Tab 2: Clip & Save STL (non-CFD use) ──
        self._build_clip_save_tab()

        # ── Tab 3: Export Settings ──
        self._build_export_settings_tab()

        # ── Tab 4: Mesh Repair ──
        self._build_repair_tab()

        # ── Tab 5: Run OpenFOAM (Beta) ──
        self._build_run_tab()

        # Object tree (left) — detected surface entities
        self._tree_mesh = False           # sentinel so the first refresh always builds
        self._tree_profiles = []
        self._tree_nonmanifold = []
        self._tree_pieces = []
        self._tree_patches = []
        self._tree_n_patches = -1
        self._active_profile_index = None
        self._object_tree = QTreeWidget()
        self._object_tree.setHeaderLabel("Objects")
        self._object_tree.itemClicked.connect(self._on_tree_item_clicked)
        self._object_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._object_tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        # Geometry status (X/Y/Z range + Fit) — pinned to the bottom of the left
        # (Objects) panel. Wrapping the TREE widget is safe; the interactor stays a
        # direct splitter child so the orientation gizmo is untouched.
        self._lbl_bounds_x = QLabel("X: -")
        self._lbl_bounds_y = QLabel("Y: -")
        self._lbl_bounds_z = QLabel("Z: -")
        self.btn_zoom_fit = QPushButton("⤡ Fit to view")
        self.btn_zoom_fit.clicked.connect(self._on_zoom_to_fit)
        _geom_box = QGroupBox("Geometry")
        _geom_lay = QVBoxLayout(_geom_box)
        _geom_lay.setContentsMargins(6, 4, 6, 4)
        _geom_lay.addWidget(self._lbl_bounds_x)
        _geom_lay.addWidget(self._lbl_bounds_y)
        _geom_lay.addWidget(self._lbl_bounds_z)
        _nrm_row = QHBoxLayout()                     # normal health + quick fix
        _nrm_row.addWidget(self._lbl_normals, 1)
        self._btn_fix_normals_quick = QPushButton("Fix")
        self._btn_fix_normals_quick.setFixedWidth(44)
        self._btn_fix_normals_quick.setToolTip(
            "Fix Normals (consistent winding) — same as the Repair tab button")
        self._btn_fix_normals_quick.clicked.connect(self._on_repair_normals)
        self._btn_fix_normals_quick.setVisible(False)
        _nrm_row.addWidget(self._btn_fix_normals_quick)
        _geom_lay.addLayout(_nrm_row)
        _geom_lay.addWidget(self.btn_zoom_fit)

        _left_pane = QWidget()
        _left_lay = QVBoxLayout(_left_pane)
        _left_lay.setContentsMargins(0, 0, 0, 0)
        _left_lay.addWidget(self._object_tree, 1)   # tree fills the column
        _left_lay.addWidget(_geom_box)              # geometry status pinned at the bottom

        # Draggable splitter: (objects tree + geometry) | viewport | tab panel. The
        # interactor stays a DIRECT splitter child — wrapping it re-parents the VTK
        # render window and resets the orientation gizmo, so we don't.
        splitter = QSplitter(Qt.Horizontal, central)
        splitter.addWidget(_left_pane)
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(self._tab_widget)
        # Side panels keep their content-sized width on window resize (stretch 0);
        # the viewport absorbs all extra space. Manual dragging still works.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        layout.addWidget(splitter)
        self._splitter = splitter
        # Size the panels to their content once the window has real geometry.
        QTimer.singleShot(0, self._init_splitter_sizes)

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
        self._clip_save_status.setMinimumWidth(1)   # wrap, never widen the panel
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

        named = self.engine.named_patches()
        if not named:
            placeholder = QLabel("(no named patches — clip or fill to create some)")
            placeholder.setStyleSheet("color: #888;")
            self._clip_save_list_layout.addWidget(placeholder)
            self.btn_save_clipped_stl.setEnabled(self.engine.current_mesh is not None)
            return

        self.btn_save_clipped_stl.setEnabled(True)

        for pid, name, nfaces in named:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)

            r, g, b = [int(c * 255) for c in _color_for_name(name)]
            swatch = QLabel("  ")
            swatch.setFixedWidth(14)
            swatch.setFixedHeight(14)
            swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"
            )
            row.addWidget(swatch)

            lbl = QLabel(f"{name}  ({nfaces:,} cap faces)")
            lbl.setMinimumWidth(180)
            row.addWidget(lbl, stretch=1)

            self._clip_save_list_layout.addWidget(row_widget)

    def _on_save_clipped_stl(self):
        """File-dialog save of the current surface as a single-solid STL."""
        if self.engine.current_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Load an STL first.")
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
            names = list(self.engine.patch_names.values())
            self._clip_save_status.setText(
                f"Saved: {os.path.basename(filepath)}  "
                f"({result['n_total_cells']:,} faces, "
                f"{result['n_caps_merged']} named patches)"
            )
            QMessageBox.information(
                self, "Saved",
                f"Single-solid STL saved to:\n{filepath}\n"
                f"Scale factor: ×{sf}\n"
                f"Total faces: {result['n_total_cells']:,}\n\n"
                f"Named patches: {', '.join(names) if names else 'none'}",
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

        self._btn_auto_repair = QPushButton("\ud83d\udd27 Auto-repair")
        self._btn_auto_repair.setToolTip("Clean + fix normals in one pass, with a before/after summary")
        self._btn_auto_repair.clicked.connect(self._on_auto_repair)
        tab3.addWidget(self._btn_auto_repair)

        pin_row = QHBoxLayout()
        self._btn_fill_pinholes = QPushButton("Fill Pinholes")
        self._btn_fill_pinholes.setToolTip(
            "Fill small holes up to the given size (tiny scan defects). "
            "Named openings larger than the size stay open.")
        self._btn_fill_pinholes.clicked.connect(self._on_fill_pinholes)
        pin_row.addWidget(self._btn_fill_pinholes)
        self._spin_pinhole_pct = QDoubleSpinBox()
        self._spin_pinhole_pct.setRange(0.1, 50.0)
        self._spin_pinhole_pct.setValue(2.0)
        self._spin_pinhole_pct.setSuffix(" % of size")
        self._spin_pinhole_pct.setToolTip("Max hole size as % of the bounding-box diagonal")
        pin_row.addWidget(self._spin_pinhole_pct)
        tab3.addLayout(pin_row)

        dec_row = QHBoxLayout()
        self._btn_decimate = QPushButton("Decimate")
        self._btn_decimate.setToolTip(
            "Reduce triangle density by the given percentage (quadric "
            "decimation). Best run before clipping; patch labels are remapped.")
        self._btn_decimate.clicked.connect(self._on_decimate)
        dec_row.addWidget(self._btn_decimate)
        self._spin_decimate_pct = QDoubleSpinBox()
        self._spin_decimate_pct.setRange(5.0, 95.0)
        self._spin_decimate_pct.setValue(50.0)
        self._spin_decimate_pct.setSuffix(" % fewer faces")
        dec_row.addWidget(self._spin_decimate_pct)
        tab3.addLayout(dec_row)

        rem_row = QHBoxLayout()
        self._btn_remesh_surface = QPushButton("Remesh Surface (isotropic)")
        self._btn_remesh_surface.setToolTip(
            "Re-tessellate the whole surface into uniform, near-equilateral "
            "triangles (ACVD). Geometry preserved; patch labels remapped — "
            "best run after repair, before clipping/naming.")
        self._btn_remesh_surface.clicked.connect(self._on_remesh_surface)
        rem_row.addWidget(self._btn_remesh_surface)
        tab3.addLayout(rem_row)

        self._btn_watertight = QPushButton("Make Watertight (MeshFix)")
        self._btn_watertight.setToolTip(
            "pymeshfix: close ALL holes, remove self-intersections/non-manifold "
            "geometry, keep the largest component. Use BEFORE clipping \u2014 it also "
            "fills named inlet/outlet openings.")
        self._btn_watertight.clicked.connect(self._on_make_watertight)
        tab3.addWidget(self._btn_watertight)

        self._lbl_repair_status = QLabel("Status: \u2014")
        self._lbl_repair_status.setWordWrap(True)
        self._lbl_repair_status.setMinimumWidth(1)   # wrap, never widen the panel
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
        # Tab intentionally NOT added: solver runs are out of scope for now.
        # Widgets are still built because export handlers update _run_case_label
        # and the worker plumbing stays valid. Re-enable by restoring addTab.
        # self._tab_widget.addTab(scroll, "Run OpenFOAM (Beta)")

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

    @staticmethod
    def _quiesce_worker(worker, timeout_ms: int = 5000) -> bool:
        """Detach a running QThread worker's signals and wait for it to finish.

        Stops a background worker (e.g. the VMTK centerline thread, which has no
        cancel path) from emitting into a window that is being destroyed, which
        would crash PyQt. Returns True if the worker was already idle or finished
        within the timeout, False if it is still running.
        """
        if worker is None or not worker.isRunning():
            return True
        worker.blockSignals(True)   # emissions can no longer reach the dying window
        return worker.wait(timeout_ms)

    def dragEnterEvent(self, event):
        """Accept drags carrying at least one local .stl file (whole window,
        viewport included — unhandled child drags propagate up to here)."""
        if any(u.isLocalFile() and u.toLocalFile().lower().endswith(".stl")
               for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Load the first dropped .stl through the same path as the Load button."""
        for u in event.mimeData().urls():
            if u.isLocalFile() and u.toLocalFile().lower().endswith(".stl"):
                event.acceptProposedAction()
                self._load_file(u.toLocalFile())
                return

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
        # VMTK centerline has no cancel; detach its signals so it can't emit
        # into the destroyed window, then give it a moment to finish.
        self._quiesce_worker(self._centerline_worker)
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

    def _on_fill_pinholes(self):
        if self.engine.current_mesh is None:
            return
        b = np.asarray(self.engine.current_mesh.bounds, dtype=float)
        diag = float(np.linalg.norm(b[1::2] - b[0::2]))
        radius = diag * self._spin_pinhole_pct.value() / 100.0
        msg = self.engine.repair_fill_pinholes(radius)
        self._lbl_repair_status.setText(msg)
        self._centerline_mesh = None
        self._refresh_patch_list()
        self._refresh_object_tree()
        self._refresh_display()
        self._update_button_states()
        self.status.showMessage(f"{msg} Ctrl+Z to undo.")

    def _on_decimate(self):
        if self.engine.current_mesh is None:
            return
        msg = self.engine.repair_decimate(self._spin_decimate_pct.value() / 100.0)
        self._lbl_repair_status.setText(msg)
        self._centerline_mesh = None
        self._refresh_patch_list()
        self._refresh_object_tree()
        self._refresh_display()
        self._update_button_states()
        self.status.showMessage(f"{msg} Ctrl+Z to undo.")

    def _on_remesh_surface(self):
        if self.engine.current_mesh is None:
            return
        target, ok = QInputDialog.getInt(
            self, "Remesh Surface",
            "Target face count:", self.engine.current_mesh.n_cells,
            100, 10_000_000, 1000)
        if not ok:
            return
        msg = self.engine.remesh_surface(target)
        self._lbl_repair_status.setText(msg)
        self._centerline_mesh = None
        self._refresh_patch_list()
        self._refresh_object_tree()
        self._refresh_display()
        self._update_button_states()
        self.status.showMessage(f"{msg} Ctrl+Z to undo.")

    def _on_make_watertight(self):
        if self.engine.current_mesh is None:
            return
        if self.engine.patch_names:
            reply = QMessageBox.question(
                self, "MeshFix Warning",
                "MeshFix fills ALL openings — including your named inlet/outlet "
                "patches — and keeps only the largest piece. It is meant to run "
                "BEFORE clipping.\n\nRun anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        msg = self.engine.repair_make_watertight()
        self._lbl_repair_status.setText(msg)
        self._centerline_mesh = None
        self._refresh_patch_list()
        self._refresh_object_tree()
        self._refresh_display()
        self._update_button_states()
        self.status.showMessage(f"{msg} Ctrl+Z to undo.")

    def _on_auto_repair(self):
        msg = self.engine.auto_repair()
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
        """Fix macOS Retina DPR mismatch for VTK widget picking (VTK < 9.4).

        On VTK 9.2.x, pyvistaqt scales mouse coords by device-pixel-ratio
        before passing them to VTK, but vtkCocoaRenderWindow reports size in
        logical pixels. This makes widget pickers (plane, box) receive
        physical-pixel coords against a logical-pixel viewport — the pick ray
        misses.  Patch the interactor to keep everything in logical-pixel space.

        VTK >= 9.4 fixed Qt/Retina handling upstream: event coords AND window
        size are both physical pixels, already consistent. Applying this patch
        there forces logical size onto a physical-coordinate stack and shifts
        every click by the device-pixel-ratio (2x offset on Retina) — so the
        patch must be skipped on modern VTK.
        """
        major = vtk.vtkVersion.GetVTKMajorVersion()
        minor = vtk.vtkVersion.GetVTKMinorVersion()
        if (major, minor) >= (9, 4):
            logger.info("Retina DPR patch SKIPPED (VTK %d.%d handles Qt "
                        "device-pixel-ratio natively)", major, minor)
            return
        logger.info("Retina DPR patch applied (VTK %d.%d reports logical-pixel "
                    "window size)", major, minor)
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
        has_mesh = self.engine.current_mesh is not None
        has_patches = bool(self.engine.patch_names)
        plane_active = self._plane_widget_active

        self.btn_add_plane.setEnabled(has_mesh and not plane_active)
        self.btn_confirm_plane.setEnabled(plane_active and not self._plane_confirmed)
        self.btn_add_constraint.setEnabled(self._plane_confirmed and not self._constraint_box_active)
        self.btn_confirm.setEnabled(self._plane_confirmed)
        self.btn_cancel.setEnabled(plane_active)
        self.btn_flip.setEnabled(plane_active and not self._plane_confirmed)
        self.btn_rename.setEnabled(has_patches)
        self.btn_delete.setEnabled(has_patches)
        has_export = has_patches
        self.btn_export_foam.setEnabled(has_export)
        self.btn_export_sep.setEnabled(has_export)
        self.btn_export_comb.setEnabled(has_export)
        self.btn_save_of_stl.setEnabled(has_export)

        # Centerline buttons
        names = list(self.engine.patch_names.values())
        has_inlet = any(openfoam_case.classify_patch(n) == "inlet" for n in names)
        has_outlet = any(openfoam_case.classify_patch(n) == "outlet" for n in names)
        computing = self._centerline_worker is not None and self._centerline_worker.isRunning()
        self.btn_compute_cl.setEnabled(has_inlet and has_outlet and not computing)
        self.btn_clear_cl.setEnabled(self._centerline_mesh is not None)
        self.btn_save_cl.setEnabled(self._centerline_mesh is not None)

        # Repair buttons
        self._btn_repair_clean.setEnabled(has_mesh)
        self._btn_repair_normals.setEnabled(has_mesh)
        self._btn_auto_repair.setEnabled(has_mesh)

        # Selection buttons
        has_sel = bool(getattr(self, "_selection", set()))
        can_edit = self._edit_enabled()
        if hasattr(self, "_btn_select"):
            self._btn_select.setEnabled(can_edit)
            self._btn_grow.setEnabled(can_edit and has_sel)
            self._btn_smooth.setEnabled(can_edit and has_sel)
            self._btn_remesh_sel.setEnabled(can_edit and has_sel)
            self._btn_delete.setEnabled(can_edit and has_sel)

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
        self._clear_selection()
        self._refresh_display(fit_camera=True)

        self._refresh_patch_list()          # previous file's patches are gone
        self._refresh_object_tree()
        self._update_status()
        fname = os.path.basename(filepath)
        n = mesh.n_cells
        self.status.showMessage(f"Loaded: {fname} | {n:,} faces")
        self._update_button_states()

    # ------------------------------------------------------------------
    # Clip plane & constraint box widgets
    # ------------------------------------------------------------------

    def _on_add_plane(self):
        """Start clip workflow: add interactive plane widget."""
        if getattr(self, "_twopt_active", False):
            self._end_two_point()               # leave two-point capture before the plane widget
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
        diag = np.linalg.norm(np.ptp(bounds, axis=1))
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
        radius = np.linalg.norm(np.ptp(bounds, axis=1)) * 0.5
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
        extents = np.ptp(bounds, axis=1)  # [dx, dy, dz]
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
        extents = np.ptp(bounds, axis=1)
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
            color="green", name="box_center_marker", reset_camera=False,
        )

        self._update_preview()

    def _on_reset_box(self):
        """Reset box controls to initial values (mesh center, no rotation, default size)."""
        mesh = self.engine.get_wall_mesh()
        if mesh is None:
            return
        center = np.array(mesh.center, dtype=float)
        bounds = np.array(mesh.bounds).reshape(3, 2)
        extents = np.ptp(bounds, axis=1)
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

    def _on_two_point_plane(self):
        """Enter two-point plane capture mode: orbit freely, then Shift+click two
        points to define a cut plane parallel to the view direction."""
        if self._twopt_active:
            self._end_two_point()      # click while armed = cancel (real toggle)
            self.status.showMessage("2-Point Plane cancelled.")
            return
        if self.engine.original_mesh is None:
            self.btn_two_point_plane.setChecked(False)   # don't look armed when we aren't
            self.status.showMessage("Load an STL first.")
            return
        for btn in (getattr(self, "_btn_select", None), getattr(self, "_btn_trim", None)):
            if btn is not None and btn.isChecked():
                btn.setChecked(False)                    # exit select/trim so styles never stack
        self._cancel_clip_widgets()
        self._twopt_active = True
        self._twopt_first = None
        self.plotter.remove_actor("twopt_marker", render=False)
        iren = self.plotter.iren
        self._twopt_saved_style = iren.style or iren.interactor.GetInteractorStyle()
        self._twopt_style = _TwoPointStyle(self)
        iren.style = self._twopt_style
        self.btn_two_point_plane.setChecked(True)
        self.status.showMessage(
            "2-Point Plane armed: Shift+click the FIRST point on the surface "
            "(then a second; Esc cancels).")

    def _screen_to_focal_world(self, x, y):
        """Back-project display pixel (x, y) onto the camera focal plane -> world xyz.
        Depth is irrelevant here since the cut plane is parallel to the view axis."""
        ren = self.plotter.renderer
        fp = np.asarray(self.plotter.camera.focal_point, dtype=float)
        ren.SetWorldPoint(fp[0], fp[1], fp[2], 1.0)
        ren.WorldToDisplay()
        z = ren.GetDisplayPoint()[2]
        ren.SetDisplayPoint(float(x), float(y), z)
        ren.DisplayToWorld()
        w = np.asarray(ren.GetWorldPoint(), dtype=float)
        return w[:3] / w[3]

    def _twopt_click(self, pos):
        """Handle one Shift+click during two-point capture."""
        if not self._twopt_active:
            return
        world = self._screen_to_focal_world(pos[0], pos[1])
        if self._twopt_first is None:
            self._twopt_first = world
            b = np.asarray(self.engine.original_mesh.bounds, dtype=float)
            r = 0.01 * float(np.linalg.norm(b[1::2] - b[0::2]))
            self.plotter.add_mesh(pv.Sphere(radius=r, center=world),
                                  color="yellow", name="twopt_marker", reset_camera=False)
            self.status.showMessage("Point 1 set — Shift+click the second point.")
            return
        res = plane_from_two_points(self._twopt_first, world, self.plotter.camera.direction)
        if res is None:
            self.status.showMessage("Pick two distinct points.")
            return
        self._current_plane_origin, self._current_plane_normal = res
        self._end_two_point()
        self._update_preview()
        self.status.showMessage("Plane set — press ✂ Cut.")

    def _end_two_point(self):
        """Exit two-point capture: clear state + marker, restore the trackball style."""
        self._twopt_active = False
        self._twopt_first = None
        if hasattr(self, "btn_two_point_plane"):
            self.btn_two_point_plane.setChecked(False)
        self.plotter.remove_actor("twopt_marker", render=False)
        if getattr(self, "_twopt_saved_style", None) is not None:
            self.plotter.iren.style = self._twopt_saved_style
        self.plotter.render()                   # clear the marker immediately (e.g. Esc-cancel)

    def _update_preview(self):
        """Show a cyan slice preview (clipped to box) and a green normal arrow."""
        wall = self.engine.get_wall_mesh()
        if wall is None or wall.n_cells == 0:
            return
        # A lingering tree-highlight wireframe visually mixes with the preview;
        # drop it once the user starts placing a new plane.
        self.plotter.remove_actor("tree_highlight", render=False)

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
            np.ptp(np.array(wall.bounds).reshape(3, 2), axis=1)
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
        existing_names = set(self.engine.patch_names.values())
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

        result = self.engine.clip_and_name(name, origin, normal, box_planes_data)
        if result is None:
            QMessageBox.warning(
                self, "Clip Error",
                "Clip produced no surface — the plane may not intersect the geometry.")
            return

        self._cancel_clip_widgets()
        self._refresh_display()
        self._refresh_patch_list()
        self._refresh_object_tree()
        self._update_status()
        self._update_button_states()
        self.status.showMessage(f"Clipped and named patch '{name}'. Ctrl+Z to undo.")

    def _on_cut(self):
        if self.engine.original_mesh is None:
            return
        box = self._current_box_planes_data
        if box:
            result = self.engine.cut_by_box(box)
        elif self._current_plane_origin is not None and self._current_plane_normal is not None:
            result = self.engine.cut_by_plane(self._current_plane_origin, self._current_plane_normal)
        else:
            self.status.showMessage("Position a cut plane/box first.")
            return
        if result is None:
            self.status.showMessage("Cut did not intersect the surface.")
            return
        self._cancel_clip_widgets()
        self._refresh_display()
        self._update_status()
        self._update_button_states()
        self.status.showMessage(f"Cut applied — {len(self.engine._feature_curves)} feature curve(s).")

    # ------------------------------------------------------------------
    # Patch management
    # ------------------------------------------------------------------

    def _on_rename(self):
        row = self.patch_list.currentRow()
        named = self.engine.named_patches()
        if row < 0 or row >= len(named):
            return
        self._rename_patch_dialog(named[row][0])

    def _on_delete(self):
        row = self.patch_list.currentRow()
        named = self.engine.named_patches()
        if row < 0 or row >= len(named):
            return
        self._remove_patch_by_pid(named[row][0])

    # ------------------------------------------------------------------
    # Trim region (freehand lasso)
    # ------------------------------------------------------------------

    def _begin_select_lasso(self, on_release):
        """Install the shared Shift+drag lasso style. on_release(display_points)
        is invoked when a lasso stroke completes — Select passes _apply_select,
        Trim passes _apply_trim. Plain drag rotates the camera (trackball)."""
        if getattr(self, "_twopt_active", False):
            self._end_two_point()               # leave two-point capture before installing lasso
        iren = self.plotter.iren
        if isinstance(iren.style, _SelectLassoStyle):
            self._end_select_lasso()            # never stack two lasso styles
        self._lasso_points = []
        self._lasso_on_release = on_release
        # Install via pyvista's style setter (which updates iren._style_class) so
        # the style survives pyvista's update_style() re-assertion on the next
        # render. Installing with the raw vtk SetInteractorStyle left _style_class
        # pointing at the trackball, which got re-asserted and made the FIRST drag
        # rotate instead of lasso.
        self._select_saved_style = iren.style or iren.interactor.GetInteractorStyle()
        self._select_style = _SelectLassoStyle(self)
        iren.style = self._select_style

    def _end_select_lasso(self):
        if getattr(self, "_select_saved_style", None) is not None:
            self.plotter.iren.style = self._select_saved_style
        self._end_lasso_overlay()

    def _select_lasso_press(self):
        self._lasso_points = [self.plotter.iren.get_event_position()]
        self._start_lasso_overlay()
        self._update_lasso_overlay()

    def _select_lasso_move(self):
        self._lasso_points.append(self.plotter.iren.get_event_position())
        self._update_lasso_overlay()

    def _select_lasso_release(self):
        self._end_lasso_overlay()
        points = list(self._lasso_points)
        self._lasso_points = []
        if len(points) >= 3:
            self._lasso_on_release(points)

    def _toggle_trim_mode(self, checked):
        if checked:
            if getattr(self, "_btn_select", None) is not None and self._btn_select.isChecked():
                self._btn_select.setChecked(False)
            self._begin_select_lasso(self._apply_trim)
            self.status.showMessage(
                "Trim mode: Shift+drag to lasso a region to delete (through-model); "
                "drag to rotate. Toggle off to exit."
            )
        else:
            self._end_select_lasso()
            self.status.showMessage("Trim mode off.")

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
        pts = self._lasso_points
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
        if len(display_points) < 3:
            self.status.showMessage("Trim: stroke too short — draw a closed shape.")
            return
        cam = self.plotter.camera
        size = self.plotter.render_window.GetSize()
        width, height = int(size[0]), int(size[1])
        aspect = width / height if height else 1.0
        vtk_m = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
        matrix = np.array([[vtk_m.GetElement(i, j) for j in range(4)] for i in range(4)])
        try:
            result = self.engine.trim_by_screen_polygon(display_points, matrix, (width, height))
        except ValueError as exc:
            self.status.showMessage(str(exc))
            return
        if result is None:
            self.status.showMessage("Trim: nothing selected.")
            return
        self._clear_selection()
        self._refresh_display()
        self.status.showMessage(f"Trimmed region. Mesh now {result.n_cells} cells. Ctrl+Z to undo.")

    def _undo_trim(self):
        if self.engine.undo_trim():
            self._clear_selection()
            self._refresh_display()
            # The registry may have changed (clip/fill/split/remove undone):
            # every patch-derived panel must follow.
            self._refresh_patch_list()
            self._refresh_object_tree()
            self._update_status()
            self._update_button_states()
            self.status.showMessage("Undid last operation.")
        else:
            self.status.showMessage("Nothing to undo.")

    # ------------------------------------------------------------------
    # Selection (Select faces / Grow / Smooth)
    # ------------------------------------------------------------------

    def _edit_enabled(self):
        """Every operation compounds on the single current mesh."""
        return self.engine.current_mesh is not None

    def _toggle_select_mode(self, checked):
        self._select_mode = checked
        if checked:
            if not self._edit_enabled():
                self._btn_select.setChecked(False)
                self.status.showMessage("Clear clips to select/edit the base mesh.")
                return
            if getattr(self, "_btn_trim", None) is not None and self._btn_trim.isChecked():
                self._btn_trim.setChecked(False)
            self._begin_select_lasso(self._apply_select)
            self.status.showMessage(
                "Select mode: Shift+drag to lasso faces (adds to selection); "
                "drag to rotate. Esc clears."
            )
        else:
            self._end_select_lasso()
            self.status.showMessage("Select mode off.")

    def _apply_select(self, display_points):
        if len(display_points) < 3:
            return
        cam = self.plotter.camera
        size = self.plotter.render_window.GetSize()
        width, height = int(size[0]), int(size[1])
        aspect = width / height if height else 1.0
        vtk_m = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
        matrix = np.array([[vtk_m.GetElement(i, j) for j in range(4)] for i in range(4)])
        view_dir = np.asarray(self.plotter.camera.direction, dtype=float)
        # Through-model, like Trim: select every face under the lasso polygon —
        # front-facing AND the ones behind / hidden from this view.
        ids = self.engine.select_cells_in_polygon(
            display_points, matrix, (width, height), view_dir, front_only=False)
        self._selection |= set(int(i) for i in ids)
        self._refresh_selection_highlight()
        self.status.showMessage(f"Selected {len(self._selection)} faces.")
        self._update_button_states()

    def _pick_wall_cell(self, pos):
        """Cell id (in current_mesh space) of the surface face under display
        position `pos`, or None. The picker is restricted to the surface actors
        — wall AND named caps — so it cannot catch the centerline or feature
        curves, but a click on a capped outlet picks the cap face instead of
        missing (or hitting the wall hidden behind it). Shared by single-click
        and double-click selection."""
        mesh = self.engine.original_mesh
        actors = getattr(self, "_pickable_surface_actors", None)
        if actors is None:                       # display not built via helper yet
            wall_actor = getattr(self, "_wall_actor", None)
            actors = ([(wall_actor, getattr(self, "_wall_cell_map", None))]
                      if wall_actor is not None else [])
        if mesh is None or not actors:
            return None
        picker = vtk.vtkCellPicker()
        picker.InitializePickList()
        for actor, _ in actors:
            picker.AddPickList(actor)
        picker.PickFromListOn()
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        cid = picker.GetCellId()
        if cid is None or cid < 0:
            return None
        # Translate the hit actor's local cell id back into current_mesh space
        # (each subset actor carries its own original-id map; None = identity).
        hit = picker.GetActor()
        cmap = None
        for actor, m in actors:
            if actor is hit:
                cmap = m
                break
        if cmap is not None:
            if cid >= len(cmap):
                return None
            cid = int(cmap[cid])
        if cid >= mesh.n_cells:
            return None
        return int(cid)

    def _select_click_pick(self, pos):
        """A single click in Select mode adds the one face under the cursor to the
        active selection."""
        if not self._btn_select.isChecked() or not self._edit_enabled():
            return
        cid = self._pick_wall_cell(pos)
        if cid is None:
            return
        self._selection.add(cid)
        self._refresh_selection_highlight()
        self.status.showMessage(f"Selected {len(self._selection)} faces.")
        self._update_button_states()

    def _select_double_click(self, pos):
        """A double click in Select mode floods the connected surface region under the
        cursor (bounded by feature curves) into the active selection."""
        if not self._btn_select.isChecked() or not self._edit_enabled():
            return
        cid = self._pick_wall_cell(pos)
        if cid is None:
            return
        region = self.engine.flood_select(cid)
        self._selection |= set(region)
        self._refresh_selection_highlight()
        self.status.showMessage(
            f"Flood-selected {len(region)} faces ({len(self._selection)} total).")
        self._update_button_states()

    @staticmethod
    def _pull_to_front(actor):
        """Depth-bias an overlay actor toward the camera so it always wins the
        depth test against the coplanar base surface. Selection/highlight actors
        redraw the SAME triangles as the base mesh; without this bias the two
        actors tie in the depth buffer and flicker while the camera moves
        (z-fighting). Relative coincident-topology offset shifts depth only —
        no visible geometry change. Same mechanism ParaView uses for selection."""
        if actor is None:
            return
        m = actor.GetMapper()
        if m is not None:
            m.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -66000.0)
            m.SetRelativeCoincidentTopologyLineOffsetParameters(-2.0, -66000.0)

    def _draw_selection_actor(self):
        """(Re)draw the actor holding ONLY the selected faces — opaque orange.
        The same faces are excluded from the wall/cap actors, so this is the
        single on-screen copy: visually a recolor of the selected region."""
        try:
            self.plotter.remove_actor("selection", render=False)
        except Exception:
            pass
        mesh = self.engine.original_mesh
        if self._selection and mesh is not None:
            ids = sorted(i for i in self._selection if 0 <= i < mesh.n_cells)
            if ids:
                actor = self.plotter.add_mesh(
                    mesh.extract_cells(ids), color=(1.0, 0.55, 0.0),
                    name="selection", lighting=True, pickable=False,
                    reset_camera=False)
                self._pull_to_front(actor)

    def _draw_surface_actors(self, exclude=None):
        """Draw/replace the wall + named-cap actors, excluding `exclude` cell ids
        (the active selection). Excluded faces are rendered ONLY by the selection
        actor: no duplicate coincident geometry to shade twice, and the
        translucent wall can no longer composite over the highlight — the two
        causes of the confusing 'double mesh' selection look."""
        mesh = self.engine.current_mesh
        if mesh is None or mesh.n_cells == 0:
            return
        n = mesh.n_cells
        excl = {int(i) for i in (exclude or ()) if 0 <= int(i) < n}
        if (PATCH_ID in mesh.cell_data
                and len(mesh.cell_data[PATCH_ID]) == n):
            labels = np.asarray(mesh.cell_data[PATCH_ID])
        else:
            labels = np.zeros(n, dtype=np.int64)
        keep = np.ones(n, dtype=bool)
        if excl:
            keep[list(excl)] = False
        named = self.engine.named_patches()
        show_mesh = self._btn_show_mesh_edges.isChecked()
        wall_opacity = 1.0 if self._btn_opaque_wall.isChecked() else 0.4

        # Wall (patch 0). Fast path: pristine mesh, nothing excluded -> whole mesh.
        self._wall_cell_map = None       # subset-actor cell id -> current_mesh cell id
        if not excl and not named:
            wall = mesh
        else:
            wall_idx = np.nonzero((labels == 0) & keep)[0]
            wall = mesh.extract_cells(wall_idx).extract_surface() if len(wall_idx) else None
            if wall is not None and "vtkOriginalCellIds" in wall.cell_data:
                # extract_surface may reorder; its original-ids array indexes into
                # the extract_cells order, which is wall_idx's order.
                sub = np.asarray(wall.cell_data["vtkOriginalCellIds"])
                self._wall_cell_map = wall_idx[sub]
            elif wall is not None:
                self._wall_cell_map = wall_idx
        pickable = []                            # [(actor, cell_map or None)]
        if wall is not None and wall.n_cells > 0:
            self._wall_actor = self.plotter.add_mesh(
                wall, color=WALL_COLOR, opacity=wall_opacity,
                show_edges=show_mesh, edge_color="black", line_width=0.5,
                specular=0.15, specular_power=20.0, ambient=0.15, diffuse=0.9,
                name="wall", reset_camera=False,
            )
            pickable.append((self._wall_actor, self._wall_cell_map))
        else:                                   # everything selected -> no wall actor
            self.plotter.remove_actor("wall", render=False)
            self._wall_actor = None

        # Named patches — flat name-based colors. No edge lines: caps are sliver
        # triangle fans over jagged rims, and white edges shred the solid color.
        drawn = set()
        for pid, pname, _nf in named:
            idx = np.nonzero((labels == pid) & keep)[0]
            if len(idx) == 0:
                continue                        # fully selected -> selection draws it
            cap = mesh.extract_cells(idx).extract_surface()
            aname = f"cap_{pname}"
            cap_actor = self.plotter.add_mesh(
                cap, color=_color_for_name(pname), opacity=1.0,
                name=aname, reset_camera=False,
            )
            # Same original-id mapping as the wall: extract_surface may reorder,
            # and its original-ids index into the extract_cells order (= idx).
            if "vtkOriginalCellIds" in cap.cell_data:
                cmap = idx[np.asarray(cap.cell_data["vtkOriginalCellIds"])]
            else:
                cmap = idx
            pickable.append((cap_actor, cmap))
            drawn.add(aname)
        for stale in getattr(self, "_cap_actor_names", set()) - drawn:
            self.plotter.remove_actor(stale, render=False)
        self._cap_actor_names = drawn
        # Every surface actor is click-pickable — named caps included, so
        # Select mode works on a capped outlet, not only on the wall.
        self._pickable_surface_actors = pickable

    def _refresh_selection_highlight(self):
        self._draw_selection_actor()
        self._draw_surface_actors(exclude=self._selection)
        self.plotter.render()

    def _refresh_object_tree(self):
        """Rebuild the object tree from detected surface entities. Identity-guarded:
        skips recompute when the mesh is unchanged since the last build, so view-only
        refreshes stay free."""
        mesh = self.engine.current_mesh
        patch_sig = tuple(sorted(self.engine.patch_names.items()))
        if mesh is self._tree_mesh and patch_sig == self._tree_n_patches:
            return
        self._tree_mesh = mesh
        self._tree_n_patches = patch_sig
        self._active_profile_index = None       # stale once the profile list is rebuilt
        self._tree_profiles = self.engine.unfilled_open_profiles()
        self._tree_nonmanifold = self.engine.detect_nonmanifold_edges()
        self._tree_pieces = self.engine.detect_pieces()
        self._tree_patches = self.engine.named_patches()
        tree = self._object_tree
        tree.clear()

        prof = QTreeWidgetItem(tree, ["Open Profiles"])
        prof.setExpanded(True)
        if self._tree_profiles:
            for i, g in enumerate(self._tree_profiles):
                it = QTreeWidgetItem(prof, [f"Open Profile {i + 1} ({g.n_cells} edges)"])
                it.setData(0, Qt.UserRole, ("profile", i))
        else:
            QTreeWidgetItem(prof, ["(none)"]).setDisabled(True)

        nm = QTreeWidgetItem(tree, ["Non-manifold edges"])
        nm.setExpanded(True)
        if self._tree_nonmanifold:
            for i, g in enumerate(self._tree_nonmanifold):
                it = QTreeWidgetItem(nm, [f"Non-manifold group {i + 1} ({g.n_cells} edges)"])
                it.setData(0, Qt.UserRole, ("nonmanifold", i))
        else:
            QTreeWidgetItem(nm, ["(none)"]).setDisabled(True)

        pc = QTreeWidgetItem(tree, ["Disconnected pieces"])
        pc.setExpanded(True)
        if self._tree_pieces:
            for i, cells in enumerate(self._tree_pieces):
                it = QTreeWidgetItem(pc, [f"Piece {i + 1} ({len(cells)} faces)"])
                it.setData(0, Qt.UserRole, ("piece", i))
        else:
            QTreeWidgetItem(pc, ["(none)"]).setDisabled(True)

        pt = QTreeWidgetItem(tree, ["Named patches"])
        pt.setExpanded(True)
        if self._tree_patches:
            for pid, pname, nfaces in self._tree_patches:
                it = QTreeWidgetItem(pt, [f"{pname} ({nfaces} faces)"])
                it.setData(0, Qt.UserRole, ("patch", pid))
        else:
            QTreeWidgetItem(pt, ["(none)"]).setDisabled(True)

    def _on_tree_item_clicked(self, item, column):
        """Highlight the clicked entity (no camera move). Pieces also load into the
        face selection so Delete faces removes them."""
        data = item.data(0, Qt.UserRole)
        self.plotter.remove_actor("tree_highlight", render=False)
        if data is None:                                  # category header or (none)
            self.plotter.render()
            return
        kind, index = data
        if kind == "profile":
            geom = self._tree_profiles[index]
            self._pull_to_front(self.plotter.add_mesh(
                geom, color="orange", line_width=6,
                name="tree_highlight", reset_camera=False))
            self._active_profile_index = index
            # Also seed the face selection from the rim faces so Grow/Smooth/Delete
            # work from an edge — Grow then expands inward to the surrounding
            # surface (Fill still uses _active_profile_index, unaffected).
            cells = self.engine.faces_on_edges(geom)
            if cells:
                self._selection = set(cells)
                self._refresh_selection_highlight()
                self._update_button_states()
            self.status.showMessage(
                f"Open Profile {index + 1} — {geom.n_cells} edges, "
                f"{len(cells)} rim faces selected. Grow to expand.")
        elif kind == "nonmanifold":
            geom = self._tree_nonmanifold[index]
            self._pull_to_front(self.plotter.add_mesh(
                geom, color="red", line_width=6,
                name="tree_highlight", reset_camera=False))
            # Load the attached faces into the selection so Grow/Smooth/Delete
            # work from here — growing a few rings makes a tiny edge findable.
            cells = self.engine.faces_on_edges(geom)
            if cells:
                self._selection = set(cells)
                self._refresh_selection_highlight()
                self._update_button_states()
            self.status.showMessage(
                f"Non-manifold group {index + 1} — {geom.n_cells} edges, "
                f"{len(cells)} attached faces selected. Grow to see the area.")
        elif kind == "piece":
            cells = self._tree_pieces[index]
            self._selection = set(cells)
            self._refresh_selection_highlight()
            self.status.showMessage(
                f"Piece {index + 1} — {len(cells)} faces selected. Delete faces to remove.")
            self._update_button_states()
        elif kind == "patch":
            cap = self.engine.patches_by_id().get(index)   # index carries the patch_id
            pname = self.engine.patch_name_for(index)
            if cap is not None and cap.n_cells > 0:
                # Wireframe overlay: a solid highlight would z-fight with the
                # already-drawn colored cap (identical faces) and look broken.
                self._pull_to_front(self.plotter.add_mesh(
                    cap, color="yellow", style="wireframe",
                    line_width=4, name="tree_highlight",
                    reset_camera=False))
                self.status.showMessage(f"Patch '{pname}' — {cap.n_cells} faces.")
        self.plotter.render()

    def _init_splitter_sizes(self):
        """Give each side panel its content-sized width; the viewport gets the rest."""
        total = self._splitter.width()
        if total <= 0:
            return
        left = max(self._splitter.widget(0).sizeHint().width(), 220)
        right = max(self._tab_widget.sizeHint().width(), 300)
        right = min(right, int(total * 0.30))       # never let the panel eat the viewport
        center = max(total - left - right, 400)
        self._splitter.setSizes([left, center, right])

    def _on_tree_context_menu(self, pos):
        """Right-click menu on Objects tree entries. Named patches get
        Rename / Split disconnected / Remove name; everything gets Highlight."""
        item = self._object_tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        kind, index = data
        menu = QMenu(self._object_tree)
        act_hl = menu.addAction("Highlight")
        act_rename = act_split = act_remove = act_fill = act_del_piece = None
        act_remesh = act_del_nm = act_fill_wall = act_extrude = None
        if kind == "patch":
            act_rename = menu.addAction("Rename…")
            act_split = menu.addAction("Split disconnected components")
            act_remesh = menu.addAction("Remesh (smoother cap)")
            act_remove = menu.addAction("Remove name (faces → wall)")
        elif kind == "profile":
            act_fill_wall = menu.addAction("Fill (merge into wall)")
            act_fill = menu.addAction("Fill as named patch…")
            act_extrude = menu.addAction("Extrude (flow extension)…")
        elif kind == "piece":
            act_del_piece = menu.addAction("Delete piece (faces)")
        elif kind == "nonmanifold":
            act_del_nm = menu.addAction("Delete attached faces")
        chosen = menu.exec_(self._object_tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_hl:
            self._on_tree_item_clicked(item, 0)
        elif chosen is act_rename:
            self._rename_patch_dialog(index)
        elif chosen is act_split:
            self._split_patch_from_tree(index)
        elif chosen is act_remove:
            self._remove_patch_by_pid(index)
        elif chosen is act_fill_wall:
            self._fill_profile_as_wall(index)
        elif chosen is act_fill:
            self._fill_profile_dialog(index)
        elif chosen is act_extrude:
            self._extrude_profile_dialog(index)
        elif chosen is act_del_piece:
            self._delete_piece(index)
        elif chosen is act_remesh:
            self._remesh_patch_from_tree(index)
        elif chosen is act_del_nm:
            self._delete_nonmanifold_group(index)

    def _remesh_patch_from_tree(self, pid):
        """Rebuild a cap patch with well-shaped triangles (constrained Delaunay)."""
        name = self.engine.patch_name_for(pid)
        out = self.engine.remesh_patch(pid)
        if out is None:
            self.status.showMessage(
                f"Could not remesh '{name}' (needs a rim to triangulate).")
            return
        self.plotter.remove_actor("tree_highlight", render=False)
        self._refresh_display()
        self._refresh_object_tree()
        self._refresh_patch_list()
        self._update_status()
        self.status.showMessage(f"Remeshed '{name}'. Ctrl+Z to undo.")

    def _delete_nonmanifold_group(self, index):
        """Delete the faces attached to one non-manifold edge group (undoable)."""
        groups = self._tree_nonmanifold
        if not (0 <= index < len(groups)):
            return
        cells = self.engine.faces_on_edges(groups[index])
        if not cells:
            self.status.showMessage("No faces found on those edges.")
            return
        result = self.engine.delete_cells(cells)
        if result is None:
            self.status.showMessage("Cannot delete those faces (would empty the mesh).")
            return
        self._clear_selection()
        self._refresh_display()
        self._refresh_object_tree()
        self._update_status()
        self.status.showMessage(
            f"Deleted {len(cells)} faces on non-manifold edges. Ctrl+Z to undo.")

    def _delete_piece(self, index):
        """Delete a disconnected piece's faces from the current mesh (undoable)."""
        pieces = self._tree_pieces
        if not (0 <= index < len(pieces)):
            return
        cells = pieces[index]
        result = self.engine.delete_cells(cells)
        if result is None:
            self.status.showMessage(
                "Cannot delete this piece (it may be the whole mesh).")
            return
        self._clear_selection()
        self._refresh_display()
        self._refresh_object_tree()
        self._update_status()
        self.status.showMessage(
            f"Deleted piece ({len(cells)} faces). Ctrl+Z to undo.")

    def _rename_patch_dialog(self, pid):
        """Prompt for a new name for patch `pid` and apply it everywhere."""
        if pid not in self.engine.patch_names:
            return
        old_name = self.engine.patch_names[pid]
        new_name, ok = QInputDialog.getText(
            self, "Rename Patch", "New name:", text=old_name,
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        existing = {n for p, n in self.engine.patch_names.items() if p != pid}
        if new_name in existing or new_name == "wall":
            QMessageBox.warning(self, "Duplicate Name", f"'{new_name}' is already used.")
            return
        self.engine.rename_patch(pid, new_name)
        self._centerline_mesh = None  # invalidate — inlet/outlet classification may have changed
        self._refresh_patch_list()
        self._refresh_object_tree()
        self._refresh_display()
        self._update_status()

    def _split_patch_from_tree(self, pid):
        """Split a patch into its disconnected components (bifurcation case)."""
        name = self.engine.patch_name_for(pid)
        new_pids = self.engine.split_patch(pid)
        if new_pids is None:
            self.status.showMessage(
                f"'{name}' is a single connected piece — nothing to split.")
            return
        self.plotter.remove_actor("tree_highlight", render=False)
        self._refresh_display()
        self._refresh_object_tree()
        self._refresh_patch_list()
        self._update_status()
        self._update_button_states()
        names = ", ".join(self.engine.patch_names[p] for p in new_pids)
        self.status.showMessage(
            f"Split '{name}' into {len(new_pids)}: {names}. "
            f"Right-click each to rename. Ctrl+Z to undo.")

    def _remove_patch_by_pid(self, pid):
        """Un-name a patch (faces → wall) after confirmation."""
        name = self.engine.patch_name_for(pid)
        reply = QMessageBox.question(
            self, "Remove Patch Name",
            f"Un-name patch '{name}'? Its faces become wall again "
            f"(geometry is kept; Ctrl+Z to restore the name).",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.engine.remove_patch(pid)
            self._centerline_mesh = None  # invalidate — patch set changed
            self._refresh_display()
            self._refresh_patch_list()
            self._refresh_object_tree()
            self._update_status()
            self._update_button_states()

    def _clear_selection(self):
        self._selection = set()
        self._refresh_selection_highlight()
        self._update_button_states()

    def _on_grow_selection(self):
        if not self._selection or not self._edit_enabled():
            return
        self._selection = set(self.engine.grow_cells(self._selection, rings=1))
        self._refresh_selection_highlight()
        self.status.showMessage(f"Grown to {len(self._selection)} faces.")

    def _show_select_context_menu(self):
        """Right-click menu in Select mode (popped by _SelectLassoStyle on a
        right-CLICK; right-drag still zooms). Operates on the current selection:
        grow it to the surrounding sharp-edge boundary, or assign it as a named
        patch — the click-a-cap -> outlet1 workflow."""
        if not getattr(self, "_select_mode", False):
            return
        if not self._selection:
            self.status.showMessage("Click or lasso a face first, then right-click.")
            return
        menu = QMenu(self)
        act_sharp = menu.addAction("Select to sharp edges (30°)")
        act_sharp_custom = menu.addAction("Select to sharp edges (custom angle…)")
        menu.addSeparator()
        act_assign = menu.addAction("Assign selection as patch…")
        act_clear = menu.addAction("Clear selection")
        chosen = menu.exec_(QCursor.pos())
        if chosen is act_sharp:
            self._select_to_sharp(30.0)
        elif chosen is act_sharp_custom:
            angle, ok = QInputDialog.getDouble(
                self, "Sharp Edge Angle",
                "Stop at edges sharper than (degrees):", 30.0, 1.0, 179.0, 1)
            if ok:
                self._select_to_sharp(float(angle))
        elif chosen is act_assign:
            self._assign_selection_dialog()
        elif chosen is act_clear:
            self._clear_selection()
            self.status.showMessage("Selection cleared.")

    def _select_to_sharp(self, angle_deg):
        """Grow the selection to the sharp-edge boundary around it."""
        grown = self.engine.select_to_sharp_edges(self._selection, angle_deg)
        if not grown:
            self.status.showMessage("Nothing to grow — select a face first.")
            return
        self._selection = set(grown)
        self._refresh_selection_highlight()
        self._update_button_states()
        self.status.showMessage(
            f"Selected {len(grown)} faces bounded by edges sharper than {angle_deg:g}°.")

    def _assign_selection_dialog(self):
        """Name the current selection as a new patch (outlet1, outlet2, …)."""
        if not self._selection:
            return
        used = set(self.engine.patch_names.values()) | {"wall"}
        n = 1
        while f"outlet{n}" in used:
            n += 1
        default = f"outlet{n}"
        name, ok = QInputDialog.getText(
            self, "Patch Name", "Name for the selected faces:", text=default)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in used:
            QMessageBox.warning(self, "Duplicate Name",
                                f"'{name}' is already used. Choose another.")
            return
        pid = self.engine.assign_patch_from_cells(self._selection, name)
        if pid is None:
            self.status.showMessage("Could not assign the selection as a patch.")
            return
        count = len(self._selection)
        self._clear_selection()
        self._refresh_display()
        self._refresh_object_tree()
        self._refresh_patch_list()
        self._update_button_states()
        self.status.showMessage(
            f"Assigned {count} faces as patch '{name}'. Ctrl+Z to undo.")

    def _on_remesh_selection(self):
        """Remesh the selected region: delete + re-triangulate the rim (fixes a
        non-manifold junction locally). One undo step."""
        if not self._selection or not self._edit_enabled():
            return
        msg = self.engine.remesh_region(self._selection)
        if msg is None:
            self.status.showMessage("Could not remesh that selection.")
            return
        self._clear_selection()
        self.plotter.remove_actor("tree_highlight", render=False)
        self._refresh_display()
        self._refresh_object_tree()
        self._refresh_patch_list()
        self._update_status()
        self._update_button_states()
        self.status.showMessage(f"{msg}. Ctrl+Z to undo.")

    def _on_smooth_selection(self):
        if not self._selection or not self._edit_enabled():
            return
        result = self.engine.smooth_cells(self._selection, iterations=5, relaxation=0.5)
        if result is None:
            self.status.showMessage("Smooth: nothing selected.")
            return
        self._refresh_display()                 # rebuilds the wall (and clears the highlight actor)
        self._refresh_selection_highlight()     # ids still valid (topology unchanged)
        self.status.showMessage(f"Smoothed {len(self._selection)} faces (\xd75). Ctrl+Z to undo.")

    def _on_delete_selection(self):
        if not self._selection or not self._edit_enabled():
            return
        result = self.engine.delete_cells(self._selection)
        if result is None:
            self.status.showMessage("Nothing deleted.")
            return
        self._clear_selection()              # deleted ids no longer exist
        self._refresh_display()
        n = result.n_cells if result is not None else 0
        self.status.showMessage(f"Deleted faces — wall now {n:,} faces.")

    def _on_fill_profile(self):
        """Fill the open profile selected in the Objects tree into a named patch."""
        idx = self._active_profile_index
        if idx is None:
            self.status.showMessage("Select an open profile in the Objects tree first.")
            return
        self._fill_profile_dialog(idx)

    def _extrude_profile_dialog(self, idx):
        """Flow extension: prompt for a length and extrude open profile `idx`
        along its cut-plane normal. Default length = 3x equivalent diameter."""
        if self.engine.current_mesh is None:
            self.status.showMessage("Load an STL first.")
            return
        profiles = self._tree_profiles
        if not (0 <= idx < len(profiles)):
            self.status.showMessage("That open profile is stale — refreshing the tree.")
            self._refresh_object_tree()
            return
        rim = profiles[idx]
        pts = np.asarray(rim.points, dtype=float)
        diameter = 2.0 * float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())
        default = max(3.0 * diameter, 1e-6)
        length, ok = QInputDialog.getDouble(
            self, "Flow Extension",
            "Extension length (mesh units):", default, 1e-6, 1e9, 3)
        if not ok:
            return
        msg = self.engine.extrude_profile(rim, float(length))
        if msg is None:
            self.status.showMessage("Could not extrude this profile.")
            return
        self._active_profile_index = None
        self.plotter.remove_actor("tree_highlight", render=False)
        self._refresh_display()
        self._refresh_object_tree()
        self._refresh_patch_list()
        self._update_button_states()
        self.status.showMessage(f"{msg} Ctrl+Z to undo.")

    def _fill_profile_as_wall(self, idx):
        """Quick-fill open profile `idx` and merge the cap into the wall — no name
        prompt, no new patch (right-click 'Fill (merge into wall)')."""
        if self.engine.current_mesh is None:
            self.status.showMessage("Load an STL first.")
            return
        profiles = self._tree_profiles
        if not (0 <= idx < len(profiles)):
            self.status.showMessage("That open profile is stale — refreshing the tree.")
            self._refresh_object_tree()
            return
        result = self.engine.fill_profile(profiles[idx])          # name=None -> wall
        if result is None:
            self.status.showMessage("Could not fill this profile.")
            return
        self._active_profile_index = None
        self._refresh_display()
        self._refresh_object_tree()
        self._refresh_patch_list()
        self._update_button_states()
        self.status.showMessage("Filled profile into wall. Ctrl+Z to undo.")

    def _fill_profile_dialog(self, idx):
        """Prompt for a patch name and fill open profile `idx` (shared by the
        right-panel Fill button and the tree's right-click menu)."""
        if self.engine.current_mesh is None:
            self.status.showMessage("Load an STL first.")
            return
        profiles = self._tree_profiles
        if not (0 <= idx < len(profiles)):
            self.status.showMessage("That open profile is stale — refreshing the tree.")
            self._refresh_object_tree()
            return
        used = set(self.engine.patch_names.values()) | {"wall"}
        if "inlet" not in used:
            default = "inlet"
        else:
            n = 1
            while f"outlet_{n}" in used:
                n += 1
            default = f"outlet_{n}"
        name, ok = QInputDialog.getText(self, "Patch Name", "Name for this patch:", text=default)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in used:
            QMessageBox.warning(self, "Duplicate Name", f"'{name}' is already used. Choose another.")
            return
        result = self.engine.fill_profile(profiles[idx], name)
        if result is None:
            self.status.showMessage("Could not fill this profile.")
            return
        self._active_profile_index = None
        self._refresh_display()
        self._refresh_object_tree()
        self._refresh_patch_list()
        self._update_button_states()
        self.status.showMessage(f"Filled patch '{name}'. Ctrl+Z to undo.")

    def _on_escape_selection(self):
        if getattr(self, "_twopt_active", False):
            self._end_two_point()
            self.status.showMessage("Two-point plane cancelled.")
            return
        if getattr(self, "_select_mode", False):
            self._btn_select.setChecked(False)   # exits select mode via _toggle_select_mode
        self.plotter.remove_actor("tree_highlight", render=False)  # clear any tree-entity highlight
        self._active_profile_index = None
        self._clear_selection()                  # clears face-selection highlight + renders

    # ------------------------------------------------------------------

    def _refresh_patch_list(self):
        self.patch_list.clear()
        for pid, name, nfaces in self.engine.named_patches():
            r, g, b = [int(c * 255) for c in _color_for_name(name)]
            item = QListWidgetItem(f"{name}  ({nfaces:,} faces)")
            item.setForeground(Qt.black)
            item.setBackground(QColor(r, g, b, 60))
            self.patch_list.addItem(item)
        # Keep the 'Clip & Save STL' tab's list in sync — same source of truth.
        self._refresh_clip_save_list()

    # ------------------------------------------------------------------
    # Centerline
    # ------------------------------------------------------------------

    def _on_compute_centerline(self):
        """Seed VMTK from named-patch centroids on the (already capped) current mesh."""
        capped = self.engine.current_mesh
        if capped is None or capped.n_cells == 0:
            return

        patches = self.engine.patches_by_id()
        source_pts, target_pts = [], []
        for pid, name, _nfaces in self.engine.named_patches():
            cap = patches.get(pid)
            if cap is None or cap.n_cells == 0:
                continue
            kind = openfoam_case.classify_patch(name)
            if kind == "inlet":
                source_pts.extend(list(cap.center))
            elif kind == "outlet":
                target_pts.extend(list(cap.center))

        if not source_pts or not target_pts:
            QMessageBox.warning(self, "Centerline Error",
                                "Need at least one patch named 'inlet' and one named 'outlet'.")
            return
        logger.info("Seed points (patch centroids): source=%s, target=%s", source_pts, target_pts)

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

    def _refresh_display(self, fit_camera=False):
        self.plotter.clear()
        # pyvista's clear() also removes all lights (Renderer.clear ->
        # remove_all_lights), leaving only VTK's fallback headlight, which lights
        # every camera-facing face equally -> a flat, unshaded silhouette.
        # Re-establish a ParaView-style light kit on each redraw so the surface
        # renders with proper shading.
        self.plotter.enable_lightkit()

        if self.engine.current_mesh is None or self.engine.current_mesh.n_cells == 0:
            self.plotter.render()
            return

        # Wall + named-cap actors (selection faces excluded — they are drawn
        # solely by the selection actor), then the selection itself.
        self._cap_actor_names = set()            # plotter.clear() removed them all
        self._draw_surface_actors(exclude=self._selection)
        self._draw_selection_actor()

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
                    name="centerline", reset_camera=False,
                )
                logger.info("Centerline tube added to plotter")
            else:
                logger.warning("Tube generation produced empty mesh — centerline not rendered")

        # Feature curves from cuts. Render as a thin tube (like the centerline) so
        # the curve pokes out from the surface — coincident thin lines z-fight with
        # the opaque wall and are invisible.
        wall_bounds = self.engine.original_mesh.bounds if self.engine.original_mesh is not None else None
        tube_r = 0.001 * float(np.linalg.norm(
            np.array(wall_bounds[1::2]) - np.array(wall_bounds[0::2]))) if wall_bounds else 0.1
        for i, curve in enumerate(self.engine.drawable_feature_curves()):
            if curve is None or curve.n_cells == 0:
                continue
            try:
                tube = curve.tube(radius=tube_r)
            except Exception:
                tube = None
            if tube is not None and tube.n_points > 0:
                self.plotter.add_mesh(tube, color="cyan", name=f"feature_curve_{i}",
                                      reset_camera=False)
            else:
                self.plotter.add_mesh(curve, color="cyan", line_width=2,
                                      name=f"feature_curve_{i}", reset_camera=False)

        if fit_camera:
            self.plotter.reset_camera()
        # pyvista auto-resets the camera to the data bounds the next time an actor
        # is added whenever renderer.camera_set is False (reset_camera() does NOT
        # set that flag). Without this, the FIRST lasso actor-add snapped the view
        # back to the initial load fit. Mark the camera as user-set so that
        # one-time auto-reset never fires.
        self.plotter.renderer.camera_set = True
        self.plotter.render()

        # Update wall face count label — count patch-0 cells on the mesh itself
        # (the true wall size, independent of the display's selection exclusion).
        m = self.engine.current_mesh
        if m is not None and PATCH_ID in m.cell_data and len(m.cell_data[PATCH_ID]) == m.n_cells:
            n_wall = int((np.asarray(m.cell_data[PATCH_ID]) == 0).sum())
        else:
            n_wall = m.n_cells if m is not None else 0
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
        n_profiles = len(self.engine.unfilled_open_profiles())
        self._lbl_open_profiles.setText(f"Open profiles: {n_profiles}")
        self._lbl_open_profiles.setStyleSheet("")
        self._lbl_open_edges.setText(f"Open edges: {quality['open_edges']}")

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

        # Normal-orientation status
        nrm = self.engine.check_normals()
        if nrm["consistent"] is None:
            self._lbl_normals.setText("Normals: \u2014")
            self._lbl_normals.setStyleSheet("color: gray;")
        elif not nrm["consistent"]:
            self._lbl_normals.setText(
                f"Normals: \u2717 {nrm['flipped_edges']} flipped edges (use Fix Normals)")
            self._lbl_normals.setStyleSheet("color: red;")
        elif nrm["outward"] is True:
            self._lbl_normals.setText("Normals: \u2713 consistent, outward")
            self._lbl_normals.setStyleSheet("color: green;")
        elif nrm["outward"] is False:
            self._lbl_normals.setText("Normals: \u2717 consistent but INWARD (use Fix Normals)")
            self._lbl_normals.setStyleSheet("color: red;")
        else:
            self._lbl_normals.setText("Normals: \u2713 consistent (open surface)")
            self._lbl_normals.setStyleSheet("color: green;")
        if hasattr(self, "_btn_fix_normals_quick"):
            self._btn_fix_normals_quick.setVisible(
                nrm["consistent"] is False or nrm["outward"] is False)

        if is_mf is None:
            self._lbl_manifold.setText("Manifold: \u2014 (no named patches)")
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
                name="boundary_edges", reset_camera=False,
            )

        # Re-render non-manifold edges if toggle is on
        if self._btn_show_non_manifold.isChecked() and self._non_manifold_mesh is not None:
            if self._non_manifold_mesh.n_cells > 0:
                self.plotter.add_mesh(
                    self._non_manifold_mesh, color="magenta", line_width=4,
                    name="non_manifold_edges", reset_camera=False,
                )

        self.plotter.render()
        self._refresh_object_tree()

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
                    name="boundary_edges", reset_camera=False,
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
                    name="non_manifold_edges", reset_camera=False,
                )
                self.plotter.render()
        else:
            self.plotter.remove_actor("non_manifold_edges")
            self.plotter.render()

    # ------------------------------------------------------------------
    # View controls
    # ------------------------------------------------------------------

    def _on_zoom_to_fit(self):
        """Reset camera to fit all visible actors."""
        self.plotter.reset_camera()

    def _update_status(self):
        if self.engine.current_mesh is None:
            return
        n = self.engine.current_mesh.n_cells
        names_list = list(self.engine.patch_names.values())
        names = ", ".join(names_list) if names_list else "none"
        self.status.showMessage(
            f"Current mesh: {n:,} faces | {len(names_list)} named patches ({names})")

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
        if not self.engine.patch_names:
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
            patch_names = list(self.engine.patch_names.values())
            files = [f"{n}.stl" for n in patch_names] + ["wall.stl"]
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
        if not self.engine.patch_names:
            return
        case_dir = self._get_or_create_case_dir()
        if not case_dir:
            return
        sf = self._spin_scale.value()
        default_name = "boundary.stl"
        if self._loaded_filepath:
            stem = os.path.splitext(os.path.basename(self._loaded_filepath))[0]
            default_name = f"{stem}_boundary.stl"
        name, ok = QInputDialog.getText(
            self, "Combined STL Name", "File name:", text=default_name,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if not name.lower().endswith(".stl"):
            name += ".stl"
        filepath = os.path.join(case_dir, name)
        try:
            self.engine.export_combined_stl(filepath, scale_factor=sf)
            planes_path = os.path.join(case_dir, "clip_planes.json")
            self.engine.export_clip_planes(planes_path)
            patch_names = list(self.engine.patch_names.values())
            QMessageBox.information(
                self, "Export Complete",
                f"Combined STL saved to:\n{filepath}\n"
                f"Scale factor: ×{sf}\n\n"
                f"Clip planes JSON:\n{planes_path}\n\n"
                f"Patches: {', '.join(patch_names)}, wall",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_save_openfoam_stl(self):
        if not self.engine.patch_names:
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
            patch_names = list(self.engine.patch_names.values())
            patches = ", ".join(patch_names) + ", wall"
            QMessageBox.information(
                self, "Saved",
                f"Multi-solid STL saved to:\n{filepath}\n"
                f"Scale factor: \u00d7{sf}\n\n"
                f"Solids: {patches}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_export_openfoam(self):
        if not self.engine.patch_names or self._loaded_filepath is None:
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
            patch_names = list(self.engine.patch_names.values())
            patches = ", ".join(patch_names) + ", wall"
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


def _log_environment():
    """Log the runtime stack once at startup. Environment divergence (two
    installs, different numpy/vtk generations) has produced GUI-only bugs that
    tests could not see — this header makes any pasted log self-diagnosing."""
    import numpy
    import pyvista
    try:
        from PyQt5.QtCore import PYQT_VERSION_STR
    except Exception:                                    # pragma: no cover
        PYQT_VERSION_STR = "?"
    logger.info("mesh-prep starting")
    logger.info("  python  : %s (%s)", sys.version.split()[0], sys.prefix)
    logger.info("  numpy   : %s", numpy.__version__)
    logger.info("  pyvista : %s", pyvista.__version__)
    logger.info("  vtk     : %s", vtk.vtkVersion.GetVTKVersion())
    logger.info("  pyqt    : %s", PYQT_VERSION_STR)


def _setup_logging(debug: bool):
    """Console logging as before; with --debug also mirror everything to a
    timestamped file under ~/.mesh-prep/logs/. Returns the log path or None."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log_path = None
    if debug:
        log_dir = os.path.join(os.path.expanduser("~"), ".mesh-prep", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, time.strftime("mesh-prep-%Y%m%d-%H%M%S.log"))
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(fh)

    # Uncaught exceptions abort the Qt app (macOS: 'Abort trap: 6') — capture
    # the traceback into every handler (incl. the file) BEFORE that happens.
    previous_hook = sys.excepthook

    def _log_uncaught(exc_type, exc, tb):
        logger.critical("UNCAUGHT EXCEPTION — the app may abort now",
                        exc_info=(exc_type, exc, tb))
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _log_uncaught
    return log_path


def main():
    args = list(sys.argv[1:])
    debug = "--debug" in args
    if debug:
        args.remove("--debug")
    log_path = _setup_logging(debug)
    _log_environment()
    if log_path:
        logger.info("debug log: %s", log_path)

    app = QApplication.instance() or QApplication(sys.argv)

    initial_file = args[0] if args else None
    window = STLClipperApp(initial_file=initial_file)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
