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
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
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
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

# Name-based cap colors for semantic identification
INLET_COLOR = (0.9, 0.2, 0.2)       # red
OUTLET_COLOR = (0.2, 0.4, 0.9)      # blue
DEFAULT_CAP_COLOR = (0.2, 0.8, 0.3) # green (fallback)

WALL_COLOR = (0.7, 0.7, 0.75)
PREVIEW_COLOR = (0.0, 1.0, 1.0)  # cyan for slice preview


def _color_for_name(name: str) -> tuple:
    """Return color based on patch name: red for inlet, blue for outlet, green otherwise."""
    lower = name.lower()
    if "inlet" in lower:
        return INLET_COLOR
    elif "outlet" in lower:
        return OUTLET_COLOR
    return DEFAULT_CAP_COLOR


@dataclass
class ClipDefinition:
    """Holds a single clipping definition and its extracted cap mesh."""
    name: str
    origin: np.ndarray          # cut plane origin (extracted from box face)
    normal: np.ndarray          # cut plane normal — points toward KEPT side
    box_planes: list = field(default_factory=list)  # 6 planes as [(normal, point), ...]
    cap_mesh: Optional[pv.PolyData] = field(default=None, repr=False)
    color: tuple = (0.9, 0.2, 0.2)


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
                 box_planes_data: list = None) -> ClipDefinition:
        color = _color_for_name(name)
        clip_def = ClipDefinition(
            name=name,
            origin=np.asarray(origin, dtype=float),
            normal=np.asarray(normal, dtype=float),
            box_planes=box_planes_data if box_planes_data else [],
            color=color,
        )
        self.clips.append(clip_def)
        self.recompute_all()
        return clip_def

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
            return
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
                clip_def.cap_mesh = self._generate_cap(
                    self.original_mesh, clip_def.origin, clip_def.normal,
                    clip_def.box_planes if clip_def.box_planes else None
                )
            except RuntimeError:
                clip_def.cap_mesh = pv.PolyData()
        self._wall_mesh = working

    def get_wall_mesh(self) -> Optional[pv.PolyData]:
        return self._wall_mesh

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def _polydata_to_ascii_stl_block(mesh: pv.PolyData, solid_name: str) -> str:
        """Convert a PolyData to an ASCII STL solid block."""
        if mesh is None or mesh.n_cells == 0:
            return f"solid {solid_name}\nendsolid {solid_name}\n"

        # Ensure we have triangle faces and normals
        tri = mesh.triangulate()
        tri.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        lines = [f"solid {solid_name}"]
        normals = tri.cell_normals
        points = tri.points
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

    def export_combined_stl(self, filepath: str):
        """Write a single ASCII STL with multiple solid blocks."""
        blocks = []
        for clip_def in self.clips:
            blocks.append(self._polydata_to_ascii_stl_block(clip_def.cap_mesh, clip_def.name))
        blocks.append(self._polydata_to_ascii_stl_block(self._wall_mesh, "wall"))
        with open(filepath, "w") as f:
            f.write("".join(blocks))

    def export_clip_planes(self, filepath: str):
        """Save clip plane origins and normals as JSON for VMTK centerline generation."""
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

    def export_openfoam(self, case_dir: str, stl_filename: str) -> dict:
        """
        Export for OpenFOAM: multi-solid STL to case_dir/constant/triSurface/
        and clip plane data as JSON to case_dir/.

        Returns dict with output paths for confirmation.
        """
        tri_dir = os.path.join(case_dir, "constant", "triSurface")
        os.makedirs(tri_dir, exist_ok=True)

        stl_path = os.path.join(tri_dir, stl_filename)
        self.export_combined_stl(stl_path)

        planes_path = os.path.join(case_dir, "clip_planes.json")
        self.export_clip_planes(planes_path)

        return {"stl_path": stl_path, "planes_path": planes_path}

    def export_separate_stl(self, output_dir: str):
        """Write one STL file per patch into output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        for clip_def in self.clips:
            path = os.path.join(output_dir, f"{clip_def.name}.stl")
            with open(path, "w") as f:
                f.write(self._polydata_to_ascii_stl_block(clip_def.cap_mesh, clip_def.name))
        wall_path = os.path.join(output_dir, "wall.stl")
        with open(wall_path, "w") as f:
            f.write(self._polydata_to_ascii_stl_block(self._wall_mesh, "wall"))


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

        self._build_ui()
        self._build_menu()
        self._update_button_states()

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
        layout.addWidget(self.plotter.interactor, stretch=3)

        # Control panel
        panel = QVBoxLayout()
        layout.addLayout(panel, stretch=1)

        # --- Load ---
        self.btn_load = QPushButton("Load STL...")
        self.btn_load.clicked.connect(self._on_load)
        panel.addWidget(self.btn_load)

        panel.addWidget(self._separator("Clipping"))

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

        panel.addWidget(self._separator("Export"))

        self.btn_export_foam = QPushButton("Export for OpenFOAM")
        self.btn_export_foam.clicked.connect(self._on_export_openfoam)
        panel.addWidget(self.btn_export_foam)

        self.btn_export_sep = QPushButton("Export Separate STLs")
        self.btn_export_sep.clicked.connect(self._on_export_separate)
        panel.addWidget(self.btn_export_sep)

        self.btn_export_comb = QPushButton("Export Combined STL")
        self.btn_export_comb.clicked.connect(self._on_export_combined)
        panel.addWidget(self.btn_export_comb)

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

        self.wall_label = QLabel("Wall: — faces")
        panel.addWidget(self.wall_label)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load an STL file to begin.")

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

        # Compute initial box state from mesh bounds
        center = np.array(mesh.center, dtype=float)
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
            self._refresh_display()
            self._refresh_patch_list()
            self._update_status()
            self._update_button_states()

    def _refresh_patch_list(self):
        self.patch_list.clear()
        for clip_def in self.engine.clips:
            n = clip_def.cap_mesh.n_cells if clip_def.cap_mesh else 0
            r, g, b = [int(c * 255) for c in clip_def.color]
            item = QListWidgetItem(f"{clip_def.name}  ({n:,} faces)")
            item.setForeground(Qt.black)
            item.setBackground(QColor(r, g, b, 60))
            self.patch_list.addItem(item)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _refresh_display(self):
        self.plotter.clear()

        wall = self.engine.get_wall_mesh()
        if wall is None or wall.n_cells == 0:
            self.plotter.render()
            return

        # Wall mesh — semi-transparent gray
        self.plotter.add_mesh(
            wall, color=WALL_COLOR, opacity=0.4,
            show_edges=False, name="wall",
        )

        # Cap patches — name-based colors with white edges for visibility
        for clip_def in self.engine.clips:
            if clip_def.cap_mesh and clip_def.cap_mesh.n_cells > 0:
                self.plotter.add_mesh(
                    clip_def.cap_mesh, color=clip_def.color, opacity=1.0,
                    show_edges=True, edge_color="white", line_width=2,
                    name=f"cap_{clip_def.name}",
                )

        self.plotter.reset_camera()
        self.plotter.render()

        # Update wall face count label
        n_wall = wall.n_cells if wall else 0
        self.wall_label.setText(f"Wall: {n_wall:,} faces")

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
        sep_dir = os.path.join(case_dir, "separate")
        try:
            self.engine.export_separate_stl(sep_dir)
            planes_path = os.path.join(case_dir, "clip_planes.json")
            self.engine.export_clip_planes(planes_path)
            files = [f"{c.name}.stl" for c in self.engine.clips] + ["wall.stl"]
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(files)} files to:\n{sep_dir}\n\n"
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
        filepath = os.path.join(case_dir, "boundary.stl")
        try:
            self.engine.export_combined_stl(filepath)
            planes_path = os.path.join(case_dir, "clip_planes.json")
            self.engine.export_clip_planes(planes_path)
            QMessageBox.information(
                self, "Export Complete",
                f"Combined STL saved to:\n{filepath}\n\n"
                f"Clip planes JSON:\n{planes_path}\n\n"
                f"Patches: {', '.join(c.name for c in self.engine.clips)}, wall",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_openfoam(self):
        if not self.engine.clips or self._loaded_filepath is None:
            return
        case_dir = self._get_or_create_case_dir()
        if not case_dir:
            return
        stl_filename = os.path.basename(self._loaded_filepath)
        try:
            result = self.engine.export_openfoam(case_dir, stl_filename)
            patches = ", ".join(c.name for c in self.engine.clips) + ", wall"
            QMessageBox.information(
                self, "OpenFOAM Export Complete",
                f"Multi-solid STL:\n{result['stl_path']}\n\n"
                f"Clip planes JSON:\n{result['planes_path']}\n\n"
                f"Patches: {patches}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    initial_file = sys.argv[1] if len(sys.argv) > 1 else None
    window = STLClipperApp(initial_file=initial_file)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
