#!/usr/bin/env pvpython
"""
4-Panel volume rendering movie for RANS aortic flow (kOmegaSST).

Generates a time-series of 4K frames with a 2x2 layout using anatomical
camera orientation (Z-up = superior):
  Coronal (Anterior) |  Sagittal (Right)
  Axial   (Superior) |  3D Overview

Supports multiple fields: velocity, vorticity, pressure, qcriterion,
pressure_gradient.

Usage:
    PVBATCH=/Applications/ParaView-6.0.1.app/Contents/bin/pvbatch

    # Render all frames (velocity):
    $PVBATCH movie.py --case /path/to/AO_Native_001_RANS

    # Render frame range (for parallel workers):
    $PVBATCH movie.py --case /path/to/AO_Native_001_RANS --start 0 --end 40

    # Render vorticity:
    $PVBATCH movie.py --case /path/to/AO_Native_001_RANS --field vorticity

    # Query total frame count:
    $PVBATCH movie.py --case /path/to/AO_Native_001_RANS --total-frames
"""

from paraview.simple import *
import os
import sys
import math
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ct_preset import apply_ct_transfer_function
try:
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    _HAS_OVERLAY_DEPS = True
except ImportError as _e:
    _HAS_OVERLAY_DEPS = False
    print(f"WARNING: Waveform overlay disabled (missing dependency: {_e})")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="4-panel volume rendering movie")
parser.add_argument("--case", type=str, required=True,
                    help="OpenFOAM case directory (absolute path)")
parser.add_argument("--start", type=int, default=None,
                    help="First frame index (default: 0)")
parser.add_argument("--end", type=int, default=None,
                    help="Last frame index, exclusive (default: all)")
parser.add_argument("--field", type=str, default="velocity",
                    choices=["velocity", "vorticity", "pressure", "qcriterion",
                             "pressure_gradient"],
                    help="Field to render (default: velocity)")
parser.add_argument("--resample", type=int, default=256,
                    help="ResampleToImage grid size (default: 256)")
parser.add_argument("--range", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Manual colorbar range (overrides P80 auto-scan)")
parser.add_argument("--smooth", type=float, default=4.0,
                    help="Gaussian smooth sigma in voxels (default: 4.0, 0=off)")
parser.add_argument("--ct-file", type=str, default=None,
                    help="CT VTI file path for anatomical overlay (optional)")
parser.add_argument("--ct-opacity", type=float, default=1.0,
                    help="Global CT opacity multiplier 0-1 (default: 1.0)")
parser.add_argument("--ct-preset", default="bone",
                    choices=["bone", "cardiac"],
                    help="CT preset: bone (skeleton only) or cardiac (heart/aortic wall visible)")
parser.add_argument("--no-ct", action="store_true",
                    help="Skip CT volume rendering even if --ct-file is given")
parser.add_argument("--total-frames", action="store_true",
                    help="Print total frame count and exit")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CASE_DIR = args.case
FOAM_FILE = os.path.join(CASE_DIR, f"{os.path.basename(CASE_DIR)}.foam")

# --total-frames: count foam timesteps and exit
if args.total_frames:
    foam_tmp = OpenFOAMReader(FileName=FOAM_FILE)
    animScene_tmp = GetAnimationScene()
    animScene_tmp.UpdateAnimationUsingDataTimeSteps()
    print(len(list(foam_tmp.TimestepValues)))
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(CASE_DIR),
                          f"{os.path.basename(CASE_DIR)}_renders", args.field)
RESOLUTION = [3840, 2160]  # 4K combined layout
BACKGROUND = [0.0, 0.0, 0.0]
BLOOD_NU = 3.3e-6  # kinematic viscosity of blood [m^2/s]

# Field-specific settings
FIELD_CONFIG = {
    "velocity": {
        "cell_arrays": ["U"],
        "colormap": "Fast",
        "bar_title": "Velocity Magnitude",
        "bar_unit": "[m/s]",
        "color_by_field": "U",
        "color_by_component": "Magnitude",
        "prescan_field": "U",
    },
    "vorticity": {
        "cell_arrays": ["U"],
        "colormap": "Blue to Red Rainbow",
        "bar_title": "Vorticity Magnitude",
        "bar_unit": "[1/s]",
        "color_by_field": "VorticityMagnitude",
        "color_by_component": "",
        "prescan_field": "VorticityMagnitude",
    },
    "pressure": {
        "cell_arrays": ["p"],
        "colormap": "Fast",
        "bar_title": "Pressure",
        "bar_unit": "[m^2/s^2]",
        "color_by_field": "p",
        "color_by_component": "",
        "prescan_field": "p",
    },
    "qcriterion": {
        "cell_arrays": ["U"],
        "colormap": "Viridis (matplotlib)",
        "bar_title": "Q-Criterion",
        "bar_unit": "[1/s^2]",
        "color_by_field": "Q-criterion",
        "color_by_component": "",
        "prescan_field": "Q-criterion",
    },
    "pressure_gradient": {
        "cell_arrays": ["p"],
        "colormap": "Inferno (matplotlib)",
        "bar_title": "Pressure Gradient Magnitude",
        "bar_unit": "[1/s^2]",
        "color_by_field": "PressureGradientMag",
        "color_by_component": "",
        "prescan_field": "PressureGradientMag",
    },
}

fcfg = FIELD_CONFIG[args.field]

os.makedirs(OUTPUT_DIR, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

# ---------------------------------------------------------------------------
# Create 2x2 layout
# ---------------------------------------------------------------------------
layout = CreateLayout("4-Panel Volume Rendering")
layout.SplitHorizontal(0, 0.5)
layout.SplitVertical(1, 0.5)   # left column: top-left (3) and bottom-left (4)
layout.SplitVertical(2, 0.5)   # right column: top-right (5) and bottom-right (6)

view_coronal = CreateRenderView()
layout.AssignView(3, view_coronal)

view_sagittal = CreateRenderView()
layout.AssignView(5, view_sagittal)

view_axial = CreateRenderView()
layout.AssignView(4, view_axial)

view_persp = CreateRenderView()
layout.AssignView(6, view_persp)

views = {
    "coronal":  view_coronal,
    "sagittal": view_sagittal,
    "axial":    view_axial,
    "persp":    view_persp,
}

for name, view in views.items():
    view.Background = BACKGROUND
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0

# ---------------------------------------------------------------------------
# Load OpenFOAM case
# ---------------------------------------------------------------------------
print(f"Loading {FOAM_FILE} ...")
foam = OpenFOAMReader(FileName=FOAM_FILE)
foam.MeshRegions = ["internalMesh"]
foam.CellArrays = fcfg["cell_arrays"]

animScene = GetAnimationScene()
animScene.UpdateAnimationUsingDataTimeSteps()
timesteps = list(foam.TimestepValues)
print(f"Found {len(timesteps)} timesteps: {timesteps[0]:.3f} to {timesteps[-1]:.3f}s")

# Resolve frame range
frame_start = args.start if args.start is not None else 0
frame_end = args.end if args.end is not None else len(timesteps)
frame_start = max(0, min(frame_start, len(timesteps)))
frame_end = max(frame_start, min(frame_end, len(timesteps)))
print(f"Rendering frames {frame_start} to {frame_end - 1} ({frame_end - frame_start} frames)")

# Move to first timestep to build pipeline
animScene.AnimationTime = timesteps[0]
foam.UpdatePipeline(timesteps[0])

# ---------------------------------------------------------------------------
# Domain bounds for camera setup
# ---------------------------------------------------------------------------
info = foam.GetDataInformation()
bounds = info.GetBounds()
center = [
    (bounds[0] + bounds[1]) / 2.0,
    (bounds[2] + bounds[3]) / 2.0,
    (bounds[4] + bounds[5]) / 2.0,
]
dx = bounds[1] - bounds[0]
dy = bounds[3] - bounds[2]
dz = bounds[5] - bounds[4]
max_dim = max(dx, dy, dz)
cam_dist = max_dim * 2.5

print(f"Domain center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
print(f"Domain size: {dx:.2f} x {dy:.2f} x {dz:.2f}")

# ---------------------------------------------------------------------------
# Build filter chain for selected field
# ---------------------------------------------------------------------------
def build_filter_chain(foam_source, field_name):
    """Build ParaView filter chain for the selected field."""
    if field_name == "velocity":
        return foam_source

    elif field_name == "vorticity":
        grad = Gradient(Input=foam_source)
        grad.ScalarArray = ["CELLS", "U"]
        grad.ComputeVorticity = 1
        grad.ComputeGradient = 0
        grad.VorticityArrayName = "Vorticity"

        calc = Calculator(Input=grad)
        calc.Function = "mag(Vorticity)"
        calc.ResultArrayName = "VorticityMagnitude"
        calc.AttributeType = "Cell Data"
        return calc

    elif field_name == "pressure":
        return foam_source

    elif field_name == "qcriterion":
        grad = Gradient(Input=foam_source)
        grad.ScalarArray = ["CELLS", "U"]
        grad.ComputeQCriterion = 1
        grad.ComputeGradient = 0
        grad.QCriterionArrayName = "Q-criterion"
        return grad

    elif field_name == "pressure_gradient":
        grad = Gradient(Input=foam_source)
        grad.ScalarArray = ["CELLS", "p"]
        grad.ComputeGradient = 1
        grad.ResultArrayName = "PressureGradient"

        calc = Calculator(Input=grad)
        calc.Function = "mag(PressureGradient)"
        calc.ResultArrayName = "PressureGradientMag"
        calc.AttributeType = "Cell Data"
        return calc

    else:
        raise ValueError(f"Unknown field: {field_name}")


print(f"Building filter chain for '{args.field}' ...")
pipeline_output = build_filter_chain(foam, args.field)

# ---------------------------------------------------------------------------
# P80 colorbar pre-scan
# ---------------------------------------------------------------------------
def _format_range(field_name, p80):
    """Convert a P80 value into (vmin, vmax) based on field semantics."""
    if field_name == "pressure":
        return (-p80, p80)       # symmetric diverging
    else:
        return (0.0, p80)        # zero to P80


def _fallback_range(source, field_name, field_cfg, timesteps_list):
    """Tier 2: sample data ranges across timesteps using metadata API."""
    prescan_field = field_cfg["prescan_field"]
    n_samples = min(10, len(timesteps_list))
    step = max(1, len(timesteps_list) // n_samples)
    indices = list(range(0, len(timesteps_list), step))[:n_samples]

    all_maxes = []
    for idx in indices:
        source.UpdatePipeline(timesteps_list[idx])
        di = source.GetDataInformation()
        for assoc in [1, 0]:  # cell data (1), point data (0)
            ai = di.GetArrayInformation(prescan_field, assoc)
            if ai is not None:
                n_comp = ai.GetNumberOfComponents()
                r = ai.GetComponentRange(-1 if n_comp > 1 else 0)
                all_maxes.append(max(abs(r[0]), abs(r[1])))
                break

    if all_maxes:
        approx = max(all_maxes) * 0.8
        print(f"  Tier 2 fallback: max={max(all_maxes):.4g}, using {approx:.4g}")
        return _format_range(field_name, approx)
    else:
        print("  WARNING: No data range found, using [0, 0.5]")
        return (0.0, 0.5)


def compute_p80_range(source, field_name, field_cfg, timesteps_list):
    """Compute 80th percentile range by sampling ~10 timesteps."""
    try:
        from paraview.servermanager import Fetch
        from vtk.numpy_interface import dataset_adapter as dsa

        prescan_field = field_cfg["prescan_field"]
        n_samples = min(10, len(timesteps_list))
        step = max(1, len(timesteps_list) // n_samples)
        indices = list(range(0, len(timesteps_list), step))[:n_samples]

        all_values = []
        print(f"  P80 pre-scan: {len(indices)} timesteps ...")

        for idx in indices:
            t = timesteps_list[idx]
            source.UpdatePipeline(t)
            raw = Fetch(source)
            data = dsa.WrapDataObject(raw)

            arr_vtk = None
            for store in [data.CellData, data.PointData]:
                if prescan_field in store.keys():
                    arr_vtk = store[prescan_field]
                    break

            if arr_vtk is None:
                continue

            try:
                arr = np.asarray(arr_vtk, dtype=np.float64)
            except (ValueError, TypeError):
                if hasattr(arr_vtk, 'Arrays'):
                    parts = [np.asarray(a, dtype=np.float64)
                             for a in arr_vtk.Arrays if a is not None]
                    arr = np.concatenate(parts) if parts else None
                    if arr is None:
                        continue
                else:
                    continue

            if arr.ndim == 2 and arr.shape[1] >= 2:
                vals = np.linalg.norm(arr, axis=1)
            else:
                vals = arr.ravel()

            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                all_values.append(vals)

        if not all_values:
            raise RuntimeError("No data collected from any timestep")

        combined = np.concatenate(all_values)
        p80 = 0.8 * float(np.max(np.abs(combined)))
        n_cells = len(combined) // len(all_values)
        print(f"  P80 = {p80:.4g} ({n_cells:,} cells x {len(all_values)} timesteps)")
        return _format_range(field_name, p80)

    except Exception as e:
        print(f"  Tier 1 failed ({e}), trying Tier 2 ...")
        return _fallback_range(source, field_name, field_cfg, timesteps_list)


if args.range is not None:
    vel_min, vel_max = args.range
    print(f"Using manual colorbar range: [{vel_min:.4g}, {vel_max:.4g}]")
else:
    print("Computing colorbar range (from raw mesh, not resampled) ...")
    # Use foam source directly to avoid ResampleToImage extrapolation artifacts
    vel_min, vel_max = compute_p80_range(foam, args.field, fcfg, timesteps)
    print(f"Colorbar range: [{vel_min:.4g}, {vel_max:.4g}]")

# Reset to first timestep after pre-scan
animScene.AnimationTime = timesteps[0]
foam.UpdatePipeline(timesteps[0])

# ---------------------------------------------------------------------------
# Resample to uniform grid for volume rendering
# ---------------------------------------------------------------------------
N = args.resample
print(f"Creating ResampleToImage filter ({N}^3 grid, smooth σ={args.smooth}) ...")
resample = ResampleToImage(Input=pipeline_output)
resample.SamplingDimensions = [N, N, N]

# Gaussian smooth to eliminate voxel artifacts in volume rendering
if args.smooth > 0:
    smooth = ProgrammableFilter(Input=resample)
    smooth.OutputDataSetType = "vtkImageData"
    smooth.Script = f"""
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
inp = self.GetInput()
out = self.GetOutput()
out.CopyStructure(inp)

# Smooth each point-data array independently
for i in range(inp.GetPointData().GetNumberOfArrays()):
    arr = inp.GetPointData().GetArray(i)
    name = arr.GetName()

    # vtkImageGaussianSmooth operates on the active scalars
    tmp = vtk.vtkImageData()
    tmp.CopyStructure(inp)
    tmp.GetPointData().SetScalars(arr)

    gauss = vtk.vtkImageGaussianSmooth()
    gauss.SetInputData(tmp)
    gauss.SetStandardDeviations({args.smooth}, {args.smooth}, {args.smooth})
    gauss.SetRadiusFactors(3, 3, 3)
    gauss.SetDimensionality(3)
    gauss.Update()

    smoothed = gauss.GetOutput().GetPointData().GetScalars()
    smoothed.SetName(name)
    out.GetPointData().AddArray(smoothed)

# Copy cell data as-is
for i in range(inp.GetCellData().GetNumberOfArrays()):
    out.GetCellData().AddArray(inp.GetCellData().GetArray(i))
"""
    volume_source = smooth
else:
    volume_source = resample

# ---------------------------------------------------------------------------
# Colormap setup
# ---------------------------------------------------------------------------
color_field = fcfg["color_by_field"]
color_component = fcfg["color_by_component"]

if color_component:
    color_spec = ("POINTS", color_field, color_component)
else:
    color_spec = ("POINTS", color_field)

lut = GetColorTransferFunction(color_field)
lut.ApplyPreset(fcfg["colormap"], True)

# Opacity transfer function — field-specific profiles
pwf = GetOpacityTransferFunction(color_field)
if args.field in ("vorticity", "qcriterion", "pressure_gradient"):
    # Derived fields: concentrated at shear layers, need steeper opacity ramp
    pwf.Points = [
        vel_min,                              0.0,  0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.05, 0.05, 0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.15, 0.20, 0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.4,  0.50, 0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.7,  0.75, 0.5, 0.0,
        vel_max,                              0.90, 0.5, 0.0,
    ]
elif args.field == "pressure":
    # Pressure: symmetric diverging, transparent near zero
    mid = (vel_min + vel_max) / 2.0
    pwf.Points = [
        vel_min,                              0.6,  0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.3,  0.10, 0.5, 0.0,
        mid,                                  0.0,  0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.7,  0.10, 0.5, 0.0,
        vel_max,                              0.6,  0.5, 0.0,
    ]
else:
    # Velocity: gentle ramp, transparent at low values
    pwf.Points = [
        vel_min,                              0.0,  0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.15, 0.02, 0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.4,  0.15, 0.5, 0.0,
        vel_min + (vel_max - vel_min) * 0.7,  0.4,  0.5, 0.0,
        vel_max,                              0.8,  0.5, 0.0,
    ]

# ---------------------------------------------------------------------------
# Wall surface for anatomical context (used when no CT provided)
# ---------------------------------------------------------------------------
print("Loading wall patch for context ...")
foam_wall = OpenFOAMReader(FileName=FOAM_FILE)
foam_wall.MeshRegions = ["patch/wall"]
foam_wall.CellArrays = []

# ---------------------------------------------------------------------------
# Load CT volume (optional — replaces wall surface when provided)
# ---------------------------------------------------------------------------
ct_source = None
ct_lut = None
ct_pwf = None

if args.ct_file and not args.no_ct:
    if os.path.exists(args.ct_file):
        print(f"\nLoading CT volume: {args.ct_file} ...")
        ct_source = XMLImageDataReader(FileName=[args.ct_file])
        ct_source.UpdatePipeline()
        ct_info = ct_source.GetDataInformation()
        ct_bounds = ct_info.GetBounds()
        print(f"  CT bounds (m): X [{ct_bounds[0]:.6f}, {ct_bounds[1]:.6f}]  "
              f"Y [{ct_bounds[2]:.6f}, {ct_bounds[3]:.6f}]  "
              f"Z [{ct_bounds[4]:.6f}, {ct_bounds[5]:.6f}]")

        ct_lut = GetColorTransferFunction("HU")
        ct_pwf = GetOpacityTransferFunction("HU")
        apply_ct_transfer_function(ct_lut, ct_pwf,
                                   preset=args.ct_preset,
                                   ct_opacity=args.ct_opacity)
        print(f"  CT preset: {args.ct_preset}, opacity multiplier: {args.ct_opacity}")
    else:
        print(f"\nWARNING: CT file not found: {args.ct_file}")
        print("  Continuing without CT volume rendering.")

# ---------------------------------------------------------------------------
# Camera configurations (anatomical orientation, Z-up = superior)
# ---------------------------------------------------------------------------
camera_configs = {
    "coronal": {
        "position": [center[0], center[1] - cam_dist, center[2]],
        "focal":    list(center),
        "up":       [0, 0, 1],
        "label":    "Coronal (Anterior)",
    },
    "sagittal": {
        "position": [center[0] + cam_dist, center[1], center[2]],
        "focal":    list(center),
        "up":       [0, 0, 1],
        "label":    "Sagittal (Right)",
    },
    "axial": {
        "position": [center[0], center[1], center[2] + cam_dist],
        "focal":    list(center),
        "up":       [0, 1, 0],
        "label":    "Axial (Superior)",
    },
    "persp": {
        "position": [
            center[0] + cam_dist * math.cos(math.radians(45)) * math.cos(math.radians(30)),
            center[1] + cam_dist * math.sin(math.radians(30)),
            center[2] + cam_dist * math.sin(math.radians(45)) * math.cos(math.radians(30)),
        ],
        "focal":    list(center),
        "up":       [0, 0, 1],
        "label":    "3D Overview",
    },
}


def setup_camera(view, position, focal_point, view_up):
    """Set camera position, focal point, and up direction."""
    view.CameraPosition = position
    view.CameraFocalPoint = focal_point
    view.CameraViewUp = view_up
    view.CameraParallelScale = max_dim * 0.6
    view.ResetCamera()
    cam = view.GetActiveCamera()
    cam.Zoom(2.2 if ct_source is not None else 1.15)


# ---------------------------------------------------------------------------
# Set up pipeline in each view
# ---------------------------------------------------------------------------
for name, view in views.items():
    SetActiveView(view)
    cfg = camera_configs[name]

    # CT volume rendering (behind flow — rendered first if available)
    if ct_source is not None:
        ct_display = Show(ct_source, view)
        ct_display.Representation = "Volume"
        ColorBy(ct_display, ("POINTS", "HU"))
        ct_display.LookupTable = ct_lut
        ct_display.VolumeRenderingMode = "GPU Based"
        ct_vp = ct_display.GetProperty("VolumeProperty")
        if ct_vp is not None:
            ct_vp.SetInterpolationType(1)

    # Flow velocity volume rendering
    resample_display = Show(volume_source, view)
    resample_display.Representation = "Volume"
    ColorBy(resample_display, color_spec)
    resample_display.LookupTable = lut

    # Use GPU ray casting with trilinear interpolation for smooth volume
    resample_display.VolumeRenderingMode = "GPU Based"
    vp = resample_display.GetProperty("VolumeProperty")
    if vp is not None:
        vp.SetInterpolationType(1)  # 1 = linear (vs 0 = nearest)

    # Wall surface overlay (only when no CT — CT provides anatomical context)
    if ct_source is None:
        wall_display = Show(foam_wall, view)
        wall_display.Representation = "Surface"
        wall_display.DiffuseColor = [0.85, 0.85, 0.85]
        wall_display.Opacity = 0.15
        wall_display.SetScalarColoring(None, 0)

    # Camera
    setup_camera(view, cfg["position"], cfg["focal"], cfg["up"])

    # View label
    label = Text()
    label.Text = cfg["label"]
    label_display = Show(label, view)
    label_display.WindowLocation = "Any Location"
    label_display.FontSize = 12
    label_display.Position = [0.02, 0.92]
    label_display.Color = [0.8, 0.8, 0.8]

# ---------------------------------------------------------------------------
# Colorbar on perspective view
# ---------------------------------------------------------------------------
SetActiveView(view_persp)
resample_display_persp = GetDisplayProperties(volume_source, view_persp)
resample_display_persp.SetScalarBarVisibility(view_persp, True)
scalar_bar = GetScalarBar(lut, view_persp)
scalar_bar.Title = fcfg["bar_title"]
scalar_bar.ComponentTitle = fcfg["bar_unit"]
scalar_bar.TitleColor = [0.9, 0.9, 0.9]
scalar_bar.LabelColor = [0.9, 0.9, 0.9]
scalar_bar.RangeLabelFormat = "%.2f"
scalar_bar.AutomaticLabelFormat = 0

# Apply colorbar range AFTER ColorBy() calls
lut.RescaleTransferFunction(vel_min, vel_max)
lut.AutomaticRescaleRangeMode = "Never"

# ---------------------------------------------------------------------------
# Time annotation (perspective view, top-right)
# ---------------------------------------------------------------------------
time_text = Text()
time_text.Text = f"t = {timesteps[frame_start]:.3f} s"
time_display = Show(time_text, view_persp)
time_display.WindowLocation = "Any Location"
time_display.FontSize = 14
time_display.Position = [0.70, 0.92]
time_display.Color = [1.0, 1.0, 0.6]

# ---------------------------------------------------------------------------
# Waveform inset setup (matplotlib + PIL compositing)
# ---------------------------------------------------------------------------
WAVEFORM_CSV = os.path.join(CASE_DIR, "constant", "volumetricFlowRate.csv")
WAVEFORM_POS = (1940, 1640)  # top-left corner on 3840x2160 frame
WAVEFORM_ENABLED = False

if _HAS_OVERLAY_DEPS and os.path.exists(WAVEFORM_CSV):
    print("Loading waveform data for inset ...")
    _csv = np.genfromtxt(WAVEFORM_CSV, delimiter=',', skip_header=1)
    _csv_time = _csv[:, 0]
    _csv_flow = _csv[:, 1]

    # Smooth: interpolate to fine grid + gaussian convolution
    _t_fine = np.linspace(_csv_time[0], _csv_time[-1], 500)
    _q_linear = np.interp(_t_fine, _csv_time, _csv_flow)
    _win = 11
    _w = np.exp(-0.5 * np.linspace(-2, 2, _win)**2)
    _w /= _w.sum()
    _q_smooth = np.convolve(_q_linear, _w, mode='same')
    _q_smooth[:_win // 2] = _q_linear[:_win // 2]
    _q_smooth[-_win // 2:] = _q_linear[-_win // 2:]

    _q_pad = (_q_smooth.max() - _q_smooth.min()) * 0.1
    _q_range = (float(_q_smooth.min() - _q_pad), float(_q_smooth.max() + _q_pad))

    def _create_waveform_overlay(t_cur):
        """Render waveform inset with current-time marker."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        ax.plot(_t_fine, _q_smooth, color='cyan', linewidth=3)
        q_now = float(np.interp(t_cur, _t_fine, _q_smooth))
        ax.axvline(t_cur, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.plot(t_cur, q_now, 'o', color='red', markersize=12, zorder=5)
        ax.set_xlim(_csv_time[0], _csv_time[-1])
        ax.set_ylim(*_q_range)
        ax.set_xlabel('Time [s]', color='white', fontsize=14)
        ax.set_ylabel('Q [mL/s]', color='white', fontsize=14)
        ax.set_title('Inlet Flow Rate', color='white', fontsize=16, pad=6)
        ax.tick_params(colors='white', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.5)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, transparent=True)
        buf.seek(0)
        plot_img = Image.open(buf).convert('RGBA')
        buf.close()
        plt.close(fig)
        bg = Image.new('RGBA', plot_img.size, (0, 0, 0, 178))
        return Image.alpha_composite(bg, plot_img)

    WAVEFORM_ENABLED = True
    print(f"  Waveform: {len(_csv_time)} points, "
          f"t=[{_csv_time[0]:.3f}, {_csv_time[-1]:.3f}]s")
elif not _HAS_OVERLAY_DEPS:
    pass
elif not os.path.exists(WAVEFORM_CSV):
    print(f"WARNING: {WAVEFORM_CSV} not found, skipping waveform inset")

# ---------------------------------------------------------------------------
# Render animation frames
# ---------------------------------------------------------------------------
n_frames = frame_end - frame_start
print(f"\nRendering {n_frames} frames [{frame_start}..{frame_end - 1}] to {OUTPUT_DIR}/ ...")
for i in range(frame_start, frame_end):
    t = timesteps[i]
    animScene.AnimationTime = t
    time_text.Text = f"t = {t:.3f} s"
    for view in views.values():
        view.Update()
    RenderAllViews()
    fname = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
    SaveScreenshot(fname, layout, ImageResolution=RESOLUTION)

    # Composite waveform inset onto frame
    if WAVEFORM_ENABLED:
        frame_img = Image.open(fname).convert('RGBA')
        waveform_img = _create_waveform_overlay(t)
        canvas = Image.new('RGBA', frame_img.size, (0, 0, 0, 0))
        canvas.paste(waveform_img, WAVEFORM_POS)
        result = Image.alpha_composite(frame_img, canvas)
        result.convert('RGB').save(fname)

    frame_num = i - frame_start + 1
    if frame_num % 10 == 0 or frame_num == 1 or i == frame_end - 1:
        print(f"  [{frame_num}/{n_frames}] t={t:.3f}s -> {fname}")

print(f"\nDone! {n_frames} frames saved to {OUTPUT_DIR}/")
if args.start is None and args.end is None:
    print("\nAssemble movie with:")
    print(f"  ffmpeg -framerate 16 -i {OUTPUT_DIR}/frame_%04d.png \\")
    print(f"      -c:v libx264 -pix_fmt yuv420p -crf 18 \\")
    print(f"      pulsatile_{args.field}.mp4")
