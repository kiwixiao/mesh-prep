#!/usr/bin/env pvpython
"""
4-Panel streamline rendering movie for RANS aortic flow (kOmegaSST).

Generates a time-series of 4K frames with a 2x2 layout using anatomical
camera orientation (Z-up = superior):
  Coronal (Anterior) |  Sagittal (Right)
  Axial   (Superior) |  3D Overview

Streamlines are seeded from the inlet patch and colored by velocity magnitude.
Recomputed at each timestep to show instantaneous flow paths.

Usage:
    PVBATCH=/Applications/ParaView-6.0.1.app/Contents/bin/pvbatch

    # Render all frames:
    $PVBATCH streamlines.py --case /path/to/AO_Native_001_RANS

    # Render with fixed colorbar range:
    $PVBATCH streamlines.py --case /path/to/AO_Native_001_RANS --range 0 3

    # Render frame range (for parallel workers):
    $PVBATCH streamlines.py --case /path/to/AO_Native_001_RANS --start 0 --end 40

    # Query total frame count:
    $PVBATCH streamlines.py --case /path/to/AO_Native_001_RANS --total-frames
"""

from paraview.simple import *
import os
import sys
import math
import argparse
import numpy as np
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
parser = argparse.ArgumentParser(description="4-panel streamline rendering movie")
parser.add_argument("--case", type=str, required=True,
                    help="OpenFOAM case directory (absolute path)")
parser.add_argument("--start", type=int, default=None,
                    help="First frame index (default: 0)")
parser.add_argument("--end", type=int, default=None,
                    help="Last frame index, exclusive (default: all)")
parser.add_argument("--range", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Manual colorbar range (default: [0, 3])")
parser.add_argument("--tube-radius", type=float, default=0.00008,
                    help="Tube radius in meters (default: 0.00008 = 0.08mm)")
parser.add_argument("--max-length", type=float, default=0.25,
                    help="Max streamline propagation length in meters (default: 0.25)")
parser.add_argument("--seed-count", type=int, default=500,
                    help="Approx number of streamline seed points (default: 500)")
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
                          f"{os.path.basename(CASE_DIR)}_renders", "streamlines")
RESOLUTION = [3840, 2160]  # 4K combined layout
BACKGROUND = [0.0, 0.0, 0.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

# ---------------------------------------------------------------------------
# Create 2x2 layout
# ---------------------------------------------------------------------------
layout = CreateLayout("4-Panel Streamline Rendering")
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
# Load OpenFOAM case — internalMesh with velocity
# ---------------------------------------------------------------------------
print(f"Loading {FOAM_FILE} (internalMesh) ...")
foam = OpenFOAMReader(FileName=FOAM_FILE)
foam.MeshRegions = ["internalMesh"]
foam.CellArrays = ["U"]

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

# Move to a mid-simulation timestep to build pipeline (avoid t=0 zero velocity)
init_idx = min(len(timesteps) // 2, len(timesteps) - 1)
animScene.AnimationTime = timesteps[init_idx]
foam.UpdatePipeline(timesteps[init_idx])

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

print(f"Domain center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
print(f"Domain size: {dx:.4f} x {dy:.4f} x {dz:.4f}")

# ---------------------------------------------------------------------------
# Compute inlet location for seed placement
# ---------------------------------------------------------------------------
print("Loading inlet patch as streamline seed source ...")
foam_inlet = OpenFOAMReader(FileName=FOAM_FILE)
foam_inlet.MeshRegions = ["patch/inlet"]
foam_inlet.CellArrays = []
foam_inlet.UpdatePipeline(timesteps[init_idx])
inlet_info = foam_inlet.GetDataInformation()
ib = inlet_info.GetBounds()
inlet_cx = (ib[0] + ib[1]) / 2.0
inlet_cy = (ib[2] + ib[3]) / 2.0
inlet_cz = (ib[4] + ib[5]) / 2.0
n_inlet_pts = inlet_info.GetNumberOfPoints()
print(f"  Inlet center: ({inlet_cx:.6f}, {inlet_cy:.6f}, {inlet_cz:.6f})")
print(f"  Inlet mesh points: {n_inlet_pts}")

# Subsample inlet to desired seed count
stride = max(1, n_inlet_pts // args.seed_count)
inlet_seeds = MaskPoints(Input=foam_inlet)
inlet_seeds.OnRatio = stride
inlet_seeds.RandomSampling = True
inlet_seeds.MaximumNumberofPoints = args.seed_count

# ---------------------------------------------------------------------------
# Load wall patch for anatomical context
# ---------------------------------------------------------------------------
print("Loading wall patch for context ...")
foam_wall = OpenFOAMReader(FileName=FOAM_FILE)
foam_wall.MeshRegions = ["patch/wall"]
foam_wall.CellArrays = []

# ---------------------------------------------------------------------------
# Build streamline pipeline
# ---------------------------------------------------------------------------
print("Building streamline pipeline ...")

# CellDatatoPointData — StreamTracer requires point data
cell2point = CellDatatoPointData(Input=foam)
cell2point.CellDataArraytoprocess = ["U"]

# StreamTracer with custom source — seed from actual inlet patch mesh points
stream = StreamTracerWithCustomSource(Input=cell2point, SeedSource=inlet_seeds)
stream.Vectors = ["POINTS", "U"]
stream.MaximumStreamlineLength = args.max_length
stream.IntegrationDirection = "FORWARD"
stream.IntegratorType = "Runge-Kutta 4-5"
stream.MaximumSteps = 20000
stream.TerminalSpeed = 0.01

# Tube filter for visibility
tubes = Tube(Input=stream)
tubes.Radius = args.tube_radius
tubes.NumberofSides = 6
tubes.VaryRadius = "Off"

# Compute velocity magnitude explicitly (Tube filter mangles vector-to-magnitude mapping)
calc_umag = Calculator(Input=tubes)
calc_umag.Function = "mag(U)"
calc_umag.ResultArrayName = "Umag"

print(f"  Seed: ~{args.seed_count} points subsampled from {n_inlet_pts} inlet mesh points")
print(f"  Max propagation length: {args.max_length} m")
print(f"  Tube radius: {args.tube_radius} m")
print(f"  Integration: FORWARD, RK4-5")

# ---------------------------------------------------------------------------
# Colorbar range
# ---------------------------------------------------------------------------
if args.range is not None:
    vel_min, vel_max = args.range
else:
    vel_min, vel_max = 0.0, 3.0  # default for aortic flow [m/s]
print(f"Colorbar range: [{vel_min:.4g}, {vel_max:.4g}]")

# ---------------------------------------------------------------------------
# Colormap setup
# ---------------------------------------------------------------------------
lut = GetColorTransferFunction("Umag")
lut.ApplyPreset("Fast", True)
lut.RescaleTransferFunction(vel_min, vel_max)
lut.AutomaticRescaleRangeMode = "Never"

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
    cam.Zoom(1.15)


# ---------------------------------------------------------------------------
# Set up pipeline in each view
# ---------------------------------------------------------------------------
for name, view in views.items():
    SetActiveView(view)
    cfg = camera_configs[name]

    # Streamline tubes colored by velocity magnitude
    tubes_display = Show(calc_umag, view)
    tubes_display.Representation = "Surface"
    ColorBy(tubes_display, ("POINTS", "Umag"))

    # Wall surface overlay — semi-transparent anatomical context
    wall_display = Show(foam_wall, view)
    wall_display.Representation = "Surface"
    wall_display.DiffuseColor = [0.85, 0.85, 0.85]
    wall_display.Opacity = 0.08
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
# Apply LUT after all ColorBy calls — ColorBy resets the preset to default
# ---------------------------------------------------------------------------
lut = GetColorTransferFunction("Umag")
lut.ApplyPreset("Fast", True)
lut.RescaleTransferFunction(vel_min, vel_max)
lut.AutomaticRescaleRangeMode = "Never"

# Re-link LUT on every display so geometry uses the updated colormap
for name, view in views.items():
    d = GetDisplayProperties(calc_umag, view)
    d.LookupTable = lut
    d.SetScalarBarVisibility(view, False)

# ---------------------------------------------------------------------------
# Colorbar on perspective view only
# ---------------------------------------------------------------------------
SetActiveView(view_persp)
tubes_display_persp = GetDisplayProperties(calc_umag, view_persp)
tubes_display_persp.SetScalarBarVisibility(view_persp, True)
scalar_bar = GetScalarBar(lut, view_persp)
scalar_bar.Title = "Velocity Magnitude [m/s]"
scalar_bar.ComponentTitle = ""
scalar_bar.TitleColor = [0.9, 0.9, 0.9]
scalar_bar.LabelColor = [0.9, 0.9, 0.9]
scalar_bar.AutomaticLabelFormat = 0
scalar_bar.LabelFormat = "%.2f"      # correct property (not RangeLabelFormat)
scalar_bar.ScalarBarLength = 0.7     # taller bar for 6-tick spacing
scalar_bar.AddRangeLabels = 0        # don't double-add endpoints
scalar_bar.UseCustomLabels = 1
scalar_bar.CustomLabels = [vel_min + i * (vel_max - vel_min) / 5.0 for i in range(6)]

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
print(f"\nRendering {n_frames} streamline frames [{frame_start}..{frame_end - 1}] to {OUTPUT_DIR}/ ...")
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
    print(f"      pulsatile_streamlines.mp4")
