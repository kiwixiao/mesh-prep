#!/usr/bin/env pvpython
"""
4-Panel pathline rendering movie for pulsatile aortic flow.

Unlike streamlines (instantaneous snapshots), pathlines track particles
seeded continuously from the inlet as they travel through the time-varying
velocity field. Particles naturally exit through outlets.

Must be run as a single sequential process — frame N depends on particle
positions from frames 0..N-1.

Usage:
    PVBATCH=/Applications/ParaView-6.0.1.app/Contents/bin/pvbatch

    # Render all frames:
    $PVBATCH pathlines.py --case /path/to/AO_Native_001_RANS --range 0 3

    # Render with injection control:
    $PVBATCH pathlines.py --case /path/to/AO_Native_001_RANS --range 0 3 --inject-every 3

    # Query total frame count:
    $PVBATCH pathlines.py --case /path/to/AO_Native_001_RANS --total-frames
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
parser = argparse.ArgumentParser(description="4-panel pathline rendering movie")
parser.add_argument("--case", type=str, required=True,
                    help="OpenFOAM case directory (absolute path)")
parser.add_argument("--start", type=int, default=None,
                    help="First frame index to save (default: 0)")
parser.add_argument("--end", type=int, default=None,
                    help="Last frame index to save, exclusive (default: all)")
parser.add_argument("--range", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Manual colorbar range (default: [0, 3])")
parser.add_argument("--tube-radius", type=float, default=0.00008,
                    help="Tube radius in meters (default: 0.00008 = 0.08mm)")
parser.add_argument("--seed-count", type=int, default=300,
                    help="Approx number of seed points on inlet (default: 300)")
parser.add_argument("--inject-every", type=int, default=1,
                    help="Inject new particles every N timesteps (default: 1)")
parser.add_argument("--trail-length", type=int, default=5,
                    help="Max trail length in timesteps (default: 5)")
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
                          f"{os.path.basename(CASE_DIR)}_renders", "pathlines")
RESOLUTION = [3840, 2160]  # 4K combined layout
BACKGROUND = [0.0, 0.0, 0.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

# ---------------------------------------------------------------------------
# Create 2x2 layout
# ---------------------------------------------------------------------------
layout = CreateLayout("4-Panel Pathline Rendering")
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

# Resolve frame range (which frames to SAVE — but we step through ALL timesteps)
frame_start = args.start if args.start is not None else 0
frame_end = args.end if args.end is not None else len(timesteps)
frame_start = max(0, min(frame_start, len(timesteps)))
frame_end = max(frame_start, min(frame_end, len(timesteps)))
print(f"Saving frames {frame_start} to {frame_end - 1} ({frame_end - frame_start} frames)")
print(f"Stepping through ALL {len(timesteps)} timesteps for particle advection")

# Start at t=0 — ParticleTracer needs to step forward from the beginning
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

print(f"Domain center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
print(f"Domain size: {dx:.4f} x {dy:.4f} x {dz:.4f}")

# ---------------------------------------------------------------------------
# Compute inlet location for seed placement
# ---------------------------------------------------------------------------
print("Loading inlet patch as pathline seed source ...")
foam_inlet = OpenFOAMReader(FileName=FOAM_FILE)
foam_inlet.MeshRegions = ["patch/inlet"]
foam_inlet.CellArrays = []
foam_inlet.UpdatePipeline(timesteps[0])
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
# Load wall patch for anatomical context (very faint, just outline)
# ---------------------------------------------------------------------------
print("Loading wall patch for context ...")
foam_wall = OpenFOAMReader(FileName=FOAM_FILE)
foam_wall.MeshRegions = ["patch/wall"]
foam_wall.CellArrays = []

# ---------------------------------------------------------------------------
# Build pathline pipeline (ParticleTracer instead of StreamTracer)
# ---------------------------------------------------------------------------
print("Building pathline pipeline ...")

# CellDatatoPointData — ParticleTracer requires point data
cell2point = CellDatatoPointData(Input=foam)
cell2point.CellDataArraytoprocess = ["U"]

# Flatten multi-block to single dataset for ParticleTracer compatibility
vol_merged = MergeBlocks(Input=cell2point)

# ParticleTracer with custom source — seed from actual inlet patch mesh points
particle_tracer = ParticleTracer(
    Input=vol_merged,
    SeedSource=inlet_seeds,
)
particle_tracer.SelectInputVectors = ["POINTS", "U"]
particle_tracer.ForceReinjectionEveryNSteps = args.inject_every
particle_tracer.StaticSeeds = 1

# Convert particle positions to short polyline trails (flowing-particle look)
pathlines = TemporalParticlesToPathlines(Input=particle_tracer)
pathlines.MaskPoints = 1          # keep all particles
pathlines.MaxTrackLength = args.trail_length  # short tails, not full history
pathlines.MaxStepDistance = [0.005, 0.005, 0.005]

# Spline-smooth pathlines: TemporalParticlesToPathlines produces straight-line
# segments between particle positions at each timestep. With short trails
# (5 points), corners are visible. vtkSplineFilter subdivides each segment
# with cardinal spline interpolation, producing smooth curves. Point data
# (velocity) is interpolated along the spline.
smooth_pathlines = ProgrammableFilter(Input=pathlines)
smooth_pathlines.Script = """
import vtk

inp = self.GetInputDataObject(0, 0)
out = self.GetOutputDataObject(0)

spline = vtk.vtkSplineFilter()
spline.SetInputData(inp)
spline.SetSubdivideToSpecified()
spline.SetNumberOfSubdivisions(6)
spline.Update()
out.ShallowCopy(spline.GetOutput())
"""

# Tube filter for visibility
tubes = Tube(Input=pathlines)  # TODO: re-enable smooth_pathlines after testing
tubes.Radius = args.tube_radius
tubes.NumberofSides = 6
tubes.VaryRadius = "Off"

# Compute velocity magnitude explicitly
calc_umag = Calculator(Input=tubes)
calc_umag.Function = "mag(U)"
calc_umag.ResultArrayName = "Umag"

print(f"  Seed: ~{args.seed_count} points subsampled from {n_inlet_pts} inlet mesh points")
print(f"  Inject every: {args.inject_every} timestep(s)")
print(f"  Trail length: {args.trail_length} timesteps")
print(f"  Tube radius: {args.tube_radius} m")

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

    # Pathline tubes colored by velocity magnitude
    tubes_display = Show(calc_umag, view)
    tubes_display.Representation = "Surface"
    ColorBy(tubes_display, ("POINTS", "Umag"))

    # Wall surface overlay — very faint outline for spatial reference
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
scalar_bar.LabelFormat = "%.2f"
scalar_bar.ScalarBarLength = 0.7
scalar_bar.AddRangeLabels = 0
scalar_bar.UseCustomLabels = 1
scalar_bar.CustomLabels = [vel_min + i * (vel_max - vel_min) / 5.0 for i in range(6)]

# ---------------------------------------------------------------------------
# Time annotation (perspective view, top-right)
# ---------------------------------------------------------------------------
time_text = Text()
time_text.Text = f"t = {timesteps[0]:.3f} s"
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
# Render animation frames — sequential with warmup
#
# ParticleTracer is time-dependent: we must step through ALL timesteps
# from t=0 so particles accumulate correctly. Only frames in
# [frame_start, frame_end) are saved to disk.
# ---------------------------------------------------------------------------
n_save = frame_end - frame_start
print(f"\nStepping through {len(timesteps)} timesteps, saving {n_save} frames "
      f"[{frame_start}..{frame_end - 1}] to {OUTPUT_DIR}/ ...")

for i in range(len(timesteps)):
    t = timesteps[i]
    animScene.AnimationTime = t
    for view in views.values():
        view.Update()

    if frame_start <= i < frame_end:
        # Save this frame
        time_text.Text = f"t = {t:.3f} s"
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

        saved = i - frame_start + 1
        if saved % 10 == 0 or saved == 1 or i == frame_end - 1:
            print(f"  [{saved}/{n_save}] t={t:.3f}s -> {fname}")
    else:
        # Warmup: step through time for particle accumulation, don't save
        print(f"  [warmup] t={t:.3f}s ({i + 1}/{len(timesteps)})")

print(f"\nDone! {n_save} frames saved to {OUTPUT_DIR}/")
