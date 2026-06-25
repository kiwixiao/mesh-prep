#!/usr/bin/env pvpython
"""
Spinning CT + Streamlines — single-panel orbiting camera movie.

Combines the orbiting camera from spinning.py (streamlines mode) with
the CT X-ray volume rendering from ct_flow.py. The camera orbits around
the aorta (Z-up) while time advances, showing CT bones + streamlines.

Output frames are decoupled from simulation timesteps: fps × duration
determines the frame count (default 25fps × 10s = 250 frames), with each
simulation timestep reused across multiple camera angles.

Usage:
    PVBATCH=/Applications/ParaView-6.0.1.app/Contents/bin/pvbatch

    # Render all frames:
    $PVBATCH spinning_ct.py --case /path/to/AO_Native_001_RANS --ct-file /path/to/CT.vti --range 0 3

    # Single-frame test:
    $PVBATCH spinning_ct.py --case /path/to/AO_Native_001_RANS --ct-file /path/to/CT.vti --range 0 3 --start 40 --end 41

    # Query total frame count (for parallel splitting):
    $PVBATCH spinning_ct.py --total-frames

    # Parallel render (10 workers):
    mesh-render --case /path/to/AO_Native_001_RANS --mode spinning-ct --ct-dir /path/to/ct --range 0 3 --workers 10
"""

import builtins as _bi
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
parser = argparse.ArgumentParser(description="Spinning CT + Streamlines movie")
parser.add_argument("--case", type=str, required=True,
                    help="OpenFOAM case directory (absolute path)")
parser.add_argument("--start", type=int, default=None,
                    help="First frame index (default: 0)")
parser.add_argument("--end", type=int, default=None,
                    help="Last frame index, exclusive (default: all)")
parser.add_argument("--range", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Manual velocity colorbar range (default: [0, 3])")
parser.add_argument("--orbits", type=float, default=2.0,
                    help="Number of full 360-degree orbits (default: 2.0)")
parser.add_argument("--elevation", type=float, default=30.0,
                    help="Camera elevation angle in degrees (default: 30)")
parser.add_argument("--fps", type=int, default=25,
                    help="Output video framerate (default: 25)")
parser.add_argument("--duration", type=float, default=10.0,
                    help="Video duration in seconds (default: 10.0)")
parser.add_argument("--total-frames", action="store_true",
                    help="Print total output frame count and exit")
parser.add_argument("--tube-radius", type=float, default=0.0001,
                    help="Tube radius for streamlines (default: 0.0001)")
parser.add_argument("--max-length", type=float, default=0.15,
                    help="Max streamline propagation length (default: 0.15)")
parser.add_argument("--seed-count", type=int, default=300,
                    help="Streamline seed point count (default: 300)")
parser.add_argument("--ct-file", type=str, default="CT_Pre_frame0_meters.vti",
                    help="CT VTI file path (default: CT_Pre_frame0_meters.vti)")
parser.add_argument("--ct-opacity", type=float, default=1.0,
                    help="Global CT opacity multiplier 0-1 (default: 1.0)")
parser.add_argument("--ct-preset", default="bone",
                    choices=["bone", "cardiac"],
                    help="CT preset: bone (skeleton only) or cardiac (heart/aortic wall visible)")
parser.add_argument("--no-ct", action="store_true",
                    help="Skip CT volume rendering (debug mode)")
args = parser.parse_args()

# --total-frames: print count and exit (fps * duration, no foam needed)
if args.total_frames:
    print(int(args.fps * args.duration))
    import sys; sys.exit(0)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CASE_DIR = args.case
FOAM_FILE = os.path.join(CASE_DIR, f"{os.path.basename(CASE_DIR)}.foam")
OUTPUT_DIR = os.path.join(os.path.dirname(CASE_DIR),
                          f"{os.path.basename(CASE_DIR)}_renders", "spinning_ct")
RESOLUTION = [1920, 1080]
BACKGROUND = [0.0, 0.0, 0.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

print(f"Case:     {CASE_DIR}")
print(f"Foam:     {FOAM_FILE}")
print(f"CT file:  {args.ct_file}")
print(f"Output:   {OUTPUT_DIR}/")

# ---------------------------------------------------------------------------
# Single render view
# ---------------------------------------------------------------------------
view = CreateRenderView()
view.ViewSize = RESOLUTION
view.Background = BACKGROUND
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 0

# ---------------------------------------------------------------------------
# Load OpenFOAM case
# ---------------------------------------------------------------------------
print(f"\nLoading {FOAM_FILE} ...")
foam = OpenFOAMReader(FileName=FOAM_FILE)
foam.MeshRegions = ["internalMesh"]
foam.CellArrays = ["U"]

animScene = GetAnimationScene()
animScene.UpdateAnimationUsingDataTimeSteps()
timesteps = list(foam.TimestepValues)
print(f"Found {len(timesteps)} timesteps: {timesteps[0]:.3f} to {timesteps[-1]:.3f}s")

# Compute total output frames (decoupled from simulation timesteps)
total_output_frames = int(args.fps * args.duration)

# Resolve output frame range
frame_start = args.start if args.start is not None else 0
frame_end = args.end if args.end is not None else total_output_frames
frame_start = _bi.max(0, _bi.min(frame_start, total_output_frames))
frame_end = _bi.max(frame_start, _bi.min(frame_end, total_output_frames))
print(f"Output: {total_output_frames} frames ({args.fps} fps x {args.duration}s)")
print(f"Rendering output frames {frame_start} to {frame_end - 1} ({frame_end - frame_start} frames)")

# Move to a mid-simulation timestep to build pipeline (avoid t=0 zero velocity)
init_idx = min(len(timesteps) // 2, len(timesteps) - 1)
animScene.AnimationTime = timesteps[init_idx]
foam.UpdatePipeline(timesteps[init_idx])

# ---------------------------------------------------------------------------
# Domain bounds
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
max_dim = _bi.max(dx, dy, dz)
cam_dist = max_dim * 2.5

print(f"Domain center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
print(f"Domain size: {dx:.4f} x {dy:.4f} x {dz:.4f}")

# ---------------------------------------------------------------------------
# Load CT volume (static — does not change per timestep)
# ---------------------------------------------------------------------------
ct_source = None
if not args.no_ct:
    if os.path.exists(args.ct_file):
        print(f"\nLoading CT volume: {args.ct_file} ...")
        ct_source = XMLImageDataReader(FileName=[args.ct_file])
        ct_source.UpdatePipeline()
        ct_info = ct_source.GetDataInformation()
        ct_bounds = ct_info.GetBounds()
        print(f"  CT bounds (m): X [{ct_bounds[0]:.6f}, {ct_bounds[1]:.6f}]  "
              f"Y [{ct_bounds[2]:.6f}, {ct_bounds[3]:.6f}]  "
              f"Z [{ct_bounds[4]:.6f}, {ct_bounds[5]:.6f}]")
        print(f"  CT opacity multiplier: {args.ct_opacity}")
    else:
        print(f"\nWARNING: CT file not found: {args.ct_file}")
        print("  Continuing without CT volume rendering.")
else:
    print("\nCT volume rendering disabled (--no-ct)")

# ---------------------------------------------------------------------------
# CT Transfer Function — X-ray style: grayscale, bones only
# ---------------------------------------------------------------------------
ct_lut = None
ct_pwf = None

if ct_source is not None:
    ct_lut = GetColorTransferFunction("HU")
    ct_pwf = GetOpacityTransferFunction("HU")
    apply_ct_transfer_function(ct_lut, ct_pwf,
                               preset=args.ct_preset,
                               ct_opacity=args.ct_opacity)

# ---------------------------------------------------------------------------
# Streamline pipeline
# ---------------------------------------------------------------------------
vel_min = args.range[0] if args.range else 0.0
vel_max = args.range[1] if args.range else 3.0

print("\nLoading inlet patch as streamline seed source ...")
foam_inlet = OpenFOAMReader(FileName=FOAM_FILE)
foam_inlet.MeshRegions = ["patch/inlet"]
foam_inlet.CellArrays = []
foam_inlet.UpdatePipeline(timesteps[init_idx])
n_inlet_pts = foam_inlet.GetDataInformation().GetNumberOfPoints()

stride = _bi.max(1, n_inlet_pts // args.seed_count)
inlet_seeds = MaskPoints(Input=foam_inlet)
inlet_seeds.OnRatio = stride
inlet_seeds.RandomSampling = True
inlet_seeds.MaximumNumberofPoints = args.seed_count

cell2point = CellDatatoPointData(Input=foam)
cell2point.CellDataArraytoprocess = ["U"]

stream = StreamTracerWithCustomSource(Input=cell2point, SeedSource=inlet_seeds)
stream.Vectors = ["POINTS", "U"]
stream.MaximumStreamlineLength = args.max_length
stream.IntegrationDirection = "FORWARD"
stream.IntegratorType = "Runge-Kutta 4-5"
stream.MaximumSteps = 20000

tubes = Tube(Input=stream)
tubes.Radius = args.tube_radius
tubes.NumberofSides = 8
tubes.VaryRadius = "Off"

calc_umag = Calculator(Input=tubes)
calc_umag.Function = "mag(U)"
calc_umag.ResultArrayName = "Umag"

print(f"  Seeds: ~{args.seed_count}, tube radius: {args.tube_radius}")
print(f"  Max propagation length: {args.max_length} m")

# ---------------------------------------------------------------------------
# Colormap setup — streamlines
# ---------------------------------------------------------------------------
lut = GetColorTransferFunction("Umag")
lut.ApplyPreset("Fast", True)
lut.RescaleTransferFunction(vel_min, vel_max)
lut.AutomaticRescaleRangeMode = "Never"

print(f"Colorbar range: [{vel_min:.4g}, {vel_max:.4g}]")

# ---------------------------------------------------------------------------
# Show pipeline in view
# ---------------------------------------------------------------------------
SetActiveView(view)

# 1. CT volume rendering (behind everything — rendered first)
if ct_source is not None:
    ct_display = Show(ct_source, view)
    ct_display.Representation = "Volume"
    ColorBy(ct_display, ("POINTS", "HU"))
    ct_display.LookupTable = ct_lut
    ct_display.VolumeRenderingMode = "GPU Based"
    vp = ct_display.GetProperty("VolumeProperty")
    if vp is not None:
        vp.SetInterpolationType(1)  # trilinear

# 2. Wall patch semi-transparent overlay
print("Loading wall patch for context ...")
foam_wall = OpenFOAMReader(FileName=FOAM_FILE)
foam_wall.MeshRegions = ["patch/wall"]
foam_wall.CellArrays = []

wall_display = Show(foam_wall, view)
wall_display.Representation = "Surface"
wall_display.DiffuseColor = [0.85, 0.85, 0.85]
wall_display.Opacity = 0.15
wall_display.SetScalarColoring(None, 0)

# 3. Streamline tubes colored by Umag (foreground)
display = Show(calc_umag, view)
display.Representation = "Surface"
ColorBy(display, ("POINTS", "Umag"))
display.LookupTable = lut

# ---------------------------------------------------------------------------
# Colorbar
# ---------------------------------------------------------------------------
display.SetScalarBarVisibility(view, True)
scalar_bar = GetScalarBar(lut, view)
scalar_bar.Title = "Velocity Magnitude"
scalar_bar.ComponentTitle = "[m/s]"
scalar_bar.TitleColor = [0.9, 0.9, 0.9]
scalar_bar.LabelColor = [0.9, 0.9, 0.9]
scalar_bar.RangeLabelFormat = "%.2f"
scalar_bar.AutomaticLabelFormat = 0
scalar_bar.TitleFontSize = 48
scalar_bar.LabelFontSize = 40

# Force LUT range after all ColorBy calls
lut.RescaleTransferFunction(vel_min, vel_max)
lut.AutomaticRescaleRangeMode = "Never"

# ---------------------------------------------------------------------------
# Time annotation
# ---------------------------------------------------------------------------
time_text = Text()
_init_ts_idx = int(round(frame_start / _bi.max(1, total_output_frames - 1) * (len(timesteps) - 1)))
time_text.Text = f"t = {timesteps[_init_ts_idx]:.3f} s"
time_display = Show(time_text, view)
time_display.WindowLocation = "Any Location"
time_display.FontSize = 56
time_display.Position = [0.70, 0.92]
time_display.Color = [1.0, 1.0, 0.6]

# Mode label
mode_label = Text()
mode_label.Text = "Spinning 3D — CT + Streamlines"
mode_display = Show(mode_label, view)
mode_display.WindowLocation = "Any Location"
mode_display.FontSize = 48
mode_display.Position = [0.02, 0.92]
mode_display.Color = [0.8, 0.8, 0.8]

# ---------------------------------------------------------------------------
# Initial camera setup
# ---------------------------------------------------------------------------
elev_rad = math.radians(args.elevation)
view.CameraFocalPoint = center
view.CameraViewUp = [0, 0, 1]
view.CameraParallelScale = max_dim * 0.6

azimuth_rad = 0.0
view.CameraPosition = [
    center[0] + cam_dist * math.cos(elev_rad) * math.cos(azimuth_rad),
    center[1] + cam_dist * math.cos(elev_rad) * math.sin(azimuth_rad),
    center[2] + cam_dist * math.sin(elev_rad),
]
view.ResetCamera()
cam = view.GetActiveCamera()
cam.Zoom(1.1)

# ---------------------------------------------------------------------------
# Waveform inset setup
# ---------------------------------------------------------------------------
WAVEFORM_CSV = os.path.join(CASE_DIR, "constant", "volumetricFlowRate.csv")
WAVEFORM_POS = (960, 780)  # position for 1920x1080 frame
WAVEFORM_ENABLED = False

if _HAS_OVERLAY_DEPS and os.path.exists(WAVEFORM_CSV):
    print("Loading waveform data for inset ...")
    _csv = np.genfromtxt(WAVEFORM_CSV, delimiter=',', skip_header=1)
    _csv_time = _csv[:, 0]
    _csv_flow = _csv[:, 1]

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
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        ax.plot(_t_fine, _q_smooth, color='cyan', linewidth=3)
        q_now = float(np.interp(t_cur, _t_fine, _q_smooth))
        ax.axvline(t_cur, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.plot(t_cur, q_now, 'o', color='red', markersize=12, zorder=5)
        ax.set_xlim(_csv_time[0], _csv_time[-1])
        ax.set_ylim(*_q_range)
        ax.set_xlabel('Time [s]', color='white', fontsize=12)
        ax.set_ylabel('Q [mL/s]', color='white', fontsize=12)
        ax.set_title('Inlet Flow Rate', color='white', fontsize=14, pad=4)
        ax.tick_params(colors='white', labelsize=10)
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
    print(f"  Waveform: {len(_csv_time)} points")

# ---------------------------------------------------------------------------
# Render spinning frames
# ---------------------------------------------------------------------------
n_frames = frame_end - frame_start
total_angle = 360.0 * args.orbits
print(f"\nRendering {n_frames} spinning CT+streamline frames, {args.orbits} orbit(s) ...")
print(f"  {len(timesteps)} simulation timesteps mapped across {total_output_frames} output frames")
print(f"  Elevation: {args.elevation} deg, output: {OUTPUT_DIR}/")

prev_ts_idx = -1

for f in range(frame_start, frame_end):
    frac = f / _bi.max(1, total_output_frames - 1)

    # Map to nearest simulation timestep
    ts_idx = int(round(frac * (len(timesteps) - 1)))
    t = timesteps[ts_idx]

    # Only update pipeline when simulation timestep actually changes
    if ts_idx != prev_ts_idx:
        animScene.AnimationTime = t
        prev_ts_idx = ts_idx

    time_text.Text = f"t = {t:.3f} s"

    # Camera orbit
    azimuth_deg = frac * total_angle
    azimuth_rad = math.radians(azimuth_deg)

    view.CameraPosition = [
        center[0] + cam_dist * math.cos(elev_rad) * math.cos(azimuth_rad),
        center[1] + cam_dist * math.cos(elev_rad) * math.sin(azimuth_rad),
        center[2] + cam_dist * math.sin(elev_rad),
    ]
    view.CameraFocalPoint = center
    view.CameraViewUp = [0, 0, 1]

    view.Update()
    Render()

    fname = os.path.join(OUTPUT_DIR, f"frame_{f:04d}.png")
    SaveScreenshot(fname, view, ImageResolution=RESOLUTION)

    # Composite waveform inset
    if WAVEFORM_ENABLED:
        frame_img = Image.open(fname).convert('RGBA')
        waveform_img = _create_waveform_overlay(t)
        canvas = Image.new('RGBA', frame_img.size, (0, 0, 0, 0))
        canvas.paste(waveform_img, WAVEFORM_POS)
        result = Image.alpha_composite(frame_img, canvas)
        result.convert('RGB').save(fname)

    frame_num = f - frame_start + 1
    if frame_num % 10 == 0 or frame_num == 1 or f == frame_end - 1:
        print(f"  [{frame_num}/{n_frames}] ts[{ts_idx}] t={t:.3f}s az={azimuth_deg:.0f}deg -> {fname}")

print(f"\nDone! {n_frames} frames saved to {OUTPUT_DIR}/")
if args.start is None and args.end is None:
    print(f"\nAssemble movie with:")
    print(f"  ffmpeg -framerate {args.fps} -i {OUTPUT_DIR}/frame_%04d.png \\")
    print(f"      -c:v libx264 -pix_fmt yuv420p -crf 18 \\")
    print(f"      pulsatile_spinning_ct.mp4")
