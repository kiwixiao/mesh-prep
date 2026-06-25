"""
mesh-render — Unified CLI entry point for all ParaView batch render scripts.

Discovers pvbatch, resolves paths, queries total frames, and spawns parallel
pvbatch workers. On success runs ffmpeg to assemble the final mp4.

Modes:
    velocity           — Flow velocity volume rendering + wall
    streamlines        — Streamlines from inlet + wall
    image-velocity     — Flow velocity volume rendering + CT/MRI image
    image-streamlines  — Streamlines from inlet + CT/MRI image
    spinning           — Spinning camera view
    spinning-ct        — Spinning camera view + CT/MRI image
    pathlines          — Lagrangian particle pathlines (sequential)

Usage:
    mesh-render --case /path/to/case --mode velocity --range 0 3

    mesh-render --case /path/to/case --mode streamlines --seed-count 500

    mesh-render --case /path/to/case --mode image-velocity \\
        --ct-file /path/to/CT.vti --range 0 3

    mesh-render --case /path/to/case --mode image-streamlines \\
        --ct-file /path/to/CT.vti --seed-count 500
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Script path resolution via package __file__
# ---------------------------------------------------------------------------
import mesh_prep.render as _render_pkg

RENDER_DIR = Path(_render_pkg.__file__).parent

SCRIPTS = {
    "velocity":          RENDER_DIR / "movie.py",
    "streamlines":       RENDER_DIR / "streamlines.py",
    "pathlines":         RENDER_DIR / "pathlines.py",
    "spinning":          RENDER_DIR / "spinning.py",
    "image-streamlines": RENDER_DIR / "ct_flow.py",
    "image-velocity":    RENDER_DIR / "movie.py",
    "spinning-ct":       RENDER_DIR / "spinning_ct.py",
    # Legacy aliases
    "movie":             RENDER_DIR / "movie.py",
    "ct-flow":           RENDER_DIR / "ct_flow.py",
}

# Default fps per mode
DEFAULT_FPS = {
    "velocity":          16,
    "streamlines":       16,
    "pathlines":         16,
    "spinning":          25,
    "image-streamlines": 16,
    "image-velocity":    16,
    "spinning-ct":       25,
    "movie":             16,
    "ct-flow":           16,
}

# ---------------------------------------------------------------------------
# pvbatch discovery
# ---------------------------------------------------------------------------
def find_pvbatch() -> str:
    """Locate pvbatch executable with informative errors."""
    # 1. $PVBATCH env var (highest priority)
    if "PVBATCH" in os.environ:
        path = os.environ["PVBATCH"]
        if not os.path.isfile(path):
            sys.exit(
                f"ERROR: $PVBATCH points to non-existent file: {path}\n"
                f"  Fix: export PVBATCH=/correct/path/to/pvbatch"
            )
        if not os.access(path, os.X_OK):
            sys.exit(
                f"ERROR: $PVBATCH at {path} is not executable\n"
                f"  Fix: chmod +x {path}"
            )
        print(f"pvbatch: {path}  (from $PVBATCH)")
        return path

    # 2. Common install locations (checked in order)
    candidates = [
        "/Applications/ParaView-6.0.1.app/Contents/bin/pvbatch",   # macOS default
        "/Applications/ParaView-5.12.0.app/Contents/bin/pvbatch",
        "/Applications/ParaView-5.11.0.app/Contents/bin/pvbatch",
        "/usr/local/bin/pvbatch",                                    # Linux
        "/usr/bin/pvbatch",
        shutil.which("pvbatch"),                                     # PATH
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            print(f"pvbatch: {path}  (auto-detected)")
            return path

    sys.exit(
        "ERROR: pvbatch (ParaView batch renderer) not found.\n"
        "\n"
        "Options to fix:\n"
        "  1. Set the PVBATCH environment variable:\n"
        "       export PVBATCH=/path/to/pvbatch\n"
        "  2. Install ParaView from https://www.paraview.org/download/\n"
        "       macOS default: /Applications/ParaView-X.Y.Z.app/Contents/bin/pvbatch\n"
        "       Linux default: /usr/local/bin/pvbatch\n"
        "\n"
        "Searched paths:\n" + "\n".join(f"  {c}" for c in candidates if c)
    )


# ---------------------------------------------------------------------------
# Build script argument list
# ---------------------------------------------------------------------------
def build_script_args(args) -> list:
    """Build the argument list to pass to the pvbatch script (excluding --start/--end)."""
    script_args = ["--case", str(Path(args.case).resolve())]

    # Modes that support CT/MRI image overlay
    if args.mode in ("image-streamlines", "image-velocity", "spinning-ct",
                      "ct-flow", "movie", "velocity"):
        if args.ct_file is not None:
            ct_file = Path(args.ct_file).resolve()
            if not args.no_ct and not ct_file.exists():
                sys.exit(
                    f"ERROR: Image file not found: {ct_file}\n"
                    f"  Fix: pass the full path, e.g. --ct-file /path/to/CT_Pre_frame0_meters.vti"
                )
            script_args += ["--ct-file", str(ct_file)]
            if args.ct_opacity != 1.0:
                script_args += ["--ct-opacity", str(args.ct_opacity)]
        if args.ct_preset != "bone":
            script_args += ["--ct-preset", args.ct_preset]
        if args.no_ct:
            script_args.append("--no-ct")

    if args.range is not None:
        script_args += ["--range", str(args.range[0]), str(args.range[1])]

    if args.mode in ("velocity", "image-velocity", "movie"):
        script_args += ["--field", args.field]

    if args.mode in ("spinning", "spinning-ct"):
        script_args += ["--fps", str(args.fps)]
        script_args += ["--duration", str(args.duration)]
        script_args += ["--orbits", str(args.orbits)]
        script_args += ["--elevation", str(args.elevation)]

    if args.mode == "spinning":
        # spinning.py writes to spinning_<mode>; forward it so the frame dir
        # the workers write to matches the dir run_parallel hands to ffmpeg.
        script_args += ["--mode", args.spinning_mode]

    if args.mode in ("streamlines", "pathlines", "spinning",
                      "image-streamlines", "spinning-ct", "ct-flow"):
        script_args += ["--tube-radius", str(args.tube_radius)]
        script_args += ["--seed-count", str(args.seed_count)]

    if args.mode in ("streamlines", "spinning", "image-streamlines",
                      "spinning-ct", "ct-flow"):
        script_args += ["--max-length", str(args.max_length)]

    if args.mode == "pathlines":
        script_args += ["--inject-every", str(args.inject_every)]
        script_args += ["--trail-length", str(args.trail_length)]

    return script_args


# ---------------------------------------------------------------------------
# Query total frames
# ---------------------------------------------------------------------------
def query_total_frames(pvbatch: str, script: Path, script_args: list) -> int:
    """Run pvbatch <script> --total-frames to get the frame count."""
    cmd = [pvbatch, str(script)] + script_args + ["--total-frames"]
    print(f"Querying total frames: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(
            f"ERROR: --total-frames query failed (exit {result.returncode})\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    # The frame count is the last line of stdout
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        sys.exit("ERROR: --total-frames returned no output")
    try:
        return int(lines[-1])
    except ValueError:
        sys.exit(f"ERROR: --total-frames output not an integer: {lines[-1]!r}")


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
def run_parallel(
    pvbatch: str,
    script: Path,
    script_args: list,
    total_frames: int,
    workers: int,
    case_dir: Path,
    mode: str,
    fps: int,
    field: str,
    spinning_mode: str,
):
    """Spawn N pvbatch workers each handling a contiguous frame range."""
    if total_frames <= 0:
        sys.exit("ERROR: no frames to render (total_frames=0). "
                 "Check that the case has written time directories.")
    workers = min(workers, total_frames)
    chunk = total_frames // workers
    remainder = total_frames % workers

    ranges = []
    start = 0
    for i in range(workers):
        end = start + chunk + (1 if i < remainder else 0)
        if start < end:
            ranges.append((start, end))
        start = end

    print(f"\nSpawning {len(ranges)} workers for {total_frames} frames ...")

    renders_dir_logs = case_dir.parent / f"{case_dir.name}_renders"
    log_dir = renders_dir_logs / ".render_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    log_files = []
    for i, (s, e) in enumerate(ranges):
        cmd = [pvbatch, str(script)] + script_args + ["--start", str(s), "--end", str(e)]
        log_path = log_dir / f"worker_{i:02d}.log"
        log_file = open(log_path, "w")
        log_files.append((i, log_path, log_file))
        print(f"  Worker {i}: frames [{s}, {e}) -> {log_path.name}")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, s, e, proc))

    # Wait for all workers
    print(f"\nWaiting for {len(procs)} workers ...")
    failures = []
    for i, s, e, proc in procs:
        rc = proc.wait()
        _, log_path, log_file = log_files[i]
        log_file.close()
        if rc != 0:
            failures.append((i, s, e, log_path, rc))
            print(f"  Worker {i} FAILED (exit {rc}): frames [{s}, {e})")
        else:
            print(f"  Worker {i} done: frames [{s}, {e})")

    if failures:
        print(f"\nERROR: {len(failures)} worker(s) failed:")
        for i, s, e, log_path, rc in failures:
            print(f"  Worker {i} (frames [{s},{e}), exit {rc}) — last lines of {log_path.name}:")
            try:
                lines = log_path.read_text().splitlines()
                for line in lines[-10:]:
                    print(f"    {line}")
            except Exception:
                pass
        sys.exit(1)

    print(f"\nAll {len(procs)} workers completed successfully.")

    # Output lives in sibling renders dir: <case_parent>/<case_name>_renders/
    renders_dir = case_dir.parent / f"{case_dir.name}_renders"

    if mode in ("velocity", "image-velocity", "movie"):
        frames_subdir = renders_dir / field
        mp4_name = f"pulsatile_{field}.mp4"
    elif mode == "spinning":
        frames_subdir = renders_dir / f"spinning_{spinning_mode}"
        mp4_name = f"pulsatile_spinning_{spinning_mode}.mp4"
    elif mode == "spinning-ct":
        frames_subdir = renders_dir / "spinning_ct"
        mp4_name = "pulsatile_spinning_ct.mp4"
    elif mode in ("image-streamlines", "ct-flow"):
        frames_subdir = renders_dir / "ct_flow"
        mp4_name = "pulsatile_image_streamlines.mp4"
    else:
        frames_subdir = renders_dir / mode.replace("-", "_")
        mp4_name = f"pulsatile_{mode.replace('-', '_')}.mp4"

    mp4_path = renders_dir / mp4_name
    _run_ffmpeg(frames_subdir, mp4_path, fps)


def _run_ffmpeg(frames_dir: Path, mp4_path: Path, fps: int):
    """Assemble frames into an mp4 using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"\nWARNING: ffmpeg not found — skipping mp4 assembly.")
        print(f"  Assemble manually:")
        print(f"    ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png \\")
        print(f"        -c:v libx264 -pix_fmt yuv420p -crf 18 {mp4_path}")
        return

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(mp4_path),
    ]
    print(f"\nAssembling mp4: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  mp4 saved: {mp4_path}")
    else:
        print(f"  WARNING: ffmpeg exited {result.returncode} — mp4 may be incomplete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="mesh-render — Unified ParaView batch render CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mesh-render --case /path/to/AO_Native_001_RANS --mode spinning-ct \\
      --ct-file /Users/xiaz9n/openfoam/CT_Pre_frame0_meters.vti --workers 14 --range 0 3

  mesh-render --case /path/to/AO_Native_001_RANS --mode movie --workers 14

  mesh-render --case /path/to/AO_Native_001_RANS --mode spinning-ct \\
      --ct-file /Users/xiaz9n/openfoam/CT_Pre_frame0_meters.vti --start 40 --end 41

  mesh-render --case /path/to/AO_Native_001_RANS --mode spinning-ct --total-frames
""",
    )

    # Required
    parser.add_argument("--case", required=True,
                        help="OpenFOAM case directory (absolute or relative)")
    parser.add_argument("--mode", required=True,
                        choices=["velocity", "streamlines", "pathlines", "spinning",
                                 "image-streamlines", "image-velocity", "spinning-ct",
                                 "movie", "ct-flow"],
                        help="Render mode (velocity, streamlines, image-streamlines, image-velocity, spinning, spinning-ct)")

    # Parallel execution
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel pvbatch workers (default: 4)")

    # CT asset (ct-flow and spinning-ct only)
    parser.add_argument("--ct-file", default=None,
                        help="Full path to CT VTI file (required for ct-flow and spinning-ct modes)")
    parser.add_argument("--ct-opacity", type=float, default=1.0,
                        help="CT opacity multiplier 0-1 (default: 1.0)")
    parser.add_argument("--ct-preset", default="bone",
                        choices=["bone", "cardiac"],
                        help="CT preset: bone (skeleton only) or cardiac (heart/aortic wall visible)")
    parser.add_argument("--no-ct", action="store_true",
                        help="Disable CT rendering (debug)")

    # Field (movie mode)
    parser.add_argument("--field", default="velocity",
                        choices=["velocity", "vorticity", "pressure", "qcriterion",
                                 "pressure_gradient"],
                        help="Field for movie mode (default: velocity)")

    # Colorbar range
    parser.add_argument("--range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Velocity colorbar range")

    # Video parameters
    parser.add_argument("--fps", type=int, default=None,
                        help="Output framerate (default: 25 for spinning, 16 for others)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Video duration in seconds — spinning modes (default: 10.0)")
    parser.add_argument("--orbits", type=float, default=2.0,
                        help="Camera orbits — spinning modes (default: 2.0)")
    parser.add_argument("--elevation", type=float, default=30.0,
                        help="Camera elevation degrees — spinning modes (default: 30.0)")
    parser.add_argument("--spinning-mode", default="streamlines",
                        choices=["volume", "streamlines"],
                        help="spinning render content: volume or streamlines (default: streamlines)")

    # Partial render
    parser.add_argument("--start", type=int, default=None,
                        help="First frame index (for testing / manual chunking)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last frame index exclusive (for testing / manual chunking)")
    parser.add_argument("--total-frames", action="store_true",
                        help="Print total frame count and exit")

    # Streamline parameters
    parser.add_argument("--tube-radius", type=float, default=0.0001,
                        help="Streamline tube radius in meters (default: 0.0001)")
    parser.add_argument("--max-length", type=float, default=0.15,
                        help="Max streamline propagation length (default: 0.15)")
    parser.add_argument("--seed-count", type=int, default=300,
                        help="Streamline seed count (default: 300)")

    # Pathline parameters
    parser.add_argument("--inject-every", type=int, default=1,
                        help="Pathlines: inject new particles every N timesteps (default: 1)")
    parser.add_argument("--trail-length", type=int, default=5,
                        help="Pathlines: max trail length in timesteps (default: 5)")

    args = parser.parse_args()

    # Resolve fps default
    if args.fps is None:
        args.fps = DEFAULT_FPS[args.mode]

    case_dir = Path(args.case).resolve()
    if not case_dir.is_dir():
        sys.exit(f"ERROR: Case directory not found: {case_dir}")

    if args.mode in ("image-streamlines", "image-velocity", "spinning-ct",
                      "ct-flow") and not args.no_ct and args.ct_file is None:
        sys.exit(
            f"ERROR: --ct-file is required for --mode {args.mode}\n"
            f"  Fix: --ct-file /path/to/CT_Pre_frame0_meters.vti"
        )

    script = SCRIPTS[args.mode]
    if not script.exists():
        sys.exit(f"ERROR: Render script not found: {script}")

    pvbatch = find_pvbatch()
    script_args = build_script_args(args)

    # --total-frames: query and print, then exit
    if args.total_frames:
        n = query_total_frames(pvbatch, script, script_args)
        print(n)
        return

    # Single-process run (--start/--end specified)
    if args.start is not None or args.end is not None:
        cmd = [pvbatch, str(script)] + script_args
        if args.start is not None:
            cmd += ["--start", str(args.start)]
        if args.end is not None:
            cmd += ["--end", str(args.end)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # Pathlines: single sequential process (time-dependent — cannot parallelize)
    if args.mode == "pathlines":
        total_frames = query_total_frames(pvbatch, script, script_args)
        print(f"Total frames: {total_frames}")
        print(f"\nPathlines mode: running single sequential process (time-dependent)")

        renders_dir = case_dir.parent / f"{case_dir.name}_renders"
        log_dir = renders_dir / ".render_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "pathlines.log"

        cmd = [pvbatch, str(script)] + script_args
        print(f"Running: {' '.join(cmd)}")
        print(f"Log: {log_path}")

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            rc = proc.wait()

        if rc != 0:
            print(f"\nERROR: pathlines render failed (exit {rc})")
            print(f"  Last lines of {log_path.name}:")
            try:
                lines = log_path.read_text().splitlines()
                for line in lines[-15:]:
                    print(f"    {line}")
            except Exception:
                pass
            sys.exit(1)

        print(f"\nPathlines render complete.")
        frames_subdir = renders_dir / "pathlines"
        mp4_path = renders_dir / "pulsatile_pathlines.mp4"
        _run_ffmpeg(frames_subdir, mp4_path, args.fps)
        return

    # Full parallel render
    total_frames = query_total_frames(pvbatch, script, script_args)
    print(f"Total frames: {total_frames}")

    # Spinning sub-mode: forwarded to spinning.py (in build_script_args) and
    # reused here for the output path, so frame dir and ffmpeg input agree.
    spinning_mode = args.spinning_mode

    run_parallel(
        pvbatch=pvbatch,
        script=script,
        script_args=script_args,
        total_frames=total_frames,
        workers=args.workers,
        case_dir=case_dir,
        mode=args.mode,
        fps=args.fps,
        field=args.field,
        spinning_mode=spinning_mode,
    )


if __name__ == "__main__":
    main()
