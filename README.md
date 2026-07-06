# mesh-prep

Interactive STL surface repair and boundary-patch preparation for CFD
(OpenFOAM). Load a segmented surface (e.g. an aorta from CT), repair it, clip
and name inlets/outlets, and export a multi-solid STL plus a ready-to-run
OpenFOAM case.

There is **one working mesh**: every operation (clip, cut, delete, trim,
fill, smooth, repair) edits the current surface and compounds on the last;
`Ctrl+Z` steps back one operation. All panels always reflect the current mesh.

---

## Install

### macOS (Apple Silicon or Intel) — pip

Any Python ≥ 3.9 (native arm64 works and is fastest):

```bash
pip install "git+https://github.com/kiwixiao/mesh-prep.git"
mesh-prep                     # launch the GUI
```

Isolated app-style install (recommended for end users):

```bash
pipx install "git+https://github.com/kiwixiao/mesh-prep.git"
# or: uv tool install "git+https://github.com/kiwixiao/mesh-prep.git"
```

### Linux / Windows (WSL2) — pip

Install the Qt/OpenGL system libraries once, then pip as above:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libxkbcommon-x11-0 \
                        libfontconfig1 libdbus-1-3 libxcb-xinerama0
pip install "git+https://github.com/kiwixiao/mesh-prep.git"
mesh-prep
```

**WSL2 notes** (untested by the authors on this exact setup, standard WSLg
patterns):

- Windows 11 / Windows 10 21H2+ include **WSLg** — GUI windows open natively,
  nothing extra needed.
- Older WSL without WSLg: run an X server on Windows (e.g. VcXsrv) and
  `export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0`.
- If the 3D viewport is black or crashes on start, force software rendering:
  `LIBGL_ALWAYS_SOFTWARE=1 mesh-prep`.
- The Docker solver workflow works through Docker Desktop's WSL integration.

### Centerlines

Centerlines compute **natively** (a pure numpy/scipy/VTK reimplementation of the
Voronoi / maximal-inscribed-sphere method with branch decomposition), so they
work in every install above — no vmtk, no compilation, native Apple Silicon.
Seeds come automatically from your named inlet/outlet patch centers.

[vmtk](http://www.vmtk.org) is an optional fallback (used only if the native
engine fails and vmtk is installed). It has no PyPI wheels — install it via
conda if you want it:

```bash
git clone https://github.com/kiwixiao/mesh-prep.git
cd mesh-prep
./setup.sh                    # creates the 'mesh-prep' conda env (adds vmtk)
conda run -n mesh-prep mesh-prep
```

---

## Usage

### Workflow

1. **Load** an STL — the Load button, drag-and-drop anywhere on the window,
   or `mesh-prep path/to/file.stl`.
2. **Inspect** — the Objects tree (left) auto-detects open profiles,
   non-manifold edge groups, and disconnected pieces. The Geometry box
   (bottom-left) shows bounds and a live normals status
   (`✓ consistent, outward` is CFD-ready; a **Fix** button appears when red).
3. **Repair** (right panel, Mesh Repair tab) — recommended order:
   - delete debris: right-click small pieces in the tree → *Delete piece*;
   - *Make Watertight (MeshFix)* for holes/self-intersections — run this
     **before** clipping (it fills every opening, named ones included);
   - *Clean Mesh* / *Fix Normals* / *Auto-repair* as needed;
   - *Fill Pinholes* later for tiny defects (size-bounded, safe after
     clipping).
4. **Clip and name patches** (Clipping tab):
   - *Add Plane* → position the widget → *Confirm* → type a name
     (`inlet`, `outlet_1`, …). The clip trims the surface and names the cap
     in one step.
   - *2-Point Plane*: click the button (stays pressed), then **Shift+click
     two points** on the surface to define a cut plane along your view.
   - *Cut* splits without removing; *Trim* (lasso) deletes a drawn region;
     *Select faces* + Grow/Smooth/Delete for local edits.
   - A clip across a bifurcation caps both branches under one name —
     right-click the patch in the tree → *Split disconnected components*,
     then rename each side.
5. **Right-click in the Objects tree** for context actions:
   - named patch: Highlight / Rename / Split / Remesh (smoother cap) /
     Remove name;
   - open profile: Fill as named patch;
   - disconnected piece: Delete piece;
   - non-manifold group: Delete attached faces.
6. **Export** (Export Settings tab):
   - *Combined STL* — one multi-solid ASCII STL (one `solid` per patch +
     `wall`; solid names become mesher patch names);
   - *Separate STLs* — one file per patch;
   - *OpenFOAM case* — complete case directory: `constant/triSurface/`,
     cfMesh pMesh `system/meshDict`, boundary conditions (`0/p,U,nut[,k,omega]`),
     `Allrun`/`run_docker.sh`, LES or RANS from the bundled templates.
   - Every export also writes `clip_planes.json` with each patch's outward
     normal and center.

### Keys and mouse

| Action | Input |
|---|---|
| Rotate / zoom | drag / scroll in viewport |
| Undo last operation | `Ctrl+Z` |
| Two-point plane points | `Shift+click` ×2 (Esc cancels) |
| Lasso (Trim/Select modes) | `Shift+drag` |
| Flood select region | double-click in Select mode |

### Commands

```bash
mesh-prep [file.stl]          # GUI
mesh-prep-create --stl X.stl --type les|rans --output CASE_DIR   # headless case
mesh-render --case CASE_DIR --mode velocity|streamlines|...      # needs ParaView
```

---

## Troubleshooting

- **`xcb` plugin error on Linux/WSL** — install the apt packages listed
  above (missing Qt platform libraries).
- **Black/crashing 3D viewport in WSL or VMs** — `LIBGL_ALWAYS_SOFTWARE=1 mesh-prep`.
- **"pymeshfix is not installed"** — `pip install pymeshfix`
  (or `pip install "mesh-prep[repair] @ git+…"`)
- **Centerline needs a closed surface** — name an inlet and an outlet first
  (that caps their openings); the native engine tetrahedralizes the interior.

## Development

```bash
git clone https://github.com/kiwixiao/mesh-prep.git
cd mesh-prep
pip install -e ".[repair]"
python -m pytest tests/ -q    # 100+ headless engine tests
```
