# mesh-prep

Interactive STL surface repair and boundary-patch preparation for CFD
(OpenFOAM). Load a segmented surface (e.g. an aorta from CT), repair it, clip
and name inlets/outlets, and export a multi-solid STL plus a ready-to-run
OpenFOAM case.

## Install

Native install (macOS Apple Silicon / Intel, Linux), any Python ≥ 3.9:

```bash
pip install "git+https://github.com/kiwixiao/mesh-prep.git@dev"
mesh-prep                     # launch the GUI
```

Isolated app-style install (recommended for end users):

```bash
pipx install "git+https://github.com/kiwixiao/mesh-prep.git@dev"
# or: uv tool install "git+https://github.com/kiwixiao/mesh-prep.git@dev"
```

### Centerlines (optional, conda only)

Centerline computation uses [vmtk](http://www.vmtk.org), which has no PyPI
wheels and no Apple Silicon build. Everything else works without it — the
Compute Centerline button reports it is unavailable. To get centerlines, use
the conda environment instead:

```bash
./setup.sh                    # creates the 'mesh-prep' conda env (Rosetta on Apple Silicon)
conda run -n mesh-prep mesh-prep
```

## What it does

- **One compounding working mesh** — every operation (clip, cut, delete,
  trim, fill, smooth, repair) edits the current surface; Ctrl+Z steps back.
- **Repair**: clean, fix normals (consistent + outward), fill pinholes,
  make watertight (pymeshfix), delete non-manifold faces, remesh caps.
- **Boundary patches**: plane-clip trims and names inlets/outlets in one
  step; split bifurcation caps into components; rename/remove from the
  Objects tree; live normal/manifold/open-profile status.
- **Export**: multi-solid ASCII STL (one solid per patch + wall), patch
  metadata JSON (outward normals + centers), or a complete OpenFOAM case
  (cfMesh pMesh `meshDict`, boundary conditions, run scripts) for LES/RANS
  blood-flow simulation.

## Commands

```bash
mesh-prep [file.stl]          # GUI: repair + clip + export
mesh-prep-create --stl X.stl --type les|rans --output CASE_DIR
mesh-render --case CASE_DIR --mode velocity|streamlines|...   # needs ParaView
```

## Development

```bash
git clone https://github.com/kiwixiao/mesh-prep.git
cd mesh-prep
pip install -e ".[repair]"
python -m pytest tests/ -q
```
