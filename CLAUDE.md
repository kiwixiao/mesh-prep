# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> A parent `CLAUDE.md` at `/Users/xiaz9n/openfoam/CLAUDE.md` holds overall project context, tech stack, and OpenFOAM naming/parsing gotchas. This file is scoped to the `mesh-prep` package itself.

## Commands

All work runs inside the `mesh-prep` conda env (Python 3.9). On Apple Silicon the env is pinned to `osx-64` (Rosetta) because `vmtk` has no arm64 build — `setup.sh` handles this automatically.

```bash
./setup.sh                              # first-time: create env, install vmtk + pyvista(qt) + this package (editable)

conda run -n mesh-prep mesh-prep [file.stl]          # GUI: STL clipper
conda run -n mesh-prep mesh-prep-create --stl X.stl --type les|rans --output CASE_DIR
conda run -n mesh-prep mesh-render --case CASE_DIR --mode velocity|streamlines|pathlines|spinning|image-velocity|image-streamlines|spinning-ct
```

There is **no test suite** and no lint/typecheck configured. Verify changes by round-tripping an STL through the GUI and running the generated `run_docker.sh` against a sample case.

## Architecture

Three concerns live in separate modules, deliberately decoupled:

1. **Interactive clipping + GUI** — `stl_clipper.py` (~3k lines)
   - `STLClipperEngine` (line 208): headless clipping core — holds the STL, list of `ClipDefinition` planes/boxes, and does `recompute_all()` + export. Safe to instantiate without Qt.
   - `STLClipperApp` (line 823): PyQt5 `QMainWindow` embedding `pyvistaqt.QtInteractor`. Wraps the engine with pickers, sliders, cap-naming, and docker-solve buttons.
   - `CenterlineWorker` / `OpenFOAMWorker`: `QThread` subclasses so VMTK and the Docker solve don't freeze the UI.
   - Two export paths on the engine: `export_openfoam()` writes only the multi-solid STL + `clip_planes.json`; `export_openfoam_case()` additionally generates the full case tree via `openfoam_case` generators.

2. **Pure-function OpenFOAM generators** — `openfoam_case.py` (~2.4k lines, Qt-free)
   - ~30 `generate_*()` functions, one per OpenFOAM dict file. Each returns a string ready to write to disk. No side effects.
   - Shared helpers: `_of_key()` quotes digit-prefixed keys (OpenFOAM parser reads leading digits as numbers — see parent `CLAUDE.md` for the full rule), `_of_val()` formats floats, `_header()` writes `FoamFile` block, `classify_patch()` does substring matching for `inlet`/`outlet`/`wall`.
   - `load_template(name)` reads `templates/{name}/config.json` — parameter dict that feeds the generators.
   - Shell scripts are also generated here: `generate_allrun(meshing_method="pmesh"|"snappy")`, `generate_allclean`, `generate_run_docker_sh`, `generate_env_sh`. Post-solve plotting helpers (`generate_plot_residuals_py`, `parse_log`, `update_plot`) are embedded in this file and written alongside the case.

3. **ParaView batch rendering** — `render/` package + `render_cli.py`
   - Each file under `render/` is a standalone `pvbatch` script (imports `paraview.simple`), not part of the normal `mesh_prep` import graph — do **not** import them from other modules.
   - `render_cli.py` discovers `pvbatch`, resolves paths via `mesh_prep.render.__file__`, spawns parallel workers over a frame range, then calls `ffmpeg` to stitch frames. `SCRIPTS` and `OUTPUT_SUBDIRS` dicts (line 44/71) map each mode to the right script + output dir.
   - `pvbatch` location order: `$PVBATCH` → `PATH` → common install dirs.

4. **Centerlines** — `centerline.py`
   - Thin wrapper around `vmtk.vmtkscripts.vmtkCenterlines` with the `pointlist` seed selector. Inputs are flat coordinate lists; VMTK snaps to nearest surface point internally. Kept in its own module so `openfoam_case.py` stays VMTK-free.

## Templates

`templates/{les_aorta,rans_aorta}/` each contain:
- `config.json` — simulation parameters (mesh, solver, fluid, turbulence, inlet, decompose) consumed by `load_template()` and merged with generator defaults.
- `fvSchemes`, `fvSolution`, `meshQualityDict` — copied verbatim (not generated).
- `volumetricFlowRate.csv` — time-varying inlet flow; copied to `constant/`.

The LES template uses the Smagorinsky SGS model; the RANS template triggers extra `k`/`omega` BC generation in both `create_case.py` and `STLClipperEngine.export_openfoam_case()`.

Static files are included via the `[tool.setuptools.package-data]` block in `pyproject.toml` (`templates/**/*`).

## Case-generation flow (what hits disk)

```
STL + named clips           →  stl_clipper.export_openfoam_case()
  (or solid-named ASCII STL →  create_case.main())
        │
        ▼
  CASE_DIR/
    constant/triSurface/<stl>        (multi-solid ASCII STL, solid names == patch names for pMesh)
    system/{meshDict,controlDict,decomposeParDict,fvSchemes,fvSolution,meshQualityDict}
    constant/{transportProperties,turbulenceProperties,volumetricFlowRate.csv}
    0/{p,U,nut[,k,omega]}
    {env.sh, Allrun, Allclean, run_docker.sh, plot_residuals.py, <stem>.foam, clip_planes.json}
```

With pMesh (default) patch names come **directly from STL solid names** — there is no `{geometryName}_` prefix (that only happens with snappyHexMesh). See parent `CLAUDE.md` for the full naming rules and the sed workaround in the snappy Allrun.

## Docker solver workflow

`run_docker.sh` (generated by `generate_run_docker_sh`) pulls `opencfd/openfoam-default:2512` and runs the stages `pMesh → checkMesh → [decomposePar →] pimpleFoam → [reconstructPar]`, piping each to a `log.<stage>` file. Containers are named `meshprep-<stage>` so the GUI's `OpenFOAMWorker` can `docker stop` them on cancel (see `_DOCKER_CONTAINERS` at the top of `stl_clipper.py`).

## Conventions

- **No tests, no CI** — validate by running the GUI end-to-end or by generating a case and invoking `./run_docker.sh` on sample STLs in `/Users/xiaz9n/openfoam/`.
- **`.gitignore` excludes `*.stl` and `*.json`** — geometry assets and exported clip definitions are not committed.
- **Editable install**: `setup.sh` does `pip install -e .`, so edits are live without reinstall.
- **Python 3.9 required** (lower bound from `pyproject.toml`).
- **`paraview`/`pvbatch` is not a pip dep** — `mesh-render` expects a separate ParaView install on `PATH` or pointed to by `$PVBATCH`.
