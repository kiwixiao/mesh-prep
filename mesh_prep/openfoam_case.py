"""
OpenFOAM Case File Generators

Pure-function module that generates OpenFOAM dictionary files for LES blood
flow simulation of aortic geometry.  No Qt dependencies — each function
returns a plain string ready to be written to disk.

Mesh pipelines:
  - pMesh (default):  cfMesh pMesh (Docker) → checkMesh → pimpleFoam
  - snappyHexMesh:    blockMesh → snappyHexMesh → checkMesh → pimpleFoam
"""

import json
import math
from pathlib import Path
from typing import Optional


# ── Template loading ────────────────────────────────────────────────

def get_template_dir(template_name: str = "les_aorta") -> Path:
    """Return path to the named template directory."""
    return Path(__file__).parent / "templates" / template_name


def load_template(template_name: str = "les_aorta") -> dict:
    """Load and return the template config.json as a dict."""
    config_path = get_template_dir(template_name) / "config.json"
    with open(config_path) as f:
        return json.load(f)

def _of_key(name: str) -> str:
    """Quote an OpenFOAM dictionary key if it starts with a digit."""
    if name and name[0].isdigit():
        return f'"{name}"'
    return name


# ── OpenFOAM header ──────────────────────────────────────────────────

_HEADER_TEMPLATE = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_};
    object      {object_};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _header(class_: str, object_: str) -> str:
    return _HEADER_TEMPLATE.format(class_=class_, object_=object_)


# ── Patch classifier ─────────────────────────────────────────────────

def classify_patch(name: str) -> str:
    """Classify patch name as 'inlet', 'outlet', or 'wall'."""
    lower = name.lower()
    if "inlet" in lower:
        return "inlet"
    if "outlet" in lower:
        return "outlet"
    return "wall"


# ── system/ dictionaries ─────────────────────────────────────────────

def generate_block_mesh_dict(bounds: tuple) -> str:
    """Background hex mesh sized to 120% of geometry bounding box.

    Parameters
    ----------
    bounds : tuple
        (xmin, xmax, ymin, ymax, zmin, zmax) of the geometry.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # Expand bbox by 20% on each side
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    dz = (zmax - zmin) or 1.0
    pad_x = 0.2 * dx
    pad_y = 0.2 * dy
    pad_z = 0.2 * dz

    x0 = xmin - pad_x
    x1 = xmax + pad_x
    y0 = ymin - pad_y
    y1 = ymax + pad_y
    z0 = zmin - pad_z
    z1 = zmax + pad_z

    # Target ~20 cells on shortest axis, proportional on others
    shortest = min(dx, dy, dz)
    cell_size = shortest / 20.0
    nx = max(4, round((x1 - x0) / cell_size))
    ny = max(4, round((y1 - y0) / cell_size))
    nz = max(4, round((z1 - z0) / cell_size))

    return (
        _header("dictionary", "blockMeshDict")
        + f"""
convertToMeters 1;

vertices
(
    ({x0:.6f} {y0:.6f} {z0:.6f})
    ({x1:.6f} {y0:.6f} {z0:.6f})
    ({x1:.6f} {y1:.6f} {z0:.6f})
    ({x0:.6f} {y1:.6f} {z0:.6f})
    ({x0:.6f} {y0:.6f} {z1:.6f})
    ({x1:.6f} {y0:.6f} {z1:.6f})
    ({x1:.6f} {y1:.6f} {z1:.6f})
    ({x0:.6f} {y1:.6f} {z1:.6f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    allBoundary
    {{
        type patch;
        faces
        (
            (3 7 6 2)
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
    )


def generate_snappy_hex_mesh_dict(
    stl_filename: str,
    patch_names: list[str],
    location_in_mesh: tuple,
) -> str:
    """snappyHexMesh with refinement + snapping + prism layers on wall.

    Parameters
    ----------
    stl_filename : str
        Name of the multi-solid STL in constant/triSurface/.
    patch_names : list[str]
        All patch names (e.g. ['inlet', 'outlet_1', 'outlet_2', 'wall']).
    location_in_mesh : tuple
        (x, y, z) point guaranteed to be inside the mesh domain.
    """
    stl_stem = stl_filename.rsplit(".", 1)[0] if "." in stl_filename else stl_filename

    geo_name = stl_stem

    # Build per-region refinement entries
    region_entries = []
    for name in patch_names:
        ptype = classify_patch(name)
        level = 2 if ptype == "wall" else 1
        region_entries.append(
            f"            {_of_key(name)}\n"
            f"            {{\n"
            f"                level ({level} {level});\n"
            f"                patchInfo {{ type {_patch_type(ptype)}; }}\n"
            f"            }}"
        )
    regions_block = "\n".join(region_entries)

    # Layer patches — only wall
    wall_patches = [n for n in patch_names if classify_patch(n) == "wall"]
    layer_entries = "\n".join(
        f"        {_of_key(f'{geo_name}_{n}')} {{ nSurfaceLayers 4; }}" for n in wall_patches
    )

    lx, ly, lz = location_in_mesh

    return (
        _header("dictionary", "snappyHexMeshDict")
        + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    "{stl_filename}"
    {{
        type triSurfaceMesh;
        name "{geo_name}";
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       100000;
    maxGlobalCells      2000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;

    features
    (
    );

    refinementSurfaces
    {{
        "{geo_name}"
        {{
            level (1 2);
            regions
            {{
{regions_block}
            }}
        }}
    }}

    refinementRegions
    {{
    }}

    locationInMesh ({lx:.6f} {ly:.6f} {lz:.6f});
}}

snapControls
{{
    nSmoothPatch            3;
    tolerance               2.0;
    nSolveIter              100;
    nRelaxIter              5;
    nFeatureSnapIter        10;
    implicitFeatureSnap     false;
    explicitFeatureSnap     true;
    multiRegionFeatureSnap  false;
}}

addLayersControls
{{
    relativeSizes       true;
    layers
    {{
{layer_entries}
    }}
    expansionRatio          1.2;
    finalLayerThickness     0.3;
    minThickness            0.1;
    nGrow                   0;
    featureAngle            130;
    nRelaxIter              5;
    nSmoothSurfaceNormals   1;
    nSmoothNormals          3;
    nSmoothThickness        10;
    maxFaceThicknessRatio   0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle      90;
    nBufferCellsNoExtrude   0;
    nLayerIter              50;
}}

meshQualityControls
{{
    #include "meshQualityDict"
}}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;

// ************************************************************************* //
"""
    )


def _patch_type(kind: str) -> str:
    """OpenFOAM patch type string for a given kind."""
    if kind == "wall":
        return "wall"
    return "patch"


def generate_mesh_quality_dict() -> str:
    """Conservative mesh quality thresholds for snappyHexMesh."""
    return (
        _header("dictionary", "meshQualityDict")
        + """
maxNonOrtho     65;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave      80;
minVol          1e-13;
minTetQuality   -1e30;
minArea         -1;
minTwist        0.02;
minDeterminant  0.001;
minFaceWeight   0.05;
minVolRatio     0.01;
minTriangleTwist -1;

nSmoothScale    4;
errorReduction  0.75;

relaxed
{
    maxNonOrtho 75;
}

// ************************************************************************* //
"""
    )


def _generate_outlet_functions(outlet_patches: list[str], geo_name: str = "") -> str:
    """Generate surfaceFieldValue function objects for outlet flow rates.

    Parameters
    ----------
    outlet_patches : list[str]
        Outlet patch names (e.g. ['outlet_1', 'outlet_2']).
    geo_name : str
        Geometry name prefix for patch names.

    Returns
    -------
    str
        OpenFOAM function-object entries (without outer braces).
    """
    entries = []
    for name in outlet_patches:
        full_name = f"{geo_name}_{name}" if geo_name else name
        entries.append(
            f"\n"
            f"    outletFlowRate_{name}\n"
            f"    {{\n"
            f"        type            surfaceFieldValue;\n"
            f"        libs            (fieldFunctionObjects);\n"
            f"        writeControl    timeStep;\n"
            f"        writeInterval   100;\n"
            f"        surfaceFormat   vtk;\n"
            f"        regionType      patch;\n"
            f"        name            {_of_key(full_name)};\n"
            f"        operation       sum;\n"
            f"        writeFields     false;\n"
            f"        fields          (phi);\n"
            f"    }}"
        )
    return "\n".join(entries)


def generate_control_dict(
    outlet_patches: Optional[list[str]] = None,
    geo_name: str = "",
    end_time: float = 1.6,
    delta_t: float = 1e-5,
    write_interval: float = 0.01,
    max_co: float = 2,
    max_delta_t: float = 1e-3,
) -> str:
    """pimpleFoam with adaptive time-stepping.

    Parameters
    ----------
    outlet_patches : list[str], optional
        If provided, adds surfaceFieldValue function objects that compute
        flow rate (phi) through each outlet patch at runtime.
    geo_name : str
        Geometry name prefix for patch names.
    end_time : float
        Simulation end time in seconds.
    delta_t : float
        Initial time step size.
    write_interval : float
        Write interval for adjustableRunTime.
    max_co : float
        Maximum Courant number for adaptive time-stepping.
    max_delta_t : float
        Maximum allowed time step.
    """
    outlet_block = ""
    if outlet_patches:
        outlet_block = _generate_outlet_functions(outlet_patches, geo_name)

    return (
        _header("dictionary", "controlDict")
        + f"""
application     pimpleFoam;

startFrom       latestTime;
startTime       0;

stopAt          endTime;
endTime         {end_time};

deltaT          {delta_t:.6g};

writeControl    adjustableRunTime;
writeInterval   {write_interval};

purgeWrite      0;

writeFormat     binary;
writePrecision  8;

writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           {max_co};
maxDeltaT       {max_delta_t:.6g};

functions
{{
    fieldAverage1
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        fields
        (
            U
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
        );
    }}

    wallShearStress1
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        patches         ({_of_key(f'{geo_name}_wall') if geo_name else 'wall'});
    }}
{outlet_block}
}}

// ************************************************************************* //
"""
    )


def generate_fv_schemes() -> str:
    """LES-appropriate discretisation schemes."""
    return (
        _header("dictionary", "fvSchemes")
        + """
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,nut)    bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ************************************************************************* //
"""
    )


def generate_fv_solution() -> str:
    """PIMPLE loop settings for LES."""
    return (
        _header("dictionary", "fvSolution")
        + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0;
        nCellsInCoarsestLevel 20;
        agglomerator    faceAreaPair;
        mergeLevels     1;
        cacheAgglomeration true;
        maxIter         100;
        minIter         1;
    }

    pFinal
    {
        $p;
        relTol          0;
    }

    "(U|nut)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-08;
        relTol          0.01;
    }

    "(U|nut)Final"
    {
        $U;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors    20;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    pRefCell            0;
    pRefValue           0;

    residualControl
    {
        U
        {
            tolerance   1e-5;
            relTol      0;
        }
        p
        {
            tolerance   1e-4;
            relTol      0;
        }
    }
}

relaxationFactors
{
    equations
    {
        U               0.7;
        UFinal          1;
        p               0.2;
        pFinal          1;
    }
}

// ************************************************************************* //
"""
    )


def generate_decompose_par_dict(n_procs: int = 4) -> str:
    """Parallel decomposition using scotch."""
    return (
        _header("dictionary", "decomposeParDict")
        + f"""
numberOfSubdomains {n_procs};

method          scotch;

// ************************************************************************* //
"""
    )


def generate_mesh_dict(
    stl_filename: str,
    patch_names: list[str],
    max_cell_size: float = 0.8,
    boundary_cell_size: float = 0.35,
    num_layers: int = 3,
    thickness_ratio: float = 0.5,
    max_first_layer_thickness: float = 0.05,
    wall_cell_size: Optional[float] = None,
) -> str:
    """cfMesh meshDict for polyhedral meshing with pMesh.

    Parameters
    ----------
    stl_filename : str
        Name of the multi-solid STL file in ``constant/triSurface/``.
    patch_names : list[str]
        All patch names (e.g. ``['inlet', 'outlet_1', 'outlet_2', 'wall']``).
    max_cell_size : float
        Global maximum cell size.
    boundary_cell_size : float
        Cell size on boundary surfaces.
    num_layers : int
        Number of boundary layer cells on wall patches.
    thickness_ratio : float
        Boundary layer expansion ratio.
    max_first_layer_thickness : float
        Maximum thickness of the first boundary layer cell.
    wall_cell_size : float, optional
        Local refinement cell size on wall patches. Defaults to
        ``boundary_cell_size * 0.5`` when None.
    """
    if wall_cell_size is None:
        wall_cell_size = boundary_cell_size * 0.5
    stl_path = f'"constant/triSurface/{stl_filename}"'

    # Wall patches get boundary layers and finer local refinement
    wall_patches = [n for n in patch_names if classify_patch(n) == "wall"]

    # Build patchBoundaryLayers entries for wall patches
    bl_entries = []
    for name in wall_patches:
        bl_entries.append(
            f"        {name}\n"
            f"        {{\n"
            f"            nLayers {num_layers};\n"
            f"            thicknessRatio {thickness_ratio};\n"
            f"            maxFirstLayerThickness {max_first_layer_thickness};\n"
            f"        }}"
        )
    bl_block = "\n".join(bl_entries)

    # Build renameBoundary entries — set correct patch types
    # cfMesh expects a dictionary keyed by original patch name, not a list
    rename_entries = []
    for name in patch_names:
        ptype = _patch_type(classify_patch(name))
        rename_entries.append(
            f"        {name}\n"
            f"        {{\n"
            f"            newName {name};\n"
            f"            type {ptype};\n"
            f"        }}"
        )
    rename_block = "\n".join(rename_entries)

    # Build localRefinement on wall for finer surface resolution
    refine_entries = []
    for name in wall_patches:
        refine_entries.append(
            f"        {name}\n"
            f"        {{\n"
            f"            cellSize {wall_cell_size};\n"
            f"        }}"
        )
    refine_block = "\n".join(refine_entries)

    return (
        _header("dictionary", "meshDict")
        + f"""
surfaceFile {stl_path};

maxCellSize {max_cell_size};

boundaryCellSize {boundary_cell_size};

boundaryLayers
{{
    patchBoundaryLayers
    {{
{bl_block}
    }}
}}

renameBoundary
{{
    defaultName fixedWalls;
    defaultType wall;

    newPatchNames
    {{
{rename_block}
    }}
}}

localRefinement
{{
{refine_block}
}}

// ************************************************************************* //
"""
    )


def generate_run_docker_sh() -> str:
    """Docker wrapper script to run pMesh + solver via ``opencfd/openfoam-default:2512``.

    Features:
    - ``tee`` on every stage for log files (``log.pMesh``, ``log.pimpleFoam``, …)
    - ``--name meshprep-<stage>`` on each container for external ``docker stop``
    - ``stdbuf -oL`` for line-buffered output (prevents tee delay)
    - ``trap`` handler to stop running containers on SIGTERM/SIGINT
    - ``set -o pipefail`` so pipe failures propagate correctly
    """
    return f"""\
#!/bin/bash
# Run cfMesh pMesh + pimpleFoam inside Docker (OpenFOAM v2512)
#
# Usage: ./run_docker.sh [nprocs]
#   nprocs  - number of CPU cores for parallel solve (default: 1 = serial)
#
# Examples:
#   ./run_docker.sh        # serial
#   ./run_docker.sh 8      # parallel on 8 cores
#
# Prerequisites:
#   - Docker installed and running
#   - Image: opencfd/openfoam-default:2512

set -eo pipefail

NPROCS="${{1:-1}}"
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="opencfd/openfoam-default:2512"
OF_BASHRC="source /usr/lib/openfoam/openfoam2512/etc/bashrc && cd /case"

cleanup() {{
    docker stop meshprep-pmesh meshprep-checkmesh meshprep-decompose meshprep-solver meshprep-reconstruct 2>/dev/null || true
}}
trap cleanup EXIT SIGTERM SIGINT

echo "=== Pulling Docker image (if needed) ==="
docker pull "$IMAGE"

echo "=== Running pMesh ==="
docker run --rm --name meshprep-pmesh \\
    -v "$CASE_DIR":/case \\
    -w /case \\
    "$IMAGE" \\
    bash -c "$OF_BASHRC && stdbuf -oL pMesh" 2>&1 | tee log.pMesh

echo "=== Running checkMesh ==="
docker run --rm --name meshprep-checkmesh \\
    -v "$CASE_DIR":/case \\
    -w /case \\
    "$IMAGE" \\
    bash -c "$OF_BASHRC && stdbuf -oL checkMesh" 2>&1 | tee log.checkMesh

if [ "$NPROCS" -gt 1 ]; then
    echo "=== Decomposing mesh for $NPROCS processors ==="
    docker run --rm --name meshprep-decompose \\
        -v "$CASE_DIR":/case \\
        -w /case \\
        "$IMAGE" \\
        bash -c "$OF_BASHRC && stdbuf -oL decomposePar" 2>&1 | tee log.decomposePar

    echo "=== Running pimpleFoam in parallel ($NPROCS cores) ==="
    echo "    Log: $CASE_DIR/solver.log  (monitor with: tail -f solver.log)"
    docker run --rm --name meshprep-solver \\
        -v "$CASE_DIR":/case \\
        -w /case \\
        "$IMAGE" \\
        bash -c "$OF_BASHRC && mpirun --allow-run-as-root -np $NPROCS pimpleFoam -parallel > /case/solver.log 2>&1"

    echo "=== Reconstructing parallel results ==="
    docker run --rm --name meshprep-reconstruct \\
        -v "$CASE_DIR":/case \\
        -w /case \\
        "$IMAGE" \\
        bash -c "$OF_BASHRC && stdbuf -oL reconstructPar" 2>&1 | tee log.reconstructPar
else
    echo "=== Running pimpleFoam (serial) ==="
    echo "    Log: $CASE_DIR/solver.log  (monitor with: tail -f solver.log)"
    docker run --rm --name meshprep-solver \\
        -v "$CASE_DIR":/case \\
        -w /case \\
        "$IMAGE" \\
        bash -c "$OF_BASHRC && pimpleFoam > /case/solver.log 2>&1"
fi

echo "=== Done ==="
echo "Results in: $CASE_DIR"
"""


# ── constant/ dictionaries ───────────────────────────────────────────

def generate_transport_properties(nu: float = 3.3e-6) -> str:
    """Blood: kinematic viscosity (default nu = 3.3e-06 m^2/s).

    Parameters
    ----------
    nu : float
        Kinematic viscosity in m^2/s.
    """
    return (
        _header("dictionary", "transportProperties")
        + f"""
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {nu:.6g};

// ************************************************************************* //
"""
    )


def generate_turbulence_properties(cs: float = 0.1) -> str:
    """LES with Smagorinsky subgrid-scale model.

    Parameters
    ----------
    cs : float
        Smagorinsky constant.
    """
    return (
        _header("dictionary", "turbulenceProperties")
        + f"""
simulationType  LES;

LES
{{
    LESModel        Smagorinsky;

    SmagorinskyCoeffs
    {{
        Cs              {cs};
    }}

    delta           cubeRootVol;

    cubeRootVolCoeffs
    {{
        deltaCoeff      1;
    }}

    printCoeffs     on;
}}

// ************************************************************************* //
"""
    )


def generate_volumetric_flow_rate_csv() -> str:
    """Template aortic pulsatile waveform (0.8 s period, ~20 points, values in m^3/s).

    This is a representative aortic flow waveform scaled for a typical
    descending aorta cross-section.  Replace with patient-specific data.
    """
    return """\
time,volumetricFlowRate
0.000,0.050
0.040,0.080
0.080,0.200
0.120,0.350
0.160,0.400
0.200,0.380
0.240,0.300
0.280,0.180
0.320,0.080
0.360,0.020
0.400,-0.010
0.440,-0.020
0.480,-0.010
0.520,0.010
0.560,0.030
0.600,0.040
0.640,0.045
0.680,0.048
0.720,0.050
0.760,0.050
0.800,0.050
"""


# ── 0/ boundary conditions ───────────────────────────────────────────

def generate_p(patch_names: list[str], geo_name: str = "") -> str:
    """Pressure BC: outlets fixed 0, inlets/wall zeroGradient."""
    entries = [
        "    allBoundary\n"
        "    {\n"
        "        type            zeroGradient;\n"
        "    }"
    ]
    for name in patch_names:
        full_name = _of_key(f"{geo_name}_{name}") if geo_name else name
        kind = classify_patch(name)
        if kind == "outlet":
            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        type            fixedValue;\n"
                f"        value           uniform 0;\n"
                f"    }}"
            )
        else:
            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        type            zeroGradient;\n"
                f"    }}"
            )

    patches_block = "\n\n".join(entries)

    return (
        _header("volScalarField", "p")
        + f"""
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{patches_block}
}}

// ************************************************************************* //
"""
    )


def generate_U(
    patch_names: list[str],
    inlet_normals: Optional[dict[str, tuple]] = None,
    geo_name: str = "",
    velocity_magnitude: float = 0.3,
) -> str:
    """Velocity BC with constant velocity active and pulsatile commented.

    Parameters
    ----------
    patch_names : list[str]
        All patch names (raw, without geometry prefix).
    inlet_normals : dict, optional
        {patch_name: (nx, ny, nz)} for inlet patches.  The normal points
        *outward* from the domain (STL/CFD convention).  Velocity is set in
        the *opposite* direction (flow enters the domain).
    geo_name : str
        Geometry name prefix for patch names.
    velocity_magnitude : float
        Inlet velocity magnitude in m/s.
    """
    if inlet_normals is None:
        inlet_normals = {}

    entries = [
        "    allBoundary\n"
        "    {\n"
        "        type            slip;\n"
        "    }"
    ]
    for name in patch_names:
        full_name = _of_key(f"{geo_name}_{name}") if geo_name else name
        kind = classify_patch(name)
        if kind == "inlet":
            normal = inlet_normals.get(name)
            if normal is not None:
                # Flow direction is opposite to the clip normal (into domain)
                mag = math.sqrt(sum(c * c for c in normal)) or 1.0
                vx = -normal[0] / mag * velocity_magnitude + 0.0  # +0.0 avoids -0.0
                vy = -normal[1] / mag * velocity_magnitude + 0.0
                vz = -normal[2] / mag * velocity_magnitude + 0.0
            else:
                vx, vy, vz = velocity_magnitude, 0.0, 0.0

            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        // === CONSTANT VELOCITY (active) ===\n"
                f"        type            fixedValue;\n"
                f"        value           uniform ({vx:.6f} {vy:.6f} {vz:.6f});\n"
                f"\n"
                f"        // === PULSATILE FLOW (uncomment below, comment out above) ===\n"
                f"        // type            flowRateInletVelocity;\n"
                f"        // volumetricFlowRate csvFile;\n"
                f"        // volumetricFlowRateCoeffs\n"
                f"        // {{\n"
                f"        //     nHeaderLine     1;\n"
                f"        //     refColumn       0;\n"
                f"        //     componentColumns (1);\n"
                f"        //     separator       \",\";\n"
                f"        //     mergeSeparators no;\n"
                f"        //     file            \"volumetricFlowRate.csv\";\n"
                f"        // }}\n"
                f"        // value           uniform (0 0 0);\n"
                f"    }}"
            )
        elif kind == "outlet":
            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        type            zeroGradient;\n"
                f"    }}"
            )
        else:
            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        type            noSlip;\n"
                f"    }}"
            )

    patches_block = "\n\n".join(entries)

    return (
        _header("volVectorField", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
{patches_block}
}}

// ************************************************************************* //
"""
    )


def generate_nut(patch_names: list[str], geo_name: str = "") -> str:
    """Subgrid-scale viscosity BC for LES."""
    entries = [
        "    allBoundary\n"
        "    {\n"
        "        type            calculated;\n"
        "        value           uniform 0;\n"
        "    }"
    ]
    for name in patch_names:
        full_name = _of_key(f"{geo_name}_{name}") if geo_name else name
        kind = classify_patch(name)
        if kind == "wall":
            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        type            nutUSpaldingWallFunction;\n"
                f"        value           uniform 0;\n"
                f"    }}"
            )
        else:
            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        type            calculated;\n"
                f"        value           uniform 0;\n"
                f"    }}"
            )

    patches_block = "\n\n".join(entries)

    return (
        _header("volScalarField", "nut")
        + f"""
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{patches_block}
}}

// ************************************************************************* //
"""
    )


# ── Shell scripts ────────────────────────────────────────────────────

def generate_env_sh() -> str:
    """OpenFOAM environment sourcing script (macOS default, Linux commented)."""
    return """\
#!/bin/bash
# OpenFOAM environment configuration
# Edit the path below for your installation.

# --- macOS (OpenFOAM-v2412.app) ---
source /Applications/OpenFOAM-v2412.app/Contents/Resources/etc/bashrc

# --- Linux (uncomment and adjust) ---
# source /opt/OpenFOAM/OpenFOAM-v2412/etc/bashrc
"""


def generate_allrun(meshing_method: str = "pmesh") -> str:
    """Full mesh + solve pipeline.

    Parameters
    ----------
    meshing_method : str
        ``"pmesh"``  — cfMesh pMesh (default, Docker-based).
        ``"snappy"`` — blockMesh + snappyHexMesh + digit-prefix sed fix.
    """
    if meshing_method == "pmesh":
        return _generate_allrun_pmesh()
    return _generate_allrun_snappy()


def _generate_allrun_pmesh() -> str:
    return """\
#!/bin/bash
cd "${0%/*}" || exit 1    # Run from this directory
source env.sh || { echo "Cannot source env.sh"; exit 1; }

# Source OpenFOAM run functions
. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

# pMesh requires cfMesh (not included in standard ESI OpenFOAM on macOS).
# If pMesh is not available natively, use ./run_docker.sh instead.
runApplication pMesh

runApplication checkMesh
runApplication pimpleFoam

echo "Opening ParaView..."
paraview --script=visualize.py &

# ----------------------------------------------------------------- end-of-file
"""


def _generate_allrun_snappy() -> str:
    return """\
#!/bin/bash
cd "${0%/*}" || exit 1    # Run from this directory
source env.sh || { echo "Cannot source env.sh"; exit 1; }

# Source OpenFOAM run functions
. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

runApplication blockMesh
runApplication snappyHexMesh -overwrite

# Fix digit-prefix bug: quote patch names starting with digits in boundary file
# (snappyHexMesh writes them unquoted, but OpenFOAM parser reads leading digits as numbers)
boundary="constant/polyMesh/boundary"
if [ -f "$boundary" ]; then
    sed -i.bak -E 's/^([[:space:]]+)([0-9][0-9]*_[a-zA-Z][a-zA-Z0-9_]*)/\\1"\\2"/' "$boundary"
    rm -f "${boundary}.bak"
fi

runApplication checkMesh
runApplication pimpleFoam

echo "Opening ParaView..."
paraview --script=visualize.py &

# ----------------------------------------------------------------- end-of-file
"""


def generate_allclean(meshing_method: str = "pmesh") -> str:
    """Remove generated mesh, results, and logs.

    Parameters
    ----------
    meshing_method : str
        ``"pmesh"`` or ``"snappy"`` — cleanup is the same for both.
    """
    return """\
#!/bin/bash
cd "${0%/*}" || exit 1    # Run from this directory

rm -rf 0.[0-9]* [1-9]* constant/polyMesh log.* processor*

# ----------------------------------------------------------------- end-of-file
"""


def generate_visualize_py(
    stl_filename: str,
    clips_data: list[dict],
) -> str:
    """Generate a ParaView Python script for post-processing visualization.

    Creates slice views at each inlet/outlet clip plane colored by velocity
    magnitude, plus the wall surface colored by wall shear stress.

    Parameters
    ----------
    stl_filename : str
        Original STL file name (used in annotation text).
    clips_data : list[dict]
        Each entry: {"name": str, "origin": (x,y,z), "normal": (nx,ny,nz)}.
    """
    # Build the per-clip slice code
    slice_blocks = []
    for i, clip in enumerate(clips_data):
        ox, oy, oz = clip["origin"]
        nx, ny, nz = clip["normal"]
        name = clip["name"]
        slice_blocks.append(
            f"# --- Slice: {name} ---\n"
            f"slice_{i} = Slice(Input=foam)\n"
            f"slice_{i}.SliceType = 'Plane'\n"
            f"slice_{i}.SliceType.Origin = [{ox:.6f}, {oy:.6f}, {oz:.6f}]\n"
            f"slice_{i}.SliceType.Normal = [{nx:.6f}, {ny:.6f}, {nz:.6f}]\n"
            f"slice_{i}Display = Show(slice_{i}, renderView)\n"
            f"slice_{i}Display.Representation = 'Surface'\n"
            f"ColorBy(slice_{i}Display, ('POINTS', 'U', 'Magnitude'))\n"
            f"slice_{i}Display.SetScalarBarVisibility(renderView, True)\n"
        )
    slices_code = "\n".join(slice_blocks)

    # Build the clip names for annotation
    clip_names_str = ", ".join(c["name"] for c in clips_data)

    # Determine .foam filename from stl_filename
    case_stem = stl_filename.rsplit(".", 1)[0] if "." in stl_filename else stl_filename

    return f'''\
#!/usr/bin/env pvpython
# ParaView visualization script for OpenFOAM case
# Generated by mesh-prep
#
# Usage:
#   pvpython visualize.py
#   -- or --
#   Open ParaView -> Tools -> Python Shell -> Run Script
#
# Requires: ParaView with OpenFOAM reader plugin

from paraview.simple import *

# Disable automatic camera reset
paraview.simple._DisableFirstRenderCameraReset()

# --- Load OpenFOAM case ---
foam_file = "{case_stem}.foam"
print(f"Loading OpenFOAM case from {{foam_file}} ...")
foam = OpenFOAMReader(FileName=foam_file)
foam.MeshRegions = ['internalMesh']
foam.CellArrays = ['U', 'p', 'wallShearStress']

# Go to last timestep
animationScene = GetAnimationScene()
animationScene.UpdateAnimationUsingDataTimeSteps()
animationScene.GoToLast()

# --- Set up render view ---
renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1400, 800]
renderView.Background = [0.18, 0.18, 0.25]

# --- Wall surface colored by wallShearStress magnitude ---
print("Showing wall surface with WSS ...")
foamDisplay = Show(foam, renderView)
foamDisplay.Representation = 'Surface'
foamDisplay.Opacity = 0.35

# Try WSS, fall back to pressure
try:
    ColorBy(foamDisplay, ('CELLS', 'wallShearStress', 'Magnitude'))
    wssLUT = GetColorTransferFunction('wallShearStress')
    wssLUT.ApplyPreset('Cool to Warm', True)
    foamDisplay.SetScalarBarVisibility(renderView, True)
    wssBar = GetScalarBar(wssLUT, renderView)
    wssBar.Title = 'Wall Shear Stress'
    wssBar.ComponentTitle = '[Pa]'
except Exception:
    ColorBy(foamDisplay, ('POINTS', 'p'))

# --- Inlet / Outlet slice planes ---
print("Creating slice planes at clip locations ...")
uLUT = GetColorTransferFunction('U')
uLUT.ApplyPreset('Jet', True)

{slices_code}

# Label the velocity color bar
uBar = GetScalarBar(uLUT, renderView)
uBar.Title = 'Velocity Magnitude'
uBar.ComponentTitle = '[m/s]'

# --- Camera setup (side view) ---
print("Setting camera ...")
renderView.ResetCamera()
camera = renderView.GetActiveCamera()
camera.Elevation(20)
camera.Azimuth(30)
camera.Zoom(1.3)

# --- Annotation ---
text = Text()
text.Text = "{case_stem}\\nPatches: {clip_names_str}, wall"
textDisplay = Show(text, renderView)
textDisplay.FontSize = 10
textDisplay.Position = [0.02, 0.92]

Render()

print("Visualization complete!")
print("Patches shown: {clip_names_str}, wall")
print("  - Wall: WSS magnitude (semi-transparent)")
print("  - Slices: velocity magnitude at each inlet/outlet plane")
print("  - Use mouse to rotate / zoom / pan")
'''


def generate_diagnose_patches_py(
    stl_filename: str,
    patch_names: list[str],
    inlet_normals: Optional[dict[str, tuple]] = None,
    geo_name: str = "",
    velocity_magnitude: float = 0.3,
) -> str:
    """Generate a ParaView script to visualize boundary patches and inlet velocity.

    Color-codes patches (red=inlet, green=outlet, blue=outlet_2+, gray=wall)
    and draws yellow arrow glyphs showing the inlet velocity direction.
    Saves diagnostic PNGs to a ``diagnostics/`` subdirectory.

    Parameters
    ----------
    stl_filename : str
        Original STL file name (for the .foam file).
    patch_names : list[str]
        Raw patch names (e.g. ``['inlet', 'outlet_1', 'wall']``).
    inlet_normals : dict, optional
        {patch_name: (nx, ny, nz)} outward-pointing normals for inlets.
    geo_name : str
        Geometry name prefix for ParaView region paths.
    velocity_magnitude : float
        Inlet velocity magnitude in m/s.
    """
    if inlet_normals is None:
        inlet_normals = {}

    case_stem = stl_filename.rsplit(".", 1)[0] if "." in stl_filename else stl_filename

    # Assign colors: red for first inlet, green for first outlet, blue for
    # second outlet, etc.  Wall is always gray.
    outlet_idx = 0
    outlet_colors = [
        (0.0, 0.8, 0.0),   # green
        (0.0, 0.3, 1.0),   # blue
        (1.0, 0.6, 0.0),   # orange
        (0.6, 0.0, 0.8),   # purple
    ]
    patch_configs = []  # (region, label, color, opacity, load_u)
    for name in patch_names:
        pv_name = f"{geo_name}_{name}" if geo_name else name
        kind = classify_patch(name)
        if kind == "inlet":
            patch_configs.append((f"patch/{pv_name}", pv_name, (1.0, 0.0, 0.0), 1.0, True))
        elif kind == "outlet":
            c = outlet_colors[outlet_idx % len(outlet_colors)]
            outlet_idx += 1
            patch_configs.append((f"patch/{pv_name}", pv_name, c, 1.0, False))
        else:
            patch_configs.append((f"patch/{pv_name}", pv_name, (0.7, 0.7, 0.7), 0.3, False))

    # Build patches list literal
    patches_literal = "[\n"
    for region, label, color, opacity, load_u in patch_configs:
        cr, cg, cb = color
        patches_literal += (
            f"    ('{region}', '{label}', "
            f"[{cr}, {cg}, {cb}], {opacity}, {load_u}),\n"
        )
    patches_literal += "]"

    # Build legend text
    legend_parts = []
    for _, label, _, _, _ in patch_configs:
        kind = classify_patch(label)
        if kind == "inlet":
            legend_parts.append(f"RED = {label}")
        elif kind == "outlet":
            legend_parts.append(f"colored = {label}")
        else:
            legend_parts.append(f"GRAY = {label}")
    legend_parts.append("YELLOW ARROWS = inlet velocity direction")
    legend_text = "  |  ".join(legend_parts)

    # Compute velocity string for the print at the end
    vel_strs = []
    for name in patch_names:
        if classify_patch(name) != "inlet":
            continue
        normal = inlet_normals.get(name)
        if normal is not None:
            mag = math.sqrt(sum(c * c for c in normal)) or 1.0
            vx = -normal[0] / mag * velocity_magnitude + 0.0
            vy = -normal[1] / mag * velocity_magnitude + 0.0
            vz = -normal[2] / mag * velocity_magnitude + 0.0
            vel_strs.append(
                f"{name}: ({vx:.6f}, {vy:.6f}, {vz:.6f}) m/s  |U|={velocity_magnitude:.2f} m/s"
            )

    vel_report = "\\n".join(vel_strs) if vel_strs else "No inlet normals provided"

    return f'''\
#!/usr/bin/env pvpython
"""
Diagnostic: Visualize boundary patches and inlet velocity direction.
Color-codes patches to verify inlet/outlet positions and flow direction.
Generated by mesh-prep.

Usage:
    pvpython diagnose_patches.py
"""

from paraview.simple import *
import os

output_dir = "diagnostics"
os.makedirs(output_dir, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

# --- Render view ---
renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1920, 1080]
renderView.Background = [0.92, 0.92, 0.92]
renderView.UseColorPaletteForBackground = 0
renderView.OrientationAxesVisibility = 1


def set_solid_color(display):
    """Disable scalar coloring so DiffuseColor is used."""
    display.SetScalarColoring(None, 0)


# --- Patch config: (region, label, color, opacity, load_U) ---
patches = {patches_literal}

foam_file = "{case_stem}.foam"
readers = {{}}
inlet_labels = []
for region_name, label, color, opacity, load_u in patches:
    print(f"Loading {{label}}...")
    reader = OpenFOAMReader(FileName=foam_file)
    reader.MeshRegions = [region_name]
    reader.CellArrays = ['U'] if load_u else []
    readers[label] = reader

    disp = Show(reader, renderView)
    disp.Representation = 'Surface'
    disp.DiffuseColor = color
    disp.Opacity = opacity
    if load_u:
        set_solid_color(disp)
        inlet_labels.append(label)

# Set animation to first timestep
animScene = GetAnimationScene()
animScene.UpdateAnimationUsingDataTimeSteps()
first_inlet = inlet_labels[0] if inlet_labels else list(readers.keys())[0]
ts = readers[first_inlet].TimestepValues
animScene.AnimationTime = ts[0]
print(f"Time set to t={{ts[0]:.3f}}s")

# --- Print patch bounding box centers ---
print("\\n--- Patch Locations (bounding box centers) ---")
for _, label, _, _, _ in patches:
    readers[label].UpdatePipeline(ts[0])
    info = readers[label].GetDataInformation()
    bounds = info.GetBounds()
    cx = (bounds[0] + bounds[1]) / 2.0
    cy = (bounds[2] + bounds[3]) / 2.0
    cz = (bounds[4] + bounds[5]) / 2.0
    print(f"  {{label:20s}}  center=({{cx:.4f}}, {{cy:.4f}}, {{cz:.4f}})")
    print(f"  {{'':20s}}  bounds X=[{{bounds[0]:.4f}}, {{bounds[1]:.4f}}]"
          f" Y=[{{bounds[2]:.4f}}, {{bounds[3]:.4f}}]"
          f" Z=[{{bounds[4]:.4f}}, {{bounds[5]:.4f}}]")

# --- Velocity arrows at each inlet ---
for inlet_label in inlet_labels:
    print(f"\\nAdding velocity arrows at {{inlet_label}}...")
    glyph = Glyph(Input=readers[inlet_label])
    glyph.OrientationArray = ['CELLS', 'U']
    glyph.ScaleArray = ['CELLS', 'U']
    glyph.GlyphType = 'Arrow'
    glyph.ScaleFactor = 20.0
    glyph.MaximumNumberOfSamplePoints = 150
    glyph_disp = Show(glyph, renderView)
    glyph_disp.DiffuseColor = [1.0, 1.0, 0.0]
    set_solid_color(glyph_disp)

# --- Legend ---
text = Text()
text.Text = "{legend_text}"
text_disp = Show(text, renderView)
text_disp.FontSize = 14
text_disp.Position = [0.01, 0.01]
text_disp.Color = [0.0, 0.0, 0.0]

# --- Save from multiple camera angles ---
views = [
    ("front",     0,   0),
    ("right",    90,   0),
    ("top",       0,  90),
    ("isometric", 35,  25),
]

print(f"\\nSaving {{len(views)}} views to {{output_dir}}/...")
for name, az, el in views:
    renderView.ResetCamera()
    camera = renderView.GetActiveCamera()
    camera.Azimuth(az)
    camera.Elevation(el)
    camera.Zoom(1.3)
    Render()
    fname = os.path.join(output_dir, f"patches_{{name}}.png")
    SaveScreenshot(fname, renderView, ImageResolution=[1920, 1080])
    print(f"  Saved {{fname}}")

# --- Velocity vectors on a mid-plane slice (first timestep) ---
print("\\nAdding velocity slice at first timestep...")

foam_vol = OpenFOAMReader(FileName=foam_file)
foam_vol.MeshRegions = ['internalMesh']
foam_vol.CellArrays = ['U']
foam_vol.UpdatePipeline(ts[0])
Hide(foam_vol, renderView)

vol_info = foam_vol.GetDataInformation()
vol_bounds = vol_info.GetBounds()
mid_z = (vol_bounds[4] + vol_bounds[5]) / 2.0
print(f"  Domain bounds: X=[{{vol_bounds[0]:.4f}},{{vol_bounds[1]:.4f}}]"
      f" Y=[{{vol_bounds[2]:.4f}},{{vol_bounds[3]:.4f}}]"
      f" Z=[{{vol_bounds[4]:.4f}},{{vol_bounds[5]:.4f}}]")

sliceZ = Slice(Input=foam_vol)
sliceZ.SliceType = 'Plane'
sliceZ.SliceType.Origin = [
    (vol_bounds[0] + vol_bounds[1]) / 2.0,
    (vol_bounds[2] + vol_bounds[3]) / 2.0,
    mid_z
]
sliceZ.SliceType.Normal = [0, 0, 1]

slice_disp = Show(sliceZ, renderView)
slice_disp.Representation = 'Surface'
ColorBy(slice_disp, ('CELLS', 'U', 'Magnitude'))
slice_disp.RescaleTransferFunctionToDataRange(False, True)
uLUT = GetColorTransferFunction('U')
uLUT.ApplyPreset('Cool to Warm (Diverging)', True)

slice_glyph = Glyph(Input=sliceZ)
slice_glyph.OrientationArray = ['CELLS', 'U']
slice_glyph.ScaleArray = ['CELLS', 'U']
slice_glyph.GlyphType = 'Arrow'
slice_glyph.ScaleFactor = 15.0
slice_glyph.MaximumNumberOfSamplePoints = 500
sg_disp = Show(slice_glyph, renderView)
sg_disp.DiffuseColor = [0.1, 0.1, 0.1]
set_solid_color(sg_disp)

for name, az, el in views:
    renderView.ResetCamera()
    camera = renderView.GetActiveCamera()
    camera.Azimuth(az)
    camera.Elevation(el)
    camera.Zoom(1.3)
    Render()
    fname = os.path.join(output_dir, f"velocity_slice_{{name}}.png")
    SaveScreenshot(fname, renderView, ImageResolution=[1920, 1080])
    print(f"  Saved {{fname}}")

print("\\nDone! Check diagnostics/ directory for:")
print("  patches_*.png        - Colored boundary patches with inlet velocity arrows")
print("  velocity_slice_*.png - Velocity magnitude + direction on mid-plane slice")
print(f"\\nInlet BC velocity: {vel_report}")
'''


def generate_plot_residuals_py() -> str:
    """Generate plot_residuals.py — solver convergence monitoring script.

    The script is case-independent: it reads all settings from system/ files
    at runtime.  Returns the full Python script as a string.
    """
    return r'''#!/usr/bin/env python
"""Parse solver.log and auto-update residual plots every 30s.

StarCCM+-style: x-axis is cumulative PIMPLE iteration (every outer iteration),
so you see the sawtooth convergence within each timestep.

Usage:
    python plot_residuals.py                    # live window + PNG, update every 5s
    python plot_residuals.py --headless         # PNG only (for background runs)
    python plot_residuals.py --interval 10      # custom update interval in seconds
"""

import re
import os
import sys
import time as pytime

HEADLESS = "--headless" in sys.argv

if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

LOG = "solver.log"
OUT = "residuals.png"
# Parse --interval N from command line, default 5s
_interval_idx = next((i for i, a in enumerate(sys.argv) if a == "--interval"), None)
INTERVAL = int(sys.argv[_interval_idx + 1]) if _interval_idx and _interval_idx + 1 < len(sys.argv) else 5
CASE_NAME = os.path.basename(os.getcwd())


def read_case_settings():
    """Read simulation settings from system/ dict files."""
    settings = {
        "application": "unknown",
        "end_time": None,
        "delta_t": None,
        "n_outer": 20,
        "u_tol": None,
        "p_tol": None,
    }

    # --- controlDict ---
    cd = os.path.join("system", "controlDict")
    if os.path.exists(cd):
        with open(cd) as f:
            for line in f:
                line = line.strip()
                m = re.match(r"application\s+(\S+);", line)
                if m:
                    settings["application"] = m.group(1)
                    continue
                m = re.match(r"endTime\s+([\d.e+-]+);", line)
                if m:
                    settings["end_time"] = float(m.group(1))
                    continue
                m = re.match(r"deltaT\s+([\d.e+-]+);", line)
                if m:
                    settings["delta_t"] = float(m.group(1))
                    continue

    # --- fvSolution ---
    fv = os.path.join("system", "fvSolution")
    if os.path.exists(fv):
        with open(fv) as f:
            text = f.read()
        m = re.search(r"nOuterCorrectors\s+(\d+)", text)
        if m:
            settings["n_outer"] = int(m.group(1))
        # Parse residualControl block — ESI uses "residualControl", Foundation uses "outerCorrectorResidualControl"
        blk = re.search(
            r"(?:outerCorrectorResidualControl|residualControl)\s*\{((?:[^{}]*\{[^{}]*\})*[^{}]*)\}",
            text,
        )
        if blk:
            body = blk.group(1)
            # Find U { tolerance ...; } and p { tolerance ...; }
            for var, key in [("U", "u_tol"), ("p", "p_tol")]:
                vm = re.search(
                    rf"\b{var}\s*\{{[^}}]*tolerance\s+([\d.e+-]+)",
                    body,
                )
                if vm:
                    settings[key] = float(vm.group(1))

    return settings


re_time = re.compile(r"^Time = ([\d.e+-]+)$")
re_courant = re.compile(r"Courant Number mean: ([\d.e+-]+) max: ([\d.e+-]+)")
re_cont = re.compile(r"time step continuity errors : sum local = ([\d.e+-]+), global = ([\d.e+-]+)")
re_ux = re.compile(r'Solving for Ux, Initial residual = ([\d.e+-]+)')
re_uy = re.compile(r'Solving for Uy, Initial residual = ([\d.e+-]+)')
re_uz = re.compile(r'Solving for Uz, Initial residual = ([\d.e+-]+)')
re_p = re.compile(r'Solving for p, Initial residual = ([\d.e+-]+)')
re_pimple = re.compile(r'PIMPLE: iteration (\d+)')


def parse_log():
    """Parse solver.log, returning per-PIMPLE-iteration data (StarCCM+ style)."""
    # Per cumulative iteration (every PIMPLE outer iter)
    all_ux, all_uy, all_uz, all_p = [], [], [], []
    cont_local = []
    # Track timestep boundaries (cumulative iter index where each timestep starts)
    ts_boundaries = []
    # Per timestep
    times, co_max, co_mean = [], [], []
    pimple_iters = []

    current_time = None
    cum_iter = 0
    ts_max_pimple = 0
    # Buffer for current PIMPLE iteration within a timestep
    iter_ux, iter_uy, iter_uz, iter_p = None, None, None, None

    with open(LOG) as f:
        for line in f:
            m = re_time.match(line.strip())
            if m:
                # Flush previous timestep
                if current_time is not None:
                    times.append(current_time)
                    pimple_iters.append(ts_max_pimple)
                current_time = float(m.group(1))
                ts_boundaries.append(cum_iter)
                ts_max_pimple = 0
                continue

            m = re_pimple.search(line)
            if m:
                pimple_num = int(m.group(1))
                ts_max_pimple = max(ts_max_pimple, pimple_num)
                # Each PIMPLE iteration = one cumulative iteration
                # Reset per-iter buffers
                iter_ux, iter_uy, iter_uz, iter_p = None, None, None, None
                continue

            m = re_courant.search(line)
            if m:
                co_mean.append(float(m.group(1)))
                co_max.append(float(m.group(2)))
                continue

            m = re_cont.search(line)
            if m:
                cont_local.append(float(m.group(1)))
                continue

            m = re_ux.search(line)
            if m:
                val = float(m.group(1))
                if iter_ux is None:
                    iter_ux = val
                    all_ux.append(val)
                    cum_iter = len(all_ux)
                continue
            m = re_uy.search(line)
            if m:
                val = float(m.group(1))
                if iter_uy is None:
                    iter_uy = val
                    all_uy.append(val)
                continue
            m = re_uz.search(line)
            if m:
                val = float(m.group(1))
                if iter_uz is None:
                    iter_uz = val
                    all_uz.append(val)
                continue
            m = re_p.search(line)
            if m:
                val = float(m.group(1))
                if iter_p is None:
                    iter_p = val
                    all_p.append(val)
                continue

    # Flush last timestep
    if current_time is not None:
        times.append(current_time)
        pimple_iters.append(ts_max_pimple)

    return dict(
        # Per cumulative PIMPLE iteration
        all_ux=all_ux, all_uy=all_uy, all_uz=all_uz, all_p=all_p,
        cont_local=cont_local,
        ts_boundaries=ts_boundaries,
        # Per timestep
        times=times, co_max=co_max, co_mean=co_mean,
        pimple_iters=pimple_iters,
    )


def _add_ts_lines(ax, boundaries, label=True):
    """Add faint vertical lines at timestep boundaries."""
    for i, b in enumerate(boundaries):
        ax.axvline(x=b, color="gray", alpha=0.15, lw=0.5,
                   label="timestep" if (i == 0 and label) else None)


def _add_time_twin_axis(ax, bounds, times, n_iter):
    """Add a twin x-axis on top showing simulation time [s]."""
    if len(bounds) < 2 or len(times) < 2:
        return
    ax2 = ax.twiny()
    # Pick ~6 evenly-spaced timesteps for tick labels
    n_ticks = min(6, len(times))
    indices = np.linspace(0, len(times) - 1, n_ticks, dtype=int)
    tick_positions = [bounds[i] if i < len(bounds) else n_iter for i in indices]
    tick_labels = [f"{times[i]:.4f}" for i in indices]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=7)
    ax2.set_xlabel("Time [s]", fontsize=8)


def update_plot(fig, axes, d, settings):
    """Clear and redraw all axes in-place."""
    n_iter = len(d["all_ux"])
    n_ts = len(d["times"])
    if n_iter < 2:
        return False

    n_outer = settings["n_outer"]

    # Remove twin axes from previous draw, then clear
    for child_ax in fig.axes[:]:
        if child_ax not in axes:
            child_ax.remove()
    for ax in axes:
        ax.clear()

    iters = np.arange(n_iter)
    bounds = d["ts_boundaries"]

    # Build dynamic title from case settings
    app = settings["application"]
    dt_str = f"dt={settings['delta_t']}" if settings["delta_t"] is not None else ""
    end_str = f"/{settings['end_time']}s" if settings["end_time"] is not None else "s"
    title = f"{CASE_NAME} - {app}"
    if dt_str:
        title += f"  |  {dt_str}"
    title += f"  |  t={d['times'][-1]:.4f}{end_str}  |  {n_ts} steps  |  {n_iter} iters"

    fig.suptitle(title, fontsize=13)

    # Row 0: Ux, Uy, Uz residuals (one subplot each)
    for i, (vals, name) in enumerate(
        [(d["all_ux"], "Ux"), (d["all_uy"], "Uy"), (d["all_uz"], "Uz")]
    ):
        ax = axes[i]
        n = min(len(vals), n_iter)
        ax.semilogy(iters[:n], vals[:n], lw=0.8, alpha=0.9)
        _add_ts_lines(ax, bounds)
        if settings["u_tol"] is not None:
            ax.axhline(y=settings["u_tol"], color="green", ls=":", alpha=0.5,
                       label="target")
            ax.legend(fontsize=7)
        ax.set_ylabel("Residual")
        ax.set_title(f"{name} Residual")
        ax.grid(True, alpha=0.3)
        _add_time_twin_axis(ax, bounds, d["times"], n_iter)
        if i >= 1:
            ax.set_xlabel("Cumulative PIMPLE iteration")

    # Row 1 left: Pressure residual
    ax = axes[3]
    n = min(len(d["all_p"]), n_iter)
    ax.semilogy(iters[:n], d["all_p"][:n], lw=0.8, alpha=0.9, color="C3")
    _add_ts_lines(ax, bounds)
    if settings["p_tol"] is not None:
        ax.axhline(y=settings["p_tol"], color="green", ls=":", alpha=0.5,
                   label="target")
        ax.legend(fontsize=7)
    ax.set_ylabel("Residual")
    ax.set_title("Pressure Residual")
    ax.set_xlabel("Cumulative PIMPLE iteration")
    ax.grid(True, alpha=0.3)
    _add_time_twin_axis(ax, bounds, d["times"], n_iter)

    # Row 1 right: Continuity errors
    ax = axes[4]
    ax.semilogy(d["cont_local"], lw=0.6, alpha=0.7, color="C4")
    ax.set_ylabel("Continuity Error")
    ax.set_xlabel("Cumulative PIMPLE iteration")
    ax.set_title("Continuity Errors (sum local)")
    ax.grid(True, alpha=0.3)

    # Row 1 right: Courant number (expanded to cumulative PIMPLE iter axis)
    ax = axes[5]
    nt = min(n_ts, len(d["co_max"]))
    if nt > 0 and len(bounds) >= nt:
        # Expand per-timestep Co to per-iteration using step function
        co_max_exp, co_mean_exp = [], []
        for ti in range(nt):
            start = bounds[ti]
            end = bounds[ti + 1] if ti + 1 < len(bounds) else n_iter
            n_fill = end - start
            co_max_exp.extend([d["co_max"][ti]] * n_fill)
            co_mean_exp.extend([d["co_mean"][ti]] * n_fill)
        x = np.arange(len(co_max_exp))
        ax.plot(x, co_max_exp, label="Co max", color="red", lw=0.8)
        ax.plot(x, co_mean_exp, label="Co mean", color="blue", lw=0.8)
        _add_ts_lines(ax, bounds)
    ax.set_ylabel("Courant Number")
    ax.set_xlabel("Cumulative PIMPLE iteration")
    ax.set_title("Courant Number")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2 right: PIMPLE outer iterations per timestep
    ax = axes[6]
    pi = d["pimple_iters"]
    np_t = min(n_ts, len(pi))
    if np_t > 1:
        dt_w = (d["times"][1] - d["times"][0]) * 0.8
        ax.bar(d["times"][:np_t], pi[:np_t], width=dt_w,
               color="steelblue", alpha=0.7)
    ax.set_ylabel("PIMPLE Outer Iters")
    ax.set_xlabel("Time [s]")
    ax.set_title(f"PIMPLE Outer Correctors (max={n_outer})")
    ax.axhline(y=n_outer, color="red", ls="--", alpha=0.5,
               label="nOuterCorrectors limit")
    if settings["end_time"] is not None:
        ax.axvline(x=settings["end_time"], color="purple", ls="--", alpha=0.5,
                   label=f"endTime={settings['end_time']}")
    ax.set_ylim(0, n_outer * 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    if not HEADLESS:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    if HEADLESS:
        plt.close(fig)
    return True


def _create_figure():
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.3)
    axes = [
        fig.add_subplot(gs[0, 0]),  # Ux
        fig.add_subplot(gs[0, 1]),  # Uy
        fig.add_subplot(gs[0, 2]),  # Uz
        fig.add_subplot(gs[1, 0]),  # p
        fig.add_subplot(gs[1, 1]),  # continuity
        fig.add_subplot(gs[1, 2]),  # courant
        fig.add_subplot(gs[2, :]),  # PIMPLE iters (full width)
    ]
    return fig, axes


def _sleep(seconds):
    if HEADLESS:
        pytime.sleep(seconds)
    else:
        plt.pause(seconds)


def main():
    settings = read_case_settings()

    if not HEADLESS:
        plt.ion()
        fig, axes = _create_figure()
        plt.show(block=False)

    prev_size = 0
    iteration = 0
    while True:
        if not HEADLESS and not plt.fignum_exists(fig.number):
            print("Window closed. Exiting.")
            break

        if not os.path.exists(LOG):
            print("Waiting for solver.log...")
            _sleep(INTERVAL)
            continue

        cur_size = os.path.getsize(LOG)
        if cur_size == prev_size and iteration > 0:
            print(f"[{pytime.strftime('%H:%M:%S')}] No new data (log size {cur_size}). Solver done?")
            d = parse_log()
            if d["all_ux"]:
                if HEADLESS:
                    fig, axes = _create_figure()
                update_plot(fig, axes, d, settings)
                print(f"Final plot saved: t={d['times'][-1]:.4f}s, "
                      f"{len(d['times'])} steps, {len(d['all_ux'])} iters")
            break

        prev_size = cur_size
        d = parse_log()
        if d["all_ux"]:
            n_ts = len(d["times"])
            n_it = len(d["all_ux"])
            co_str = f"Co max={max(d['co_max']):.1f}" if d["co_max"] else ""
            cont_str = f"cont={max(d['cont_local']):.1e}" if d["cont_local"] else ""
            if HEADLESS:
                fig, axes = _create_figure()
            update_plot(fig, axes, d, settings)
            print(f"[{pytime.strftime('%H:%M:%S')}] t={d['times'][-1]:.4f}s  "
                  f"steps={n_ts}  iters={n_it}  {co_str}  {cont_str}  -> {OUT}")
        else:
            print(f"[{pytime.strftime('%H:%M:%S')}] No data parsed yet")

        iteration += 1
        _sleep(INTERVAL)

    if not HEADLESS:
        plt.ioff()
        print("Done. Close window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
'''
