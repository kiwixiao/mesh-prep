"""
OpenFOAM Case File Generators

Pure-function module that generates OpenFOAM dictionary files for LES blood
flow simulation of aortic geometry.  No Qt dependencies — each function
returns a plain string ready to be written to disk.

Mesh pipeline: blockMesh → snappyHexMesh → checkMesh → pimpleFoam
"""

import math
from typing import Optional

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
) -> str:
    """pimpleFoam with adaptive time-stepping, maxCo=0.5, 2 cardiac cycles.

    Parameters
    ----------
    outlet_patches : list[str], optional
        If provided, adds surfaceFieldValue function objects that compute
        flow rate (phi) through each outlet patch at runtime.
    geo_name : str
        Geometry name prefix for patch names.
    """
    outlet_block = ""
    if outlet_patches:
        outlet_block = _generate_outlet_functions(outlet_patches, geo_name)

    return (
        _header("dictionary", "controlDict")
        + f"""
application     pimpleFoam;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         1.6;    // 2 cardiac cycles (0.8 s each)

deltaT          1e-05;

writeControl    adjustableRunTime;
writeInterval   0.01;

purgeWrite      0;

writeFormat     binary;
writePrecision  8;

writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           0.5;
maxDeltaT       1e-03;

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
    default         backward;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss LUST grad(U);
    div(phi,nut)    Gauss limitedLinear 1;
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
        relTol          0.01;
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
        relTol          0.1;
    }

    "(U|nut)Final"
    {
        $U;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors    2;
    nCorrectors         1;
    nNonOrthogonalCorrectors 0;
    pRefCell            0;
    pRefValue           0;
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


# ── constant/ dictionaries ───────────────────────────────────────────

def generate_transport_properties() -> str:
    """Blood: kinematic viscosity nu = 3.3e-06 m^2/s (mu=3.5e-3 Pa.s, rho=1060 kg/m^3)."""
    return (
        _header("dictionary", "transportProperties")
        + """
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] 3.3e-06;

// ************************************************************************* //
"""
    )


def generate_turbulence_properties() -> str:
    """LES with Smagorinsky subgrid-scale model."""
    return (
        _header("dictionary", "turbulenceProperties")
        + """
simulationType  LES;

LES
{
    LESModel        Smagorinsky;

    SmagorinskyCoeffs
    {
        Cs              0.1;
    }

    delta           cubeRootVol;

    cubeRootVolCoeffs
    {
        deltaCoeff      1;
    }

    printCoeffs     on;
}

// ************************************************************************* //
"""
    )


def generate_mass_flow_rate_csv() -> str:
    """Template aortic pulsatile waveform (0.8 s period, ~20 points, values in kg/s).

    This is a representative aortic flow waveform scaled for a typical
    descending aorta cross-section.  Replace with patient-specific data.
    """
    return """\
time,massFlowRate
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
) -> str:
    """Velocity BC with constant velocity active and pulsatile commented.

    Parameters
    ----------
    patch_names : list[str]
        All patch names (raw, without geometry prefix).
    inlet_normals : dict, optional
        {patch_name: (nx, ny, nz)} for inlet patches.  The normal points
        *outward* from the domain (STL/CFD convention).  Velocity is set in
        the *opposite* direction (flow enters the domain).  Magnitude ~0.3 m/s.
    geo_name : str
        Geometry name prefix for patch names.
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
                vx = -normal[0] / mag * 0.3 + 0.0  # +0.0 avoids -0.0
                vy = -normal[1] / mag * 0.3 + 0.0
                vz = -normal[2] / mag * 0.3 + 0.0
            else:
                vx, vy, vz = 0.3, 0.0, 0.0

            entries.append(
                f"    {full_name}\n"
                f"    {{\n"
                f"        // === CONSTANT VELOCITY (active) ===\n"
                f"        type            fixedValue;\n"
                f"        value           uniform ({vx:.6f} {vy:.6f} {vz:.6f});\n"
                f"\n"
                f"        // === PULSATILE FLOW (uncomment below, comment out above) ===\n"
                f"        // type            flowRateInletVelocity;\n"
                f"        // massFlowRate    csvFile;\n"
                f"        // massFlowRateCoeffs\n"
                f"        // {{\n"
                f"        //     nHeaderLine     1;\n"
                f"        //     refColumn       0;\n"
                f"        //     componentColumns (1);\n"
                f"        //     separator       \",\";\n"
                f"        //     mergeSeparators no;\n"
                f"        //     file            \"massFlowRate.csv\";\n"
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


def generate_allrun() -> str:
    """Full mesh + solve pipeline:
    blockMesh → snappyHexMesh → checkMesh → pimpleFoam
    """
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
    sed -i.bak -E 's/^([[:space:]]+)([0-9][0-9]*_[a-zA-Z][a-zA-Z0-9_]*)/\1"\2"/' "$boundary"
    rm -f "${boundary}.bak"
fi

runApplication checkMesh
runApplication pimpleFoam

echo "Opening ParaView..."
paraview --script=visualize.py &

# ----------------------------------------------------------------- end-of-file
"""


def generate_allclean() -> str:
    """Remove generated mesh, results, and logs."""
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
            vx = -normal[0] / mag * 0.3 + 0.0
            vy = -normal[1] / mag * 0.3 + 0.0
            vz = -normal[2] / mag * 0.3 + 0.0
            vel_strs.append(
                f"{name}: ({vx:.6f}, {vy:.6f}, {vz:.6f}) m/s  |U|=0.30 m/s"
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
