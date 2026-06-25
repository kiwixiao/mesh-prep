"""Standalone CLI to create an OpenFOAM case from an STL and a template.

Usage:
    mesh-prep-create --stl <path> --type les|rans --output <dir> [options]
"""

import argparse
import os
import shutil
import sys

from . import openfoam_case


def _read_stl_solid_names(stl_path: str) -> list[str]:
    """Extract solid names from an ASCII or binary STL file.

    Returns the list of solid names found.  For binary STL (no solid names),
    returns ``["solid"]``.
    """
    with open(stl_path, "rb") as f:
        header = f.read(80)

    # Check if ASCII by looking for 'solid' keyword at start
    try:
        text = header.decode("ascii", errors="ignore")
    except Exception:
        text = ""

    if text.strip().startswith("solid"):
        # ASCII STL — parse solid names
        names = []
        with open(stl_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("solid "):
                    name = stripped[6:].strip()
                    if name:
                        names.append(name)
        return names if names else ["solid"]
    else:
        return ["solid"]


def main():
    parser = argparse.ArgumentParser(
        description="Create an OpenFOAM case directory from STL + template.",
    )
    parser.add_argument("--stl", required=True, help="Path to input STL file")
    parser.add_argument(
        "--type", dest="sim_type", choices=["les", "rans"], default="les",
        help="Simulation type (default: les)",
    )
    parser.add_argument("--output", required=True, help="Output case directory")
    parser.add_argument("--delta-t", type=float, help="Override initial time step")
    parser.add_argument("--end-time", type=float, help="Override end time")
    parser.add_argument("--max-co", type=float, help="Override max Courant number")
    parser.add_argument("--max-delta-t", type=float, help="Override max deltaT")
    parser.add_argument("--nu", type=float, help="Override kinematic viscosity")
    parser.add_argument("--velocity", type=float, help="Override inlet velocity magnitude")
    parser.add_argument("--nprocs", type=int, help="Override number of processors")

    args = parser.parse_args()

    stl_path = os.path.abspath(args.stl)
    if not os.path.isfile(stl_path):
        print(f"Error: STL file not found: {stl_path}", file=sys.stderr)
        sys.exit(1)

    # Load template
    tmpl_name = "rans_aorta" if args.sim_type == "rans" else "les_aorta"
    cfg = openfoam_case.load_template(tmpl_name)

    # Apply overrides
    solver = cfg["solver"]
    if args.delta_t is not None:
        solver["deltaT"] = args.delta_t
    if args.end_time is not None:
        solver["endTime"] = args.end_time
    if args.max_co is not None:
        solver["maxCo"] = args.max_co
    if args.max_delta_t is not None:
        solver["maxDeltaT"] = args.max_delta_t
    if args.nu is not None:
        cfg["fluid"]["nu"] = args.nu
    if args.velocity is not None:
        cfg["inlet"]["velocityMagnitude"] = args.velocity
    if args.nprocs is not None:
        cfg["decompose"]["nProcs"] = args.nprocs

    # Read STL solid names as patch names
    solid_names = _read_stl_solid_names(stl_path)
    # If only one solid named 'solid', expect user to have named solids
    patch_names = solid_names if solid_names != ["solid"] else ["inlet", "outlet", "wall"]

    # Ensure 'wall' is in patch_names
    if "wall" not in [n.lower() for n in patch_names]:
        patch_names.append("wall")

    stl_filename = os.path.basename(stl_path)
    stl_stem = stl_filename.rsplit(".", 1)[0] if "." in stl_filename else stl_filename

    # Create directory structure
    case_dir = os.path.abspath(args.output)
    dirs = {
        "system": os.path.join(case_dir, "system"),
        "constant": os.path.join(case_dir, "constant"),
        "triSurface": os.path.join(case_dir, "constant", "triSurface"),
        "zero": os.path.join(case_dir, "0"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Copy STL (must already be in metres — use the GUI for unit conversion)
    shutil.copy2(stl_path, os.path.join(dirs["triSurface"], stl_filename))

    # Copy static template files
    tmpl_dir = openfoam_case.get_template_dir(tmpl_name)
    for static_name in ("fvSchemes", "fvSolution", "meshQualityDict"):
        src = tmpl_dir / static_name
        if src.exists():
            shutil.copy2(str(src), os.path.join(dirs["system"], static_name))

    # Copy volumetricFlowRate.csv
    csv_src = tmpl_dir / "volumetricFlowRate.csv"
    if csv_src.exists():
        shutil.copy2(str(csv_src), os.path.join(dirs["constant"], "volumetricFlowRate.csv"))

    # Generate system/ files
    mesh_p = cfg["mesh"]
    outlet_patches = [n for n in patch_names if openfoam_case.classify_patch(n) == "outlet"]

    system_files = {
        "meshDict": openfoam_case.generate_mesh_dict(
            stl_filename, patch_names,
            max_cell_size=mesh_p.get("maxCellSize", 0.8),
            boundary_cell_size=mesh_p.get("boundaryCellSize", 0.35),
            num_layers=mesh_p.get("nLayers", 3),
            thickness_ratio=mesh_p.get("thicknessRatio", 0.5),
            wall_cell_size=mesh_p.get("wallCellSize"),
        ),
        "controlDict": openfoam_case.generate_control_dict(
            outlet_patches=outlet_patches,
            geo_name="",
            end_time=solver.get("endTime", 1.6),
            delta_t=solver.get("deltaT", 1e-5),
            write_interval=solver.get("writeInterval", 0.01),
            max_co=solver.get("maxCo", 0.5),
            max_delta_t=solver.get("maxDeltaT", 1e-3),
            simulation_type=args.sim_type,
        ),
        "decomposeParDict": openfoam_case.generate_decompose_par_dict(
            n_procs=cfg["decompose"].get("nProcs", 4),
        ),
    }
    for name, content in system_files.items():
        path = os.path.join(dirs["system"], name)
        with open(path, "w") as f:
            f.write(content)

    # Generate constant/ files
    const_files = {
        "transportProperties": openfoam_case.generate_transport_properties(
            nu=cfg["fluid"].get("nu", 3.3e-6),
        ),
        "turbulenceProperties": openfoam_case.generate_turbulence_properties(
            simulation_type=args.sim_type,
            cs=cfg["turbulence"].get("Cs", 0.1),
        ),
    }
    for name, content in const_files.items():
        path = os.path.join(dirs["constant"], name)
        with open(path, "w") as f:
            f.write(content)

    # Generate 0/ boundary conditions
    vel_mag = cfg["inlet"].get("velocityMagnitude", 0.3)
    bc_files = {
        "p": openfoam_case.generate_p(patch_names, geo_name=""),
        "U": openfoam_case.generate_U(
            patch_names, inlet_normals=None, geo_name="",
            velocity_magnitude=vel_mag,
            simulation_type=args.sim_type,
        ),
        "nut": openfoam_case.generate_nut(
            patch_names, geo_name="",
            simulation_type=args.sim_type,
        ),
    }
    if args.sim_type == "rans":
        turb = cfg["turbulence"]
        bc_files["k"] = openfoam_case.generate_k(
            patch_names, geo_name="",
            k_value=turb.get("k", 3.375e-4),
        )
        bc_files["omega"] = openfoam_case.generate_omega(
            patch_names, geo_name="",
            omega_value=turb.get("omega", 30),
        )
    for name, content in bc_files.items():
        path = os.path.join(dirs["zero"], name)
        with open(path, "w") as f:
            f.write(content)

    # Generate shell scripts
    scripts = {
        "env.sh": openfoam_case.generate_env_sh(),
        "Allrun": openfoam_case.generate_allrun(meshing_method="pmesh"),
        "Allclean": openfoam_case.generate_allclean(meshing_method="pmesh"),
        "run_docker.sh": openfoam_case.generate_run_docker_sh(),
    }
    for name, content in scripts.items():
        path = os.path.join(case_dir, name)
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755)

    # ParaView .foam file
    foam_path = os.path.join(case_dir, f"{stl_stem}.foam")
    open(foam_path, "w").close()

    # plot_residuals.py
    plot_path = os.path.join(case_dir, "plot_residuals.py")
    with open(plot_path, "w") as f:
        f.write(openfoam_case.generate_plot_residuals_py())
    os.chmod(plot_path, 0o755)

    # Summary
    bc_list = list(bc_files.keys())
    print(f"OpenFOAM case created: {case_dir}")
    print(f"  Simulation type: {args.sim_type.upper()}")
    print(f"  Template: {cfg['name']}")
    print(f"  STL: {stl_filename}")
    print(f"  Patches: {', '.join(patch_names)}")
    print(f"  Boundary conditions (0/): {', '.join(bc_list)}")
    print(f"  Scripts: Allrun, Allclean, env.sh, run_docker.sh")
    print()
    print(f"  Docker:  cd {case_dir} && ./run_docker.sh")
    print(f"  Native:  cd {case_dir} && source env.sh && ./Allrun")


if __name__ == "__main__":
    main()
