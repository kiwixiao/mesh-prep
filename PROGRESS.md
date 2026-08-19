# mesh-prep PROGRESS

## NEXT STEP
GUI-test the review-fix batch (clip a mesh: normals stay green; fill an opening named inlet and export: 0/U direction correct; centerline tube now scale-relative), then merge cl to main on approval. After that: region-scoped shape-preserving remesh if the global pass is not enough, and pMesh end-to-end.

## LAST SESSION (2026-08-17)
Remesh quality batch: rebuilt the cap remesh (remesh_patch) so it works on real concave and multi-disc caps (min angle median 1.9 to 60 deg on the aorta outlet cap); found and fixed a pre-existing clip weld bug (cap rim from vtkCutter vs wall rim from vtkClipPolyData only matched by floating point luck; two-cylinder case had 316 open seam edges, now 0); made named caps click-pickable in Select mode; added whole-surface isotropic remesh (pyacvd/ACVD) as a Mesh Repair button. Earlier same day: NumPy 2 compatibility (ptp, linalg.solve), VTK 9.4+ gate for the Retina picking patch, mesh-prep --debug logging mode, flow extension (extrude open rim along cut normal).

## STATUS
| Work item | Stage |
| --- | --- |
| Single current_mesh model + patch_id labels | Released (v0.2.0) |
| Native centerline (vmtk-free) + bifurcation splitting | On main (b5549ad) |
| Mesh repair suite (normals, non-manifold, holes, MeshFix) | On main |
| Sharp-edge select + assign patch | On main |
| Flow extension (extrude open profile) | Committed (1c3085b) |
| NumPy 2 + VTK 9.4 compatibility, --debug mode | Committed (1c3085b) |
| Cap remesh (planar isotropic), clip weld fix, pickable caps | This commit |
| Remesh Surface (isotropic, pyacvd) | This commit; GUI test pending |
| Review-fix batch (11 bugs, 8 commits) | On cl; GUI test pending |
| Region remesh with fixed boundary + reprojection | Not started (candidate next) |
| pMesh end-to-end (pimpleFoam run from exported case) | Not tested |

## ENVIRONMENTS (two installs, both point at this source tree)
- Bare `mesh-prep` = base anaconda (python 3.10, numpy 2.2.6, VTK 9.6, pyacvd OK). This is the user's launcher.
- `conda run -n mesh-prep` = legacy env (python 3.9, numpy 1.26, VTK 9.2.6, vmtk). pyacvd NOT usable here (bundled OpenMP clashes with vmtk's, hard abort); removed from this env on purpose.
- Run the test suite under BOTH pythons before handoff. GUI smoke (offscreen) only works in the conda env.

## SESSION LOG
- 2026-08-19: multi-agent bug review (12 confirmed findings); fixed all in 8 commits on cl: cap winding harmonized at every cap merge + fill patch normals recorded (reversed-inlet-velocity bug), patch-name purge on all face-removing paths, stale selection cleared in 5 mesh-replacing handlers, scale-aware centerline tube, worker quiesce on load, run_docker NPROCS sync, visualize.py origin scaling, flood barriers survive smoothing, branch decomposition dedupe. Suites: conda 158 passed, base 156 passed.
- 2026-08-17: cap remesh rebuilt; clip weld fix; caps pickable; Remesh Surface (pyacvd) added; PROGRESS.md created. Two-point DPR fix and extrude verified in GUI.
- 2026-08-17 (earlier): NumPy 2 fixes; VTK 9.4 DPR gate; --debug mode; commit 1c3085b pushed to cl/dev/main.
- 2026-08-14: extrude built and verified headless, held for GUI test.
- 2026-08-05: mesh repair batch (55c13a8) and sharp-select batch (b5549ad) pushed; merged cl to dev and main.
- 2026-07-07: native centerline validated vs vmtk; surface splitting by branches.
- 2026-07-06: v0.2.0 released (single-mesh model), pip install verified.
