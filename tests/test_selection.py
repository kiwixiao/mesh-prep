import numpy as np
import pyvista as pv
from mesh_prep.stl_clipper import STLClipperEngine, plane_from_two_points

VIEW = np.array([[0.5, 0, 0, -1],
                 [0, 0.5, 0, -1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]], dtype=float)
VIEWPORT = (100, 100)
BIG_POLY = [(0, 0), (100, 0), (100, 100), (0, 100)]  # covers x,y in world [0,4]


def _two_triangles():
    # tri 0 around x~0.3 (winding -> +z normal); tri 1 around x~3.3 (-> -z normal)
    pts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                    (3, 0, 0), (3, 1, 0), (4, 0, 0)], dtype=float)
    faces = np.array([3, 0, 1, 2, 3, 3, 4, 5])
    eng = STLClipperEngine()
    eng.original_mesh = pv.PolyData(pts, faces)
    eng._wall_mesh = eng.original_mesh.copy()
    return eng


def test_select_all_when_front_only_false():
    eng = _two_triangles()
    ids = eng.select_cells_in_polygon(BIG_POLY, VIEW, VIEWPORT, (0, 0, 1), front_only=False)
    assert sorted(ids) == [0, 1]


def test_select_front_only_excludes_back_facing():
    eng = _two_triangles()
    vd = np.array([0, 0, 1.0])
    normals = eng.original_mesh.cell_normals
    expected = [i for i in (0, 1) if float(normals[i] @ vd) < 0]
    ids = eng.select_cells_in_polygon(BIG_POLY, VIEW, VIEWPORT, vd, front_only=True)
    assert sorted(ids) == sorted(expected)
    assert len(ids) == 1  # exactly one of the two faces the camera


def test_select_empty_polygon_returns_empty():
    eng = _two_triangles()
    assert eng.select_cells_in_polygon([(0, 0), (1, 1)], VIEW, VIEWPORT, (0, 0, 1)) == []


def test_select_any_vertex_inside_catches_boundary_cell():
    # Triangle with ONE vertex inside the polygon but its centroid OUTSIDE.
    # VIEW maps world (x, y) -> display (25x, 25y); poly covers display [0,30]^2.
    # vertex (0,0) -> (0,0) inside; centroid (1.33,0.67) -> (33.3,16.7) outside.
    pts = np.array([(0, 0, 0), (2, 0, 0), (2, 2, 0)], dtype=float)
    faces = np.array([3, 0, 1, 2])
    eng = STLClipperEngine()
    eng.original_mesh = pv.PolyData(pts, faces)
    eng._wall_mesh = eng.original_mesh.copy()
    poly = [(0, 0), (30, 0), (30, 30), (0, 30)]
    ids = eng.select_cells_in_polygon(poly, VIEW, VIEWPORT, (0, 0, 1), front_only=False)
    assert ids == [0]   # selected: a vertex is inside even though the centroid is not


def _grid():
    eng = STLClipperEngine()
    eng.original_mesh = pv.Plane(i_resolution=5, j_resolution=5).triangulate()
    eng._wall_mesh = eng.original_mesh.copy()
    return eng


def test_grow_one_ring_adds_point_neighbors():
    eng = _grid()
    seed = [0]
    expected = sorted(set(seed) | set(eng.original_mesh.cell_neighbors(0, connections="points")))
    assert eng.grow_cells(seed, rings=1) == expected


def test_grow_two_rings_superset_of_one():
    eng = _grid()
    one = set(eng.grow_cells([0], rings=1))
    two = set(eng.grow_cells([0], rings=2))
    assert one.issubset(two)
    assert len(two) > len(one)


def test_smooth_cells_relaxes_spike_and_leaves_rest_fixed():
    eng = STLClipperEngine()
    grid = pv.Plane(i_resolution=5, j_resolution=5).triangulate()
    # raise the most-connected interior point into a spike
    deg = [len(grid.point_neighbors(i)) for i in range(grid.n_points)]
    spike = int(np.argmax(deg))
    pts = grid.points.copy()
    pts[spike, 2] = 1.0
    grid.points = pts
    eng.original_mesh = grid
    eng._wall_mesh = grid.copy()

    sel = list(grid.point_cell_ids(spike))      # cells touching the spike
    movable = set()
    for cid in sel:
        movable.update(int(p) for p in grid.get_cell(cid).point_ids)
    unsel_pts = [i for i in range(grid.n_points) if i not in movable]
    before_unsel = grid.points[unsel_pts].copy()

    out = eng.smooth_cells(sel, iterations=5, relaxation=0.5)
    assert out is not None
    assert eng.original_mesh.points[spike, 2] < 0.5            # spike relaxed toward neighbors
    assert np.allclose(eng.original_mesh.points[unsel_pts], before_unsel)  # others fixed
    assert len(eng._trim_history) == 1
    assert eng.undo_trim() is True
    assert eng.original_mesh.points[spike, 2] == 1.0          # restored


def test_smooth_cells_empty_is_noop():
    eng = STLClipperEngine()
    eng.original_mesh = pv.Plane(i_resolution=3, j_resolution=3).triangulate()
    eng._wall_mesh = eng.original_mesh.copy()
    assert eng.smooth_cells([], iterations=5) is None
    assert len(eng._trim_history) == 0


def test_fast_adjacency_point_neighbors_match_pyvista():
    # The cached CSR point-neighbor lists must equal pyvista's point_neighbors
    # (unique), since smooth's Laplacian mean depends on the exact neighbor set.
    eng = STLClipperEngine()
    grid = pv.Plane(i_resolution=8, j_resolution=8).triangulate()
    eng.original_mesh = grid
    eng._ensure_adjacency()
    assert eng._adj_ok
    for p in (0, grid.n_points // 2, grid.n_points - 1):
        s = eng._adj_nbr_starts
        got = set(int(x) for x in eng._adj_nbr_by_point[s[p]:s[p + 1]])
        assert got == set(int(x) for x in grid.point_neighbors(p))


def test_fast_grow_matches_pyvista_two_rings():
    # Two-ring grow via the cached fast path must equal a pyvista reference.
    eng = STLClipperEngine()
    grid = pv.Plane(i_resolution=10, j_resolution=10).triangulate()
    eng.original_mesh = grid
    seed = [grid.n_cells // 2]
    ref = set(seed)
    for _ in range(2):
        nxt = set(ref)
        for c in ref:
            nxt.update(int(x) for x in grid.cell_neighbors(c, connections="points"))
        ref = nxt
    assert set(eng.grow_cells(seed, rings=2)) == ref


def test_smooth_fast_matches_reference_loop():
    # The vectorized Laplacian must match an explicit per-point loop to float
    # precision (same math, different summation order).
    eng = STLClipperEngine()
    grid = pv.Plane(i_resolution=8, j_resolution=8).triangulate()
    deg = [len(grid.point_neighbors(i)) for i in range(grid.n_points)]
    spike = int(np.argmax(deg))
    p0 = grid.points.copy()
    p0[spike, 2] = 1.0
    grid.points = p0
    eng.original_mesh = grid.copy()
    eng._wall_mesh = eng.original_mesh.copy()
    sel = list(grid.point_cell_ids(spike))

    rel, iters = 0.5, 5
    ref = grid.points.copy()
    movable = set()
    for c in sel:
        movable.update(int(p) for p in grid.get_cell(c).point_ids)
    movable = sorted(movable)
    neigh = {p: np.asarray([int(x) for x in grid.point_neighbors(p)]) for p in movable}
    for _ in range(iters):
        nw = ref.copy()
        for p in movable:
            nb = neigh[p]
            if len(nb):
                nw[p] = (1 - rel) * ref[p] + rel * ref[nb].mean(axis=0)
        ref = nw

    out = eng.smooth_cells(sel, iterations=iters, relaxation=rel)
    assert out is not None
    assert np.allclose(eng.original_mesh.points, ref, atol=1e-5)


def test_adjacency_cache_rebuilds_after_topology_change():
    # Identity-keyed cache must rebuild when original_mesh is replaced.
    eng = STLClipperEngine()
    eng.original_mesh = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
    eng._ensure_adjacency()
    first = eng._adj_for_mesh
    eng.original_mesh = pv.Plane(i_resolution=6, j_resolution=6).triangulate()
    eng._ensure_adjacency()
    assert eng._adj_for_mesh is eng.original_mesh
    assert eng._adj_for_mesh is not first
    assert eng._adj_tri.shape[0] == eng.original_mesh.n_cells


def test_delete_cells_removes_and_undo_restores():
    eng = STLClipperEngine()
    grid = pv.Plane(i_resolution=6, j_resolution=6).triangulate()
    eng.original_mesh = grid
    eng._wall_mesh = grid.copy()
    n0 = grid.n_cells
    sel = [0, 1, 2, 5, 9]
    out = eng.delete_cells(sel)
    assert out is not None
    assert eng.original_mesh.n_cells == n0 - len(sel)
    assert len(eng._trim_history) == 1
    assert eng.undo_trim() is True
    assert eng.original_mesh.n_cells == n0


def test_delete_cells_empty_is_noop():
    eng = STLClipperEngine()
    eng.original_mesh = pv.Plane(i_resolution=3, j_resolution=3).triangulate()
    eng._wall_mesh = eng.original_mesh.copy()
    assert eng.delete_cells([]) is None
    assert len(eng._trim_history) == 0


def test_delete_cells_whole_mesh_is_noop():
    eng = STLClipperEngine()
    grid = pv.Plane(i_resolution=3, j_resolution=3).triangulate()
    eng.original_mesh = grid
    eng._wall_mesh = grid.copy()
    assert eng.delete_cells(list(range(grid.n_cells))) is None
    assert len(eng._trim_history) == 0
    assert eng.original_mesh.n_cells == grid.n_cells


def _n_open(m):
    return m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                   manifold_edges=False, non_manifold_edges=False).n_cells


def test_cut_by_plane_keeps_surface_closed_and_records_curve():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24)
    eng.original_mesh = sph
    eng._wall_mesh = sph.copy()
    n0 = sph.n_cells
    assert _n_open(eng.original_mesh) == 0
    out = eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    assert out is not None
    assert _n_open(eng.original_mesh) == 0          # still closed: feature curve, not open profile
    assert eng.original_mesh.n_cells > n0           # split added triangles
    assert len(eng._feature_curves) == 1
    assert eng._feature_curves[0].n_cells > 0
    assert len(eng._trim_history) == 1


def test_cut_undo_restores_mesh_and_removes_curve():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=20, phi_resolution=20)
    eng.original_mesh = sph
    eng._wall_mesh = sph.copy()
    n0 = sph.n_cells
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    assert eng.undo_trim() is True
    assert eng.original_mesh.n_cells == n0
    assert len(eng._feature_curves) == 0


def test_cut_plane_miss_is_noop():
    eng = STLClipperEngine()
    sph = pv.Sphere(radius=1.0)
    eng.original_mesh = sph
    eng._wall_mesh = sph.copy()
    out = eng.cut_by_plane((0, 0, 100), (0, 0, 1))  # plane outside the sphere -> one side empty
    assert out is None
    assert len(eng._trim_history) == 0
    assert len(eng._feature_curves) == 0


def test_cut_by_box_keeps_surface_closed_and_records_curve():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=24, phi_resolution=24)
    eng.original_mesh = sph
    eng._wall_mesh = sph.copy()
    n0 = sph.n_cells
    # axis-aligned box cutting through the sphere: 6 planes as (normal, point)
    box = [((1, 0, 0), (0.3, 0, 0)), ((-1, 0, 0), (-0.3, 0, 0)),
           ((0, 1, 0), (0, 0.3, 0)), ((0, -1, 0), (0, -0.3, 0)),
           ((0, 0, 1), (0, 0, 0.3)), ((0, 0, -1), (0, 0, -0.3))]
    out = eng.cut_by_box(box)
    assert out is not None
    assert _n_open(eng.original_mesh) == 0          # still closed
    assert eng.original_mesh.n_cells > n0
    assert len(eng._feature_curves) == 1
    assert len(eng._trim_history) == 1


def test_load_stl_resets_cut_state(tmp_path):
    eng = STLClipperEngine()
    p1 = tmp_path / "s1.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(p1))
    eng.load_stl(str(p1))
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    assert len(eng._feature_curves) == 1
    assert len(eng._trim_history) == 1
    p2 = tmp_path / "s2.stl"
    pv.Sphere(theta_resolution=12, phi_resolution=12).save(str(p2))
    eng.load_stl(str(p2))
    assert len(eng._feature_curves) == 0
    assert len(eng._trim_history) == 0


def _sphere_engine(theta=24, phi=24):
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=theta, phi_resolution=phi)
    eng.original_mesh = sph
    eng._wall_mesh = sph.copy()
    return eng


def _cell_on_side(mesh, axis_idx, positive):
    col = mesh.cell_centers().points[:, axis_idx]
    idx = np.where(col > 0.3)[0] if positive else np.where(col < -0.3)[0]
    return int(idx[0])


def test_flood_select_no_cut_returns_whole_component():
    eng = _sphere_engine()
    n = eng.original_mesh.n_cells
    assert len(eng.flood_select(0)) == n          # closed sphere is one component


def test_flood_select_after_cut_separates_two_sides():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    mesh = eng.original_mesh
    top = _cell_on_side(mesh, 2, True)
    bot = _cell_on_side(mesh, 2, False)
    rt = set(eng.flood_select(top))
    rb = set(eng.flood_select(bot))
    assert rt and rb
    assert rt.isdisjoint(rb)                        # the cut is a wall
    assert len(rt) + len(rb) == mesh.n_cells        # partition the whole surface
    assert top in rt and bot in rb


def test_flood_select_two_cuts_bounded_region():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    eng.cut_by_plane((0, 0, 0), (1, 0, 0))
    mesh = eng.original_mesh
    region = eng.flood_select(0)
    assert 0 < len(region) < mesh.n_cells           # a quadrant, strictly smaller


def test_flood_select_survives_delete():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))         # remove one side
    m2 = eng.original_mesh
    r2 = eng.flood_select(0)                         # no crash, no leak beyond mesh
    assert 0 < len(r2) <= m2.n_cells


def test_flood_select_out_of_range_returns_empty():
    eng = _sphere_engine()
    assert eng.flood_select(-1) == []
    assert eng.flood_select(10 ** 9) == []


def test_flood_select_no_mesh_returns_empty():
    assert STLClipperEngine().flood_select(0) == []


def test_detect_open_profiles_counts_loops():
    eng = _sphere_engine()
    assert eng.detect_open_profiles() == []                # watertight sphere
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    assert len(eng.detect_open_profiles()) == 1            # one hole after delete


def test_detect_open_profiles_two_loops():
    eng = STLClipperEngine()
    cyl = pv.Cylinder(radius=1, height=4, resolution=40, capping=True).triangulate()
    eng.original_mesh = cyl
    eng._wall_mesh = cyl.copy()
    eng.cut_by_plane((0, 0, 1.0), (0, 0, 1))
    eng.cut_by_plane((0, 0, -1.0), (0, 0, 1))
    mid = int(np.where(np.abs(eng.original_mesh.cell_centers().points[:, 2]) < 0.5)[0][0])
    eng.delete_cells(eng.flood_select(mid))
    assert len(eng.detect_open_profiles()) == 2


def test_detect_open_profiles_no_mesh():
    assert STLClipperEngine().detect_open_profiles() == []


def test_detect_pieces_body_and_stray():
    eng = STLClipperEngine()
    sph = pv.Sphere(theta_resolution=20, phi_resolution=20).triangulate()
    stray = pv.PolyData(np.array([(10, 10, 10), (10.1, 10, 10), (10, 10.1, 10)], float),
                        np.array([3, 0, 1, 2]))
    combined = sph.merge(stray, merge_points=False)
    eng.original_mesh = combined
    eng._wall_mesh = combined.copy()
    pieces = eng.detect_pieces()
    assert len(pieces) == 2
    assert sorted(len(p) for p in pieces) == [1, 720]
    allids = sorted(i for p in pieces for i in p)
    assert allids == list(range(combined.n_cells))         # valid ids, full partition


def test_detect_pieces_single_body():
    eng = _sphere_engine()
    pieces = eng.detect_pieces()
    assert len(pieces) == 1
    assert len(pieces[0]) == eng.original_mesh.n_cells


def test_detect_pieces_no_mesh():
    assert STLClipperEngine().detect_pieces() == []


def test_detect_nonmanifold_edges():
    eng = STLClipperEngine()
    pts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)], float)
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 1, 3], [3, 0, 1, 4]])   # 3 tris share edge (0,1)
    eng.original_mesh = pv.PolyData(pts, faces)
    eng._wall_mesh = eng.original_mesh.copy()
    assert len(eng.detect_nonmanifold_edges()) == 1
    assert _sphere_engine().detect_nonmanifold_edges() == []        # clean sphere


def test_plane_from_two_points_basic():
    origin, normal = plane_from_two_points((0, 0, 0), (1, 0, 0), (0, 0, 1))
    assert np.allclose(origin, (0.5, 0, 0))
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert abs(np.dot(normal, (0, 0, 1))) < 1e-9        # plane is parallel to the view axis
    assert abs(np.dot(normal, (1, 0, 0))) < 1e-9        # plane contains the A-B line


def test_plane_from_two_points_generic_orthogonality():
    A = np.array([1.0, 1.0, 0.0]); B = np.array([2.0, 3.0, 1.0]); view = np.array([0.3, -0.2, 1.0])
    origin, normal = plane_from_two_points(A, B, view)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert abs(np.dot(normal, view)) < 1e-9
    assert abs(np.dot(normal, B - A)) < 1e-9
    assert np.allclose(origin, (A + B) / 2)


def test_plane_from_two_points_identical_points_none():
    assert plane_from_two_points((1, 2, 3), (1, 2, 3), (0, 0, 1)) is None


def test_plane_from_two_points_colinear_with_view_none():
    assert plane_from_two_points((0, 0, 0), (0, 0, 2), (0, 0, 1)) is None


def test_fill_profile_caps_the_hole():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    prof = eng.detect_open_profiles()[0]
    patch = eng.fill_profile(prof, "inlet")
    assert patch is not None
    assert patch.name == "inlet"
    assert patch.cap_mesh.n_cells > 0
    assert len(eng.filled_patches) == 1
    merged = eng.original_mesh.merge(patch.cap_mesh, merge_points=True)
    assert _n_open(merged) == 0                       # cap closes the hole


def test_unfilled_open_profiles_excludes_filled():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    assert len(eng.unfilled_open_profiles()) == 1
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    assert len(eng.unfilled_open_profiles()) == 0


def test_fill_profile_none_on_empty():
    eng = _sphere_engine()
    assert eng.fill_profile(None, "x") is None
    assert eng.fill_profile(pv.PolyData(), "x") is None
    assert eng.filled_patches == []


def test_load_stl_resets_filled_patches(tmp_path):
    eng = STLClipperEngine()
    p1 = tmp_path / "s1.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(p1))
    eng.load_stl(str(p1))
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    assert len(eng.filled_patches) == 1
    eng.load_stl(str(p1))
    assert eng.filled_patches == []


def test_export_combined_stl_includes_filled_patch(tmp_path):
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    out = tmp_path / "multi.stl"
    eng.export_combined_stl(str(out))
    text = out.read_text()
    assert "solid inlet" in text
    assert "solid wall" in text


def test_export_separate_stl_includes_filled_patch(tmp_path):
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))
    eng.fill_profile(eng.detect_open_profiles()[0], "inlet")
    sep_dir = tmp_path / "sep"
    eng.export_separate_stl(str(sep_dir))
    inlet_file = sep_dir / "inlet.stl"
    assert inlet_file.exists(), "inlet.stl not written by export_separate_stl"
    assert "solid inlet" in inlet_file.read_text()
    assert (sep_dir / "wall.stl").exists()


def test_auto_repair_returns_summary_and_replaces_mesh():
    eng = _sphere_engine()
    before_obj = eng.original_mesh
    msg = eng.auto_repair()
    assert msg.startswith("Auto-repair — ")
    assert eng.original_mesh is not None
    assert eng.original_mesh is not before_obj          # replaced with the repaired mesh


def test_auto_repair_no_mesh():
    assert STLClipperEngine().auto_repair() == "No mesh loaded."


def test_health_stats_keys():
    eng = _sphere_engine()
    stats = eng._health_stats()
    assert set(stats.keys()) == {"cells", "nm", "open", "pieces"}
    assert stats["pieces"] == 1
    assert stats["cells"] == eng.original_mesh.n_cells


def test_drawable_feature_curves_excludes_deleted_side():
    eng = _sphere_engine()
    assert eng.drawable_feature_curves() == []          # no curves yet
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    assert len(eng.drawable_feature_curves()) == 1      # interior on the closed surface
    top = _cell_on_side(eng.original_mesh, 2, True)
    eng.delete_cells(eng.flood_select(top))             # delete one side -> open profile
    assert eng.drawable_feature_curves() == []          # no longer interior -> not drawn


def test_drawable_feature_curves_excludes_floating():
    eng = _sphere_engine()
    eng.cut_by_plane((0, 0, 0), (0, 0, 1))
    z = eng.original_mesh.cell_centers().points[:, 2]
    band = [int(i) for i in np.where(np.abs(z) < 0.25)[0]]
    eng.delete_cells(band)                              # removes the seam -> curve floats
    assert eng.drawable_feature_curves() == []          # floating -> not drawn
