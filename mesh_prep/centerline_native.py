"""
Native centerline computation — pure Python (numpy + scipy + VTK), no vmtk.

Reimplements the Voronoi-diagram / maximal-inscribed-sphere centerline method
(Antiga & Steinman, "Robust and objective decomposition and mapping of
bifurcating vessels", IEEE TMI 2004 — the algorithm behind vmtkCenterlines /
vmtkBranchExtractor) using only pip-installable, already-compiled building
blocks, so centerlines work in the pip / Apple-Silicon install where vmtk has
no build.

Pipeline
--------
1. Delaunay tetrahedralization of the surface points (scipy/Qhull).
2. Voronoi vertices = tet circumcenters; the circumradius at each vertex is the
   maximal inscribed sphere radius (MISR) — the local vessel radius.
3. Keep Voronoi vertices strictly inside the (closed) surface.
4. Graph over interior vertices (Delaunay-adjacent tets); edge cost =
   length / mean_radius, so shortest paths hug the fattest interior corridor
   (as central as possible) — the classic non-Eikonal equivalent vmtk uses.
5. Single super-source Dijkstra from the inlet seed(s); the predecessor array
   is one shortest-path tree. Reconstruct the path to each outlet seed.
6. Branch decomposition: the union of source->target paths is a tree; split it
   at bifurcation nodes (>= 2 children) into individual branches.

Output: ``pv.PolyData`` of branch polylines with point arrays
``MaximumInscribedSphereRadius`` (float) and ``BranchId`` (int). Signature
matches ``mesh_prep.centerline.compute_centerlines`` so it is a drop-in.
"""

import logging
from collections import defaultdict

import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

logger = logging.getLogger(__name__)

RADIUS_ARRAY = "MaximumInscribedSphereRadius"


def _as_points(flat):
    """[x,y,z,x,y,z,...] or (N,3) -> (N,3) float array."""
    a = np.asarray(flat, dtype=float)
    if a.ndim == 1:
        if a.size % 3 != 0:
            raise ValueError("seed point list length must be a multiple of 3")
        a = a.reshape(-1, 3)
    return a


def _tet_circumcenters(points, tets):
    """Circumcenter and circumradius of each tetrahedron (vectorized).

    Returns (centers (M,3), radii (M,), valid (M,) bool). Degenerate (near-flat)
    tets yield an invalid entry rather than raising."""
    a = points[tets[:, 0]]
    b = points[tets[:, 1]]
    c = points[tets[:, 2]]
    d = points[tets[:, 3]]
    # Rows 2*(b-a), 2*(c-a), 2*(d-a); rhs |b|^2-|a|^2, etc.
    M = np.stack([2 * (b - a), 2 * (c - a), 2 * (d - a)], axis=1)     # (M,3,3)
    rhs = np.stack([(b * b).sum(1) - (a * a).sum(1),
                    (c * c).sum(1) - (a * a).sum(1),
                    (d * d).sum(1) - (a * a).sum(1)], axis=1)         # (M,3)
    det = np.linalg.det(M)
    valid = np.abs(det) > 1e-12
    centers = np.full((len(tets), 3), np.nan)
    if valid.any():
        centers[valid] = np.linalg.solve(M[valid], rhs[valid])
    radii = np.linalg.norm(centers - a, axis=1)
    valid &= np.isfinite(centers).all(axis=1) & np.isfinite(radii)
    return centers, radii, valid


def _inside_surface(centers, surface):
    """Boolean mask: which points lie inside the closed surface."""
    try:
        surf = surface.extract_surface().triangulate()
        sel = pv.PolyData(centers).select_enclosed_points(
            surf, tolerance=0.0, check_surface=False)
        return np.asarray(sel["SelectedPoints"]).astype(bool)
    except Exception as e:                       # pragma: no cover - VTK edge cases
        logger.warning("select_enclosed_points failed (%s); keeping all", e)
        return np.ones(len(centers), dtype=bool)


def _build_graph(centers, radii, tets, neighbors, interior):
    """Sparse undirected graph over interior tets. Edge cost =
    euclidean length / mean radius (favours large-radius/central corridors)."""
    n = len(tets)
    rows, cols, data = [], [], []
    r_safe = np.maximum(radii, 1e-9)
    for i in range(n):
        if not interior[i]:
            continue
        for j in neighbors[i]:
            if j < 0 or j <= i or not interior[j]:
                continue
            length = float(np.linalg.norm(centers[i] - centers[j]))
            w = length / (0.5 * (r_safe[i] + r_safe[j]))
            rows.append(i); cols.append(j); data.append(w)
            rows.append(j); cols.append(i); data.append(w)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _snap(seed_points, centers, interior):
    """Nearest interior Voronoi vertex (tet id) for each seed point."""
    ids = np.where(interior)[0]
    if ids.size == 0:
        raise RuntimeError("no interior Voronoi vertices — is the surface closed?")
    tree = cKDTree(centers[ids])
    _, local = tree.query(_as_points(seed_points))
    return ids[np.atleast_1d(local)]


def _decompose_branches(paths):
    """Split a set of source->target node paths (a tree) into branches.

    Returns (branches, bifurcations): branches is a list of node-id lists, each
    a maximal chain between boundary nodes (root / bifurcation / leaf);
    bifurcations is the set of nodes with >= 2 children."""
    children = defaultdict(list)
    nodes = set()
    for p in paths:
        for u, v in zip(p[:-1], p[1:]):
            if v not in children[u]:
                children[u].append(v)
            nodes.add(u); nodes.add(v)
    roots = sorted({p[0] for p in paths if p})
    bifs = {u for u in nodes if len(children[u]) >= 2}

    def walk(u, first):
        seg = [u, first]
        cur = first
        while cur not in bifs and len(children[cur]) == 1:
            cur = children[cur][0]
            seg.append(cur)
        return seg

    branches = []
    for u in list(roots) + sorted(bifs):
        for c in children[u]:
            branches.append(walk(u, c))
    return branches, bifs


def compute_centerlines(surface_vtk, source_points, target_points,
                        max_points=25000):
    """Native centerline(s) from surface, inlet seed(s), outlet seed(s).

    Parameters
    ----------
    surface_vtk : pyvista.PolyData
        A CLOSED surface (wall + caps). Openings must be capped — as they are
        after Fill/clip in the app — or the interior test degenerates.
    source_points, target_points : flat [x,y,z,...] lists (or (N,3) arrays).
    max_points : int
        Surface points above this are strided down to keep Delaunay tractable.

    Returns
    -------
    pv.PolyData
        Branch polylines with point arrays MaximumInscribedSphereRadius, BranchId.

    Raises
    ------
    ValueError  : bad/empty seeds.
    RuntimeError: no interior vertices, or no source reaches a target.
    """
    src = _as_points(source_points)
    tgt = _as_points(target_points)
    if len(src) == 0 or len(tgt) == 0:
        raise ValueError("Need at least one source and one target point.")

    surf = surface_vtk.extract_surface().triangulate()
    pts = np.asarray(surf.points, dtype=float)
    if len(pts) < 4:
        raise RuntimeError("Surface has too few points for tetrahedralization.")
    if len(pts) > max_points:                    # deterministic stride subsample
        stride = int(np.ceil(len(pts) / max_points))
        pts = pts[::stride]
        logger.info("Subsampled surface %d -> %d points for Delaunay", len(surf.points), len(pts))

    logger.info("Delaunay tetrahedralization of %d points ...", len(pts))
    tri = Delaunay(pts)
    tets = tri.simplices
    neighbors = tri.neighbors

    centers, radii, valid = _tet_circumcenters(pts, tets)
    interior = valid & _inside_surface(centers, surf)
    n_in = int(interior.sum())
    logger.info("Voronoi vertices: %d tets, %d interior", len(tets), n_in)
    if n_in == 0:
        raise RuntimeError("No interior Voronoi vertices — surface may be open.")

    graph = _build_graph(centers, radii, tets, neighbors, interior)

    src_nodes = _snap(src, centers, interior)
    tgt_nodes = _snap(tgt, centers, interior)

    # Super-source node (index n) connected to every source with ~0 cost, so a
    # single Dijkstra yields ONE shortest-path tree rooted at all sources.
    n = len(tets)
    super_idx = n
    g = graph.tolil()
    g.resize((n + 1, n + 1))
    for s in np.unique(src_nodes):
        g[super_idx, s] = 1e-9
        g[s, super_idx] = 1e-9
    g = g.tocsr()

    dist, pred = dijkstra(g, directed=False, indices=super_idx,
                          return_predecessors=True)

    paths = []
    for t in tgt_nodes:
        if not np.isfinite(dist[t]):
            logger.warning("Target node %d unreachable from any source", t)
            continue
        seq = []
        cur = int(t)
        guard = 0
        while cur != super_idx and cur >= 0 and guard <= n + 1:
            seq.append(cur)
            cur = int(pred[cur])
            guard += 1
        if cur != super_idx:
            continue
        seq.reverse()                            # source ... target
        if len(seq) >= 2:
            paths.append(seq)
    if not paths:
        raise RuntimeError("No centerline path connects the seeds — check that "
                           "inlet/outlet points are inside the vessel.")

    branches, bifs = _decompose_branches(paths)

    # Assemble the branch polylines into one PolyData.
    all_pts, rad, bid, lines = [], [], [], []
    offset = 0
    for k, seg in enumerate(branches):
        seg = np.asarray(seg, dtype=int)
        all_pts.append(centers[seg])
        rad.append(radii[seg])
        bid.append(np.full(len(seg), k, dtype=np.int64))
        lines.append(np.concatenate([[len(seg)], np.arange(offset, offset + len(seg))]))
        offset += len(seg)

    poly = pv.PolyData(np.vstack(all_pts), lines=np.concatenate(lines))
    poly[RADIUS_ARRAY] = np.concatenate(rad)
    poly["BranchId"] = np.concatenate(bid)
    logger.info("Native centerline: %d branches, %d bifurcations, %d points",
                len(branches), len(bifs), poly.n_points)
    return poly
