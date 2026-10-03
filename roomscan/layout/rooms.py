"""Rasters in the Manhattan frame -> room regions -> refined rectilinear polygons.

Pipeline
  1. rotate the fused cloud so walls are axis-aligned (global Manhattan angle phi from the pose graph)
  2. rasters: observed floor, wall evidence (count of occupied height slices: walls are tall, furniture is not)
  3. free-space = closed floor mask minus walls; rooms = watershed on its distance transform (doors are necks)
  4. polygon per room: contour -> Douglas-Peucker -> axis snapping -> every edge re-fitted to *wall points*
     (sub-centimetre placement of the wall face), vertices = intersections of refined edge lines
"""
import logging
from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import h_maxima, disk
from skimage.segmentation import watershed

from ..geom import robust_mode_1d

log = logging.getLogger("roomscan")


def to_manhattan(xz, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.c_[xz[:, 0] * c + xz[:, 1] * s, -xz[:, 0] * s + xz[:, 1] * c]


def from_manhattan(q, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.c_[q[:, 0] * c - q[:, 1] * s, q[:, 0] * s + q[:, 1] * c]


@dataclass
class Grid:
    mn: np.ndarray
    res: float
    shape: tuple

    def ij(self, q):
        return np.floor((q - self.mn) / self.res).astype(int)

    def xy(self, ij):
        return (np.asarray(ij, float) + 0.5) * self.res + self.mn


@dataclass
class RasterSet:
    grid: Grid
    floor: np.ndarray        # bool, observed floor
    wall_cnt: np.ndarray     # int, # occupied 10-cm slices in wall band
    wall: np.ndarray         # bool, tall-structure mask
    path: np.ndarray         # bool, camera path cells
    q: np.ndarray            # (N,2) cloud in Manhattan frame
    h: np.ndarray            # (N,) height above floor
    wall_pt: np.ndarray      # (N,) bool: point belongs to a wall cell inside the wall band
    edge_q: np.ndarray = None   # (E,2) jamb-candidate points (Manhattan frame)
    edge_h: np.ndarray = None   # (E,) their heights above the floor


def build_rasters(P, floor_y, phi, cam_xz, cfg, edge_pts=None):
    q = to_manhattan(P[:, [0, 2]], phi)
    h = P[:, 1] - floor_y
    res = cfg.raster_res_m
    mn = q.min(0) - 0.3
    shape = tuple(np.ceil((q.max(0) + 0.3 - mn) / res).astype(int))
    g = Grid(mn, res, shape)
    ij = g.ij(q)
    fm = np.abs(h) < cfg.floor_band_m
    floor = np.zeros(shape, bool); floor[ij[fm, 0], ij[fm, 1]] = True
    nsl = int(round((cfg.wall_band_hi_m - cfg.wall_band_lo_m) / 0.1))
    occ = np.zeros(shape + (nsl,), bool)
    for k in range(nsl):
        m = (h > cfg.wall_band_lo_m + 0.1 * k) & (h <= cfg.wall_band_lo_m + 0.1 * (k + 1))
        occ[ij[m, 0], ij[m, 1], k] = True
    cnt = occ.sum(2).astype(np.int32)
    thr = max(5, int(0.5 * nsl))
    wall = cnt >= thr
    wall = ndi.binary_opening(wall, structure=np.ones((1, 3))) | ndi.binary_opening(wall, structure=np.ones((3, 1)))  # keep line-like
    path = np.zeros(shape, bool)
    cij = g.ij(to_manhattan(cam_xz, phi))
    cij = np.clip(cij, 0, np.array(shape) - 1)
    path[cij[:, 0], cij[:, 1]] = True
    path = ndi.binary_dilation(path, iterations=int(0.25 / res))
    inb = (h > cfg.wall_band_lo_m) & (h < cfg.wall_band_hi_m)
    wall_pt = inb & wall[ij[:, 0], ij[:, 1]]
    eq = eh = None
    if edge_pts is not None and len(edge_pts):
        eq = to_manhattan(edge_pts[:, [0, 2]], phi); eh = edge_pts[:, 1] - floor_y
    return RasterSet(g, floor, cnt, wall, path, q, h, wall_pt, eq, eh)


def _runs_along(mask, axis, min_cells):
    """Pixels of `mask` that belong to straight runs (>= min_cells) along `axis` (0 or 1)."""
    arr = mask if axis == 1 else mask.T            # arr[row, along]
    st = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    lab, n = ndi.label(arr, structure=st)
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_cells; keep[0] = False
    out = keep[lab]
    return out if axis == 1 else out.T


def _close_along(mask, length_cells, axis):
    k = np.ones((1, length_cells), np.uint8) if axis == 1 else np.ones((length_cells, 1), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def close_doors(W, res, door_max_w, min_run_m=0.25):
    """Bridge gaps between COLLINEAR wall pieces only (never across a corridor between parallel walls)."""
    k = int(round(door_max_w / res))
    m = int(round(min_run_m / res))
    W0 = _runs_along(W, 0, m)     # runs along axis 0 = walls of constant 2nd coordinate
    W1 = _runs_along(W, 1, m)
    return _close_along(W0, k, 0) | _close_along(W1, k, 1)


def segment_rooms(rs, cfg):
    """Wall-driven room segmentation.

    1. wall mask W (tall structure). Gaps in W between *collinear* wall pieces up to door_max_w are closed with
       virtual walls (these are the door / passage candidates, returned for the opening stage).
    2. interior = every observed cell (floor, furniture, wall points, camera path) that is not wall.
    3. rooms = connected components of interior that contain real floor observations; exterior views through
       windows / doors form components with no floor and are dropped.
    4. watershed on the distance transform splits only components that are still 'leaky' (necks narrower than rooms).
    Returns labels, virtual_door_mask.
    """
    res = rs.grid.res
    W = rs.wall
    Wc = close_doors(W, res, cfg.door_max_w)
    virt = Wc & ~W & ~rs.floor
    Wall = W | virt
    Wall = ndi.binary_dilation(Wall, iterations=1)
    observed = rs.floor | rs.path | (rs.wall_cnt > 0)
    observed = ndi.binary_closing(observed, structure=disk(int(0.12 / res)))
    interior = observed & ~Wall
    interior = ndi.binary_opening(interior, structure=disk(int(0.08 / res)))
    lab, n = ndi.label(interior)
    keep = np.zeros_like(interior)
    for i in range(1, n + 1):
        m = lab == i
        area = m.sum() * res * res
        floor_frac = (rs.floor | rs.path)[m].mean()
        if area >= cfg.min_room_area_m2 and floor_frac > 0.15:
            keep |= m
    # fill unobserved-floor bites (furniture shadows, unscanned patches) up to ~1.0 m wide, never across walls
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.5 / res) + 1,) * 2)
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, kk).astype(bool) & ~Wall
    # split leaks (necks narrower than the rooms they join) without cutting genuine open-plan spaces
    dist = ndi.distance_transform_edt(keep) * res
    ds = ndi.gaussian_filter(dist, 1.0)
    hm = h_maxima(ds, 0.18)
    markers, _ = ndi.label(hm & keep)
    labels = watershed(-ds, markers, mask=keep)
    labels = _merge_small(labels, res, cfg.min_room_area_m2)
    return labels, virt, dist


def _merge_small(labels, res, min_area):
    changed = True
    while changed:
        changed = False
        ids = [i for i in np.unique(labels) if i != 0]
        areas = {i: (labels == i).sum() * res * res for i in ids}
        for i in sorted(ids, key=lambda k: areas[k]):
            if areas[i] >= min_area:
                continue
            m = labels == i
            ring = ndi.binary_dilation(m, iterations=3) & ~m
            nb = labels[ring]; nb = nb[nb > 0]
            if len(nb):
                tgt = np.bincount(nb).argmax()
                labels[m] = tgt
            else:
                labels[m] = 0
            changed = True
            break
    # relabel
    out = np.zeros_like(labels)
    for k, i in enumerate([i for i in np.unique(labels) if i != 0], 1):
        out[labels == i] = k
    return out


# ---------------------------------------------------------------------------------- polygons
def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def wall_line_coords(rs, min_run_m=0.6, nms_m=0.05):
    """Global wall-face coordinates from *continuous wall runs* (>= min_run_m) in the wall raster, refined to
    sub-cm with the wall points. Returns (xs, ys, strength_x, strength_y): xs = positions of vertical walls."""
    res = rs.grid.res
    out = []
    for ax in (0, 1):
        arr = rs.wall if ax == 0 else rs.wall.T            # arr[const_idx, along_idx]
        lab, n = ndi.label(arr, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
        if n == 0:
            out.append((np.array([]), np.array([]))); continue
        sizes = np.bincount(lab.ravel())[1:]
        rows = np.zeros(arr.shape[0])
        # row index of each label (labels never span rows with this structure)
        first = ndi.minimum_position(np.zeros_like(lab), lab, range(1, n + 1)) if False else None
        rr, cc = np.nonzero(lab)
        row_of = np.zeros(n + 1, int); row_of[lab[rr, cc]] = rr
        good = sizes * res >= min_run_m
        np.add.at(rows, row_of[1:][good], sizes[good] * res)
        prof = np.convolve(rows, np.ones(3), "same")
        coords, strength = [], []
        hh = prof.copy(); w = int(nms_m / res)
        v = rs.q[rs.wall_pt, ax]
        while True:
            k = int(hh.argmax())
            if hh[k] < min_run_m:
                break
            c0 = rs.grid.mn[ax] + (k + 0.5) * res
            m = np.abs(v - c0) < 0.03
            coords.append(float(np.median(v[m])) if m.sum() >= 15 else c0); strength.append(float(prof[k]))
            hh[max(0, k - w):k + w + 1] = 0
        o = np.argsort(coords)
        out.append((np.array(coords)[o], np.array(strength)[o]))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def _snap_to(vals, lines, tol):
    if len(lines) == 0:
        return vals
    d = np.abs(vals[:, None] - lines[None, :])
    k = d.argmin(1)
    ok = d[np.arange(len(vals)), k] <= tol
    out = vals.copy()
    out[ok] = lines[k[ok]]
    return out


def region_polygon(mask, grid, xs, ys, snap_tol=0.12, min_step=0.15, allowed=None):
    """Region mask -> clean rectilinear polygon (Manhattan metres)."""
    m = ndi.binary_dilation(mask, iterations=2)
    if allowed is not None:
        m &= allowed                              # never grow into another room (no stitched-plan overlaps)
    sq = np.ones((9, 9), np.uint8)                                   # remove notches/spurs < ~18 cm
    mu = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, sq)
    mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, sq)
    if allowed is not None:
        mu = (mu.astype(bool) & allowed).astype(np.uint8)
    if mu.sum() == 0:
        mu = m.astype(np.uint8)
    loop = trace_boundary(mu.astype(bool))
    if loop is None:
        return None
    q = grid.mn + loop.astype(float) * grid.res
    q[:, 0] = _snap_to(q[:, 0], xs, snap_tol)
    q[:, 1] = _snap_to(q[:, 1], ys, snap_tol)
    return _rectilinear_simplify(q, min_step)


def trace_boundary(M):
    """Exact outer boundary of the largest 4-connected component of a boolean mask, as a rectilinear
    polygon on the pixel-corner lattice (vertices only at corners). No diagonal edges by construction."""
    lab, n = ndi.label(M)
    if n == 0:
        return None
    if n > 1:
        M = lab == (1 + int(np.argmax(ndi.sum(M, lab, range(1, n + 1)))))
    M = M.astype(bool)
    P = np.pad(M, 1)
    I, J = np.nonzero(M)
    I1, J1 = I + 1, J + 1
    nxt = {}
    def add(mask, s0, s1, e0, e1):
        for a_, b_, c_, d_ in zip(s0[mask], s1[mask], e0[mask], e1[mask]):
            nxt.setdefault((a_, b_), []).append((c_, d_))
    top = ~P[I1 - 1, J1]; right = ~P[I1, J1 + 1]; bottom = ~P[I1 + 1, J1]; left = ~P[I1, J1 - 1]
    add(top, I, J, I, J + 1)
    add(right, I, J + 1, I + 1, J + 1)
    add(bottom, I + 1, J + 1, I + 1, J)
    add(left, I + 1, J, I, J)
    if not nxt:
        return None
    # chain loops; keep the longest
    best = []
    visited = set()
    for start in list(nxt.keys()):
        if start in visited:
            continue
        loop, cur = [start], start
        visited.add(start)
        while True:
            outs = nxt.get(cur)
            if not outs:
                break
            nx_ = outs.pop() if len(outs) > 1 else outs[0]
            if nx_ == start:
                break
            if nx_ in visited:
                break
            loop.append(nx_); visited.add(nx_); cur = nx_
        if len(loop) > len(best):
            best = loop
    pts = np.array(best, float)
    return _clean_poly(pts)


def _clean_poly(q):
    """Remove duplicate (keeping ONE of each pair), collinear and spike vertices until stable."""
    pts = []
    for p in np.asarray(q, float):
        if not pts or not np.allclose(p, pts[-1], atol=1e-9):
            pts.append(p)
    while len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=1e-9):
        pts.pop()
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        k = 0
        while k < len(pts) and len(pts) >= 3:
            a, b, c = pts[k - 1], pts[k], pts[(k + 1) % len(pts)]
            col = (abs(a[0] - b[0]) < 1e-9 and abs(b[0] - c[0]) < 1e-9) or (abs(a[1] - b[1]) < 1e-9 and abs(b[1] - c[1]) < 1e-9)
            if col or np.allclose(a, b, atol=1e-9) or np.allclose(b, c, atol=1e-9):
                pts.pop(k); changed = True
            else:
                k += 1
        # a spike leaves two identical neighbours: dedupe again
        out = []
        for p in pts:
            if not out or not np.allclose(p, out[-1], atol=1e-9):
                out.append(p)
        while len(out) > 1 and np.allclose(out[0], out[-1], atol=1e-9):
            out.pop()
        if len(out) != len(pts):
            changed = True
        pts = out
    return np.array(pts)


def _is_rectilinear(q, tol=1e-6):
    d = np.abs(np.roll(q, -1, 0) - q)
    return bool(((d[:, 0] < tol) | (d[:, 1] < tol)).all())


def _rectilinear_simplify(q, min_step=0.15):
    """Remove steps shorter than min_step: the short edge's two parallel neighbours are re-aligned onto the
    line of the *longer* neighbour (so real walls keep their position), then collinear vertices are merged."""
    q = _clean_poly(np.array(q, float))
    q_in = q.copy()
    for _ in range(500):
        n = len(q)
        if n <= 4:
            break
        e = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1)
        k = int(np.argmin(e))
        if e[k] >= min_step:
            break
        i0, i1, i2, im = k, (k + 1) % n, (k + 2) % n, (k - 1) % n
        ax = 0 if abs(q[i1][0] - q[i0][0]) > abs(q[i1][1] - q[i0][1]) else 1     # axis along which e_k varies
        l_prev = np.linalg.norm(q[i0] - q[im]); l_next = np.linalg.norm(q[i2] - q[i1])
        q = q.copy()
        if l_prev >= l_next:
            q[i2][ax] = q[i0][ax]
        else:
            q[im][ax] = q[i1][ax]
        keep = [j for j in range(n) if j not in (i0, i1)]
        q = _clean_poly(q[keep])
    return q if (_is_rectilinear(q) and len(q) >= 4) else q_in


def refine_edges(poly, rs, cfg, max_shift=0.12):
    """Re-fit each axis-aligned edge to the actual wall points. Returns (poly, edge_info list)."""
    n = len(poly)
    lines = []     # (orient, coord, observed, n_pts, spread)
    info = []
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        e = b - a
        L = np.linalg.norm(e)
        horizontal = abs(e[0]) >= abs(e[1])
        ax_c = 1 if horizontal else 0          # coord that is constant
        ax_t = 0 if horizontal else 1
        c0 = 0.5 * (a[ax_c] + b[ax_c])
        t0, t1 = sorted([a[ax_t], b[ax_t]])
        t0 += 0.06; t1 -= 0.06
        sel = rs.wall_pt & (np.abs(rs.q[:, ax_c] - c0) < max_shift) & (rs.q[:, ax_t] > t0) & (rs.q[:, ax_t] < t1)
        pts = rs.q[sel, ax_c]
        obs, spread = False, np.nan
        coord = c0
        if len(pts) >= 40:
            # the wall *face* nearest the room interior: take the mode of the points; thick clutter shifts it little
            coord, nin = robust_mode_1d(pts, 0.005, 0.02)
            spread = float(np.std(pts[np.abs(pts - coord) < 0.04])) if nin else np.nan
            obs = nin >= 25
            if not obs:
                coord = c0
        lines.append(("h" if horizontal else "v", coord, obs, int(len(pts)), spread))
    # recompute vertices by intersecting consecutive refined lines
    V = []
    for k in range(n):
        a, b = lines[k - 1], lines[k]
        if a[0] == "h" and b[0] == "v":
            V.append((b[1], a[1]))
        elif a[0] == "v" and b[0] == "h":
            V.append((a[1], b[1]))
        else:
            V.append(tuple(poly[k]))
    newp = np.array(V, float)
    return newp, lines


def extract_rooms(rs, labels, cfg):
    rooms = []
    xs, ys, _, _ = wall_line_coords(rs)
    for lid in [i for i in np.unique(labels) if i != 0]:
        mask = labels == lid
        poly = region_polygon(mask, rs.grid, xs, ys, allowed=(labels == 0) | (labels == lid))
        if poly is None or len(poly) < 3:
            continue
        refined, lines = refine_edges(poly, rs, cfg)
        rooms.append(dict(label=int(lid), poly_q=refined, poly_q0=poly, lines=lines, mask=mask))
    return rooms
