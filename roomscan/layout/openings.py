"""Door / window / passage detection per wall, from depth evidence on the fitted wall plane.

For each polygon edge we build a (u = along-wall) x (height) evidence raster:
  on-plane points      -> wall is present
  points beyond plane  -> we can see *through* the wall: an opening
  floor beyond plane   -> doorway (a wall has no floor on its far side)
A column is 'open' when the wall is absent over the mid-height band AND there is see-through evidence.
Unobserved columns are *unknown*, never openings: occlusion must not create phantom doors.
"""
import numpy as np
from scipy import ndimage as ndi


def edge_frame(poly, k, room_mask, grid):
    a, b = poly[k], poly[(k + 1) % len(poly)]
    e = b - a
    L = float(np.linalg.norm(e))
    horizontal = abs(e[0]) >= abs(e[1])
    ax_c, ax_t = (1, 0) if horizontal else (0, 1)
    t0, t1 = sorted([a[ax_t], b[ax_t]])
    c = 0.5 * (a[ax_c] + b[ax_c])
    # outward direction: probe which side is outside the room raster
    mid = 0.5 * (a + b)
    sgn = 1.0
    for s_ in (1.0, -1.0):
        p = mid.copy(); p[ax_c] += s_ * 0.12
        ij = grid.ij(p[None])[0]
        inside = (0 <= ij[0] < grid.shape[0] and 0 <= ij[1] < grid.shape[1] and room_mask[ij[0], ij[1]])
        if not inside:
            sgn = s_; break
    return dict(ax_c=ax_c, ax_t=ax_t, t0=float(t0), t1=float(t1), c=float(c), sgn=sgn, length=L,
                a=a, b=b, horizontal=horizontal)


def u_axis(nu, ures, pad):
    return np.arange(nu) * ures - pad


def _col_frac(mat, h_lo, h_hi, hb0, hres):
    i0, i1 = int((h_lo - hb0) / hres), int((h_hi - hb0) / hres)
    return mat[:, i0:i1].mean(1)


def _jamb_peaks(rs, fr, pad=0.1):
    """Histogram of near-side depth-discontinuity points close to the wall plane, along the wall (1 cm bins)."""
    if rs.edge_q is None or len(rs.edge_q) == 0:
        return np.array([]), np.array([]), np.array([])
    u = rs.edge_q[:, fr["ax_t"]] - fr["t0"]
    d = (rs.edge_q[:, fr["ax_c"]] - fr["c"]) * fr["sgn"]
    m = (u > -pad) & (u < fr["length"] + pad) & (np.abs(d) < 0.15) & (rs.edge_h > 0.35) & (rs.edge_h < 1.9)
    return u[m], d[m], rs.edge_h[m]


def _snap_jamb(u_edge_pts, coarse, search=0.10, min_pts=12, bin_w=0.01):
    """Position of the jamb-edge cluster nearest to a coarse boundary (median of the points in the cluster)."""
    sel = np.abs(u_edge_pts - coarse) < search
    if sel.sum() < min_pts:
        return None, int(sel.sum())
    x = u_edge_pts[sel]
    edges = np.arange(x.min() - bin_w, x.max() + 2 * bin_w, bin_w)
    h, e = np.histogram(x, bins=edges)
    hs = np.convolve(h, np.ones(3), "same")
    k = int(hs.argmax()); c = 0.5 * (e[k] + e[k + 1])
    mm = np.abs(x - c) < 0.015
    if mm.sum() < min_pts:
        return None, int(mm.sum())
    return float(np.median(x[mm])), int(mm.sum())


def detect_openings_on_edge(rs, fr, ceil_h, cfg, ures=0.01, hres=0.05):
    q, h = rs.q, rs.h
    ax_c, ax_t = fr["ax_c"], fr["ax_t"]
    pad = 0.25
    u = q[:, ax_t] - fr["t0"]
    d = (q[:, ax_c] - fr["c"]) * fr["sgn"]          # >0 beyond the wall (outside the room)
    near = (u > -pad) & (u < fr["length"] + pad) & (d > -0.12) & (d < 1.2) & (h > 0.0) & (h < 2.6)
    if near.sum() < 50:
        return dict(openings=[], support=0.0)
    uu, dd, hh = u[near], d[near], h[near]
    nu = int(np.ceil((fr["length"] + 2 * pad) / ures))
    hb0, hb1 = 0.0, 2.6
    nh = int((hb1 - hb0) / hres)
    ui = np.clip(((uu + pad) / ures).astype(int), 0, nu - 1)
    hi = np.clip(((hh - hb0) / hres).astype(int), 0, nh - 1)
    on = np.zeros((nu, nh), bool); beyond = np.zeros((nu, nh), bool); anyp = np.zeros((nu, nh), bool)
    # 'wall present' = ANY surface within +-10 cm of the plane (robust to stepped walls, casings, 4-8 cm face offsets);
    # 'see-through' = surfaces clearly beyond the wall thickness
    m_on = (dd > -0.10) & (dd < 0.10)
    m_by = dd > 0.18
    on[ui[m_on], hi[m_on]] = True
    beyond[ui[m_by], hi[m_by]] = True
    anyp[ui, hi] = True
    # floor beyond the plane (a doorway shows the floor continuing)
    fl_by = np.zeros(nu, bool)
    mf = (np.abs(hh) < 0.06 + hres) & (dd > 0.2)
    fl_by[ui[mf]] = True
    # smooth along u by a 5-cm window
    k = np.ones((5, 1)) / 5.0
    on_f = ndi.convolve(on.astype(float), k, mode="nearest") > 0.2
    by_f = ndi.convolve(beyond.astype(float), k, mode="nearest") > 0.2
    mid_wall = _col_frac(on_f.astype(float), 0.5, 1.7, hb0, hres)
    # unthresholded wall-evidence density per column (smoothed over 5 cm). Its 50 %-of-wall-level crossing is an
    # unbiased edge estimate for a step edge (binary thresholding would dilate the wall by ~2 columns per side)
    dens = ndi.uniform_filter1d(_col_frac(on.astype(float), 0.5, 1.7, hb0, hres), 5, mode="nearest")
    mid_by = _col_frac(by_f.astype(float), 0.5, 1.7, hb0, hres)
    low_wall = _col_frac(on_f.astype(float), 0.1, 0.45, hb0, hres)
    obs_cols = _col_frac(np.maximum(on_f, by_f).astype(float), 0.5, 1.7, hb0, hres) > 0.05
    fl_by_s = ndi.binary_dilation(fl_by, iterations=3)
    openc = (mid_wall < 0.30) & ((mid_by > 0.25) | fl_by_s)
    openc = ndi.binary_closing(openc, structure=np.ones(7)) & ((u_axis(nu, ures, pad) >= -0.02) & (u_axis(nu, ures, pad) <= fr["length"] + 0.02))
    lab, n = ndi.label(openc)
    res = []
    ua = u_axis(nu, ures, pad)
    on_edge = (ua >= 0) & (ua <= fr["length"])
    wall_cols = ndi.binary_closing(mid_wall >= 0.5, structure=np.ones(5))
    support = float(wall_cols[on_edge].mean()) if on_edge.any() else 0.0
    if support < 0.25:                    # most of this edge is not an observed wall (virtual / open-plan boundary)
        return dict(openings=[], support=support)
    flank_n = int(0.10 / ures)
    for i in range(1, n + 1):
        cols = np.where(lab == i)[0]
        w_raw = (cols[-1] - cols[0] + 1) * ures
        if w_raw < cfg.window_min_w:
            continue
        # sub-column edge placement: where mid_wall crosses 0.5 on either side
        rho_all = dens[on_edge & wall_cols]
        rho = float(np.median(rho_all)) if len(rho_all) else 0.5
        lvl = 0.5 * rho

        def edge_pos(c_in, direction):
            c = c_in
            while 0 < c < nu - 1 and dens[c] < lvl:
                c += direction
            if c_in == c:
                return float(c_in)
            a_, b_ = dens[c - direction], dens[c]
            # boundary lies between the two column centres; column c covers [c, c+1) so centres at c + 0.5
            frac = (lvl - a_) / max(b_ - a_, 1e-9)
            pos_c = (c - direction + 0.5) + direction * frac
            return float(pos_c)
        l_idx = edge_pos(cols[0], -1); r_idx = edge_pos(cols[-1], +1)
        ul, ur = l_idx * ures - pad, r_idx * ures - pad       # l_idx/r_idx are in column units (centre-based)
        ul, ur = max(ul, 0.0), min(ur, fr["length"])
        # refine both boundaries to depth-discontinuity (jamb) clusters when present: sub-cm, independent of smoothing
        eu, _, _ = _jamb_peaks(rs, fr)
        jl, nl = _snap_jamb(eu, ul); jr, nr = _snap_jamb(eu, ur)
        refined = (jl is not None) + (jr is not None)
        if jl is not None:
            ul = jl
        if jr is not None:
            ur = jr
        width = ur - ul
        if width < cfg.window_min_w:
            continue
        # flanking wall on both sides (the polygon corner counts as a flank): an opening is a *gap in a wall*
        il, ir = int((ul + pad) / ures), int((ur + pad) / ures)
        left_ok = ul < 0.06 or wall_cols[max(0, il - flank_n - 2):max(1, il - 2)].mean() >= 0.6
        right_ok = (fr["length"] - ur) < 0.06 or wall_cols[min(nu - 1, ir + 2):min(nu, ir + flank_n + 2)].mean() >= 0.6
        if not (left_ok and right_ok):
            continue
        if ul < 0.06 and (fr["length"] - ur) < 0.06:
            continue
        # vertical structure at the opening centre
        cc = slice(int((0.5 * (cols[0] + cols[-1])) - 0.15 / ures), int((0.5 * (cols[0] + cols[-1])) + 0.15 / ures) + 1)
        col_on = on_f[cc].mean(0)
        sill = 0.0
        for j in range(nh):
            if col_on[j] > 0.3:
                sill = j * hres + hres
            else:
                break
        head = None
        for j in range(int(1.4 / hres), nh - 1):
            if col_on[j] > 0.3 and col_on[j + 1] > 0.3:
                head = j * hres; break
        low = float(low_wall[cols].mean())
        kind = "door" if (low < 0.35 and sill < 0.25) else ("window" if sill >= 0.4 else "opening")
        through = float(max(mid_by[cols].mean(), fl_by_s[cols].mean()))
        if kind == "door" and width < cfg.door_min_w:
            continue                               # too narrow to be a door: jamb returns / clutter
        if width > cfg.door_max_w and kind == "door":
            kind = "opening"                       # wide: archway / open-plan connection
        res.append(dict(u0=float(ul), u1=float(ur), width=float(width), jamb_refined=int(refined), kind=kind, sill=float(sill),
                        head=None if head is None else float(head), through=through, low_wall=low,
                        t0=fr["t0"] + ul, t1=fr["t0"] + ur))
    return dict(openings=res, support=support)
