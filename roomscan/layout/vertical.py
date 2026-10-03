"""Per-room floor / ceiling heights from the fused cloud, with an explicit 'not observed' state.

A ceiling is accepted only if a *horizontal plane* above head height covers a meaningful fraction of the
room footprint. Otherwise we abstain (no number) and report only a lower bound from the highest wall return:
that is the honest behaviour for floor-only scans (confident garbage on thin input is penalised).
"""
import numpy as np
from ..geom import robust_mode_1d


def room_points(rs, room_mask):
    ij = rs.grid.ij(rs.q)
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < rs.grid.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < rs.grid.shape[1])
    sel = np.zeros(len(rs.q), bool)
    sel[inb] = room_mask[ij[inb, 0], ij[inb, 1]]
    return sel


def estimate_vertical(rs, room_mask, area_m2, cfg, vox=0.02):
    sel = room_points(rs, room_mask)
    h = rs.h[sel]
    out = dict(n_points=int(sel.sum()))
    # ---- floor ----
    fl = h[np.abs(h) < 0.15]
    if len(fl) < 200:
        out.update(floor_status="not_observed", floor_offset=0.0, floor_sigma=0.03)
        floor_off = 0.0
    else:
        floor_off, nin = robust_mode_1d(fl, 0.005, 0.02)
        sd = float(np.std(fl[np.abs(fl - floor_off) < 0.04]))
        out.update(floor_status="observed", floor_offset=float(floor_off), floor_sigma=sd, floor_n=int(nin))
    # ---- ceiling ----
    hi = h[(h > 1.9) & (h < 4.5)]
    cell = vox * vox
    if len(hi) >= 800:
        # candidate horizontal planes = 1-cm histogram peaks; take the one covering the most area
        edges = np.arange(hi.min(), hi.max() + 0.01, 0.01)
        cnt, e = np.histogram(hi, bins=edges)
        sm = np.convolve(cnt, np.ones(3), "same")
        k = int(sm.argmax())
        c0 = 0.5 * (e[k] + e[k + 1])
        m = np.abs(hi - c0) < 0.03
        n_in = int(m.sum())
        peak, _ = robust_mode_1d(hi[m], 0.005, 0.015)
        sd = float(np.std(hi[np.abs(hi - peak) < 0.03]))
        cover = n_in * cell / max(area_m2, 1e-6)
        if cover >= 0.15 and n_in >= 600 and sd < 0.02:
            out.update(ceiling_status="observed", ceiling_y=float(peak), ceiling_sigma=sd, ceiling_n=n_in,
                       ceiling_cover=float(cover))
            # secondary level (soffit / bulkhead / sloped part)
            m2 = np.abs(hi - peak) > 0.12
            if m2.sum() > 600:
                c2, n2 = robust_mode_1d(hi[m2], 0.01, 0.03)
                if n2 * cell / max(area_m2, 1e-6) > 0.10:
                    out["ceiling_secondary_y"] = float(c2)
        else:
            out.update(ceiling_status="not_observed", ceiling_cover=float(cover))
    else:
        out.update(ceiling_status="not_observed", ceiling_cover=0.0)
    if out.get("ceiling_status") != "observed":
        top = float(np.percentile(h[h < 4.5], 99)) if len(h) else np.nan
        out["wall_top_lower_bound"] = (top - floor_off) if np.isfinite(top) else None
    return out
