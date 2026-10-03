"""Drift correction: gravity-aligned 4-DoF pose graph (x, y, z, yaw per keyframe).

Constraints
  * odometry   : relative motion between consecutive keyframes (trusted locally, drifts globally)
  * manhattan  : each keyframe's local dominant wall direction should equal ONE global grid angle (mod 90 deg).
                 This is what kills rotational drift (measured: ~5 deg over 115 s on the floor_only sample).
  * floor      : the floor seen from each keyframe is at one global height (kills vertical drift)
  * loop closure: revisits detected by proximity, VERIFIED by 2-D ICP of wall-band submaps (overlap, RMS and
                 degeneracy checks) before they are allowed into the graph.
Solver: robust (Huber) sparse non-linear least squares (scipy).
"""
import logging
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

from ..geom import wrap_pi, wrap_quarter, voxel_downsample, rot_y

log = logging.getLogger("roomscan")


def R2(a):
    """2-D (x,z) rotation matching rot_y(a) acting on (x,z)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


# ----------------------------------------------------------------------------------- 2-D ICP
def icp_se2(A, B, max_iter=40, trim=0.8, d0=0.35, d1=0.07):
    """Find (theta, tau) with  R2(theta) @ b + tau ~= a. A,B: (N,2). Trimmed point-to-point ICP.
    Returns dict(theta, tau, rms, overlap, degenerate)."""
    tree = cKDTree(A)
    th, tau = 0.0, np.zeros(2)
    for it in range(max_iter):
        dmax = d0 + (d1 - d0) * min(1.0, it / (0.6 * max_iter))
        Bt = B @ R2(th).T + tau
        dist, j = tree.query(Bt, distance_upper_bound=dmax)
        ok = np.isfinite(dist)
        if ok.sum() < 30:
            return None
        sel = np.where(ok)[0]
        if len(sel) > 20:
            sel = sel[np.argsort(dist[sel])[: max(20, int(trim * len(sel)))]]
        a, b = A[j[sel]], B[sel]
        ma, mb = a.mean(0), b.mean(0)
        H = (b - mb).T @ (a - ma)
        # closed form for R2(th)=[[c,s],[-s,c]]: maximise sum a^T R2 b  =>  th = atan2(H10-H01, H00+H11), H=sum b a^T
        th_new = np.arctan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1])
        tau_new = ma - R2(th_new) @ mb
        dth = abs(wrap_pi(th_new - th)); dtau = np.linalg.norm(tau_new - tau)
        th, tau = th_new, tau_new
        if dth < 1e-5 and dtau < 1e-4:
            break
    Bt = B @ R2(th).T + tau
    dist, j = tree.query(Bt, distance_upper_bound=0.05)
    inl = np.isfinite(dist)
    overlap = inl.sum() / max(1, min(len(A), len(B)))
    rms = float(np.sqrt(np.mean(dist[inl] ** 2))) if inl.any() else 1.0
    # degeneracy: do the matched surfaces constrain translation in both axes?
    kq = min(8, len(A))
    _, nb = tree.query(A[j[inl]] if inl.any() else A[:1], k=kq)
    if inl.any():
        pts = A[nb]  # (m,k,2)
        c = pts - pts.mean(1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", c, c)
        w, v = np.linalg.eigh(cov)
        nrm = v[:, :, 0]               # smallest-variance direction = surface normal
        Imat = nrm.T @ nrm
        ev = np.linalg.eigvalsh(Imat)
        degenerate = bool(ev[0] / max(ev[1], 1e-9) < 0.08)
    else:
        degenerate = True
    return dict(theta=float(th), tau=tau, rms=rms, overlap=float(overlap), degenerate=degenerate)


# ----------------------------------------------------------------------------------- loop closure
def _submap(kfs, j, floor_y, cfg, half=4):
    lo, hi = max(0, j - half), min(len(kfs), j + half + 1)
    out = []
    for q in range(lo, hi):
        k = kfs[q]
        P = k.pts_cam.astype(np.float64) @ k.R0.T + k.p0
        m = (P[:, 1] > floor_y + cfg.wall_band_lo_m) & (P[:, 1] < floor_y + cfg.wall_band_hi_m)
        out.append(P[m][:, [0, 2]])
    P = np.vstack(out) if out else np.zeros((0, 2))
    return voxel_downsample(np.c_[P, np.zeros(len(P))], 0.03)[:, :2] if len(P) else P


def find_loop_closures(kfs, floor_y, cfg, max_candidates=120):
    n = len(kfs)
    t = np.array([k.t for k in kfs]); p = np.array([k.p0 for k in kfs])
    cands = []
    for i in range(n):
        d = np.linalg.norm(p[:, [0, 2]] - p[i, [0, 2]], axis=1)
        for k in np.where((d < cfg.lc_max_dist_m) & (t - t[i] > cfg.lc_min_dt_s))[0]:
            cands.append((d[k], i, k))
    cands.sort()
    # thin: at most one candidate per (i//6, k//6) cell
    seen, chosen = set(), []
    for d, i, k in cands:
        key = (i // 6, k // 6)
        if key in seen:
            continue
        seen.add(key); chosen.append((i, k))
        if len(chosen) >= max_candidates:
            break
    lcs = []
    for i, k in chosen:
        A, B = _submap(kfs, i, floor_y, cfg), _submap(kfs, k, floor_y, cfg)
        if len(A) < 400 or len(B) < 400:
            continue
        r = icp_se2(A, B)
        if r is None or r["degenerate"]:
            continue
        if r["rms"] > cfg.lc_icp_max_rms_m or r["overlap"] < cfg.lc_icp_min_overlap:
            continue
        if np.linalg.norm(r["tau"]) > 0.6 or abs(r["theta"]) > np.radians(10):
            continue
        lcs.append((i, k, r))
    log.info("loop closure: %d candidates -> %d verified", len(chosen), len(lcs))
    return lcs



def _huber_w(r, f=3.0):
    a = np.abs(r)
    return np.where(a <= f, 1.0, np.sqrt(f / np.maximum(a, 1e-12)))


def _lm_solve(resid, x0, S, iters=40, f=3.0):
    """Sparse Levenberg-Marquardt with Huber IRLS. Direct sparse solves: robust on long chain graphs where
    scipy's lsmr-based trust-region solver stalls."""
    from scipy.optimize._numdiff import approx_derivative
    from scipy.sparse import diags, identity
    from scipy.sparse.linalg import spsolve
    x = x0.copy()
    lam = 1e-3

    def cost_of(r):
        a = np.abs(r)
        return float(np.sum(np.where(a <= f, 0.5 * r * r, f * (a - 0.5 * f))))
    r = resid(x); c = cost_of(r)
    nit = 0
    for nit in range(1, iters + 1):
        J = approx_derivative(resid, x, method="2-point", rel_step=1e-7, sparsity=S, f0=r)
        w = _huber_w(r, f)
        Jw = diags(w) @ J; rw = w * r
        H = (Jw.T @ Jw).tocsc(); g = Jw.T @ rw
        improved = False
        for _ in range(8):
            A = H + lam * diags(np.maximum(H.diagonal(), 1e-9)) + 1e-12 * identity(H.shape[0])
            dx = spsolve(A.tocsc(), -g)
            xn = x + dx
            rn = resid(xn); cn = cost_of(rn)
            if cn < c:
                dec = c - cn
                x, r, c = xn, rn, cn
                lam = max(lam / 4, 1e-9); improved = True
                break
            lam *= 6
        if not improved or dec < 1e-6 * max(c, 1.0):
            break
    return x, c, nit


# ----------------------------------------------------------------------------------- the graph
def optimise(kfs, floor_y, cfg, mode="full"):
    """Returns dict(psi, pos, phi, info). mode: off | manhattan | full."""
    n = len(kfs)
    p0 = np.array([k.p0 for k in kfs]); psi0 = np.array([k.psi0 for k in kfs])
    if mode == "off" or n < 3:
        return dict(psi=psi0.copy(), pos=p0.copy(), phi=np.nan, info=dict(mode=mode, lc=0))
    th = np.array([k.man_theta for k in kfs]); Rc = np.array([k.man_R for k in kfs])
    ok = np.isfinite(th) & (Rc > 0.45)
    phi0 = float(np.angle(np.sum(Rc[ok] * np.exp(4j * th[ok]))) / 4) if ok.any() else 0.0
    fy = np.array([k.floor_y for k in kfs]); fn = np.array([k.floor_n for k in kfs])
    fok = np.isfinite(fy) & (fn > 300)
    F = float(np.median(fy[fok])) if fok.any() else floor_y
    lcs = find_loop_closures(kfs, floor_y, cfg) if mode == "full" else []

    # unknowns: [x y z psi]*n + [phi]
    def unpack(x):
        X = x[:4 * n].reshape(n, 4)
        return X[:, :3], X[:, 3], x[4 * n]

    rel0 = []
    for i in range(n - 1):
        d = p0[i + 1] - p0[i]
        a = rot_y(-psi0[i]) @ d
        rel0.append((a, wrap_pi(psi0[i + 1] - psi0[i]), np.linalg.norm(d)))
    sig_t = lambda L: 0.003 + 0.010 * L
    sig_a = lambda L: 0.0006 + 0.004 * L
    SIG_M, SIG_F = 0.035, 0.012

    lc_meas = []
    for i, k, r in lcs:
        Tp = R2(r["theta"]) @ p0[k][[0, 2]] + r["tau"]
        pk = np.array([Tp[0], p0[k][1], Tp[1]])
        a = rot_y(-psi0[i]) @ (pk - p0[i])
        lc_meas.append((i, k, a, wrap_pi(psi0[k] + r["theta"] - psi0[i])))

    def resid(x):
        pos, psi, phi = unpack(x)
        out = []
        for i in range(n - 1):
            a, b, L = rel0[i]
            e = rot_y(-psi[i]) @ (pos[i + 1] - pos[i]) - a
            out.append(e / sig_t(L))
            out.append([wrap_pi(psi[i + 1] - psi[i] - b) / sig_a(L)])
        for i in np.where(ok)[0]:
            out.append([wrap_quarter(th[i] - (psi[i] - psi0[i]) - phi) / (SIG_M / max(Rc[i], 0.3))])
        for i in np.where(fok)[0]:
            out.append([(fy[i] + (pos[i, 1] - p0[i, 1]) - F) / SIG_F])
        for (i, k, a, b) in lc_meas:
            e = rot_y(-psi[i]) @ (pos[k] - pos[i]) - a
            out.append(e / 0.02)
            out.append([wrap_pi(psi[k] - psi[i] - b) / 0.004])
        out.append((pos[0] - p0[0]) / 1e-3)
        out.append([wrap_pi(psi[0] - psi0[0]) / 1e-4])
        return np.concatenate([np.atleast_1d(np.asarray(o, float)) for o in out])

    # sparsity pattern
    rows = 0
    pat = []
    for i in range(n - 1):
        pat.append(([i, i + 1], 3)); pat.append(([i, i + 1], 1))
    for i in np.where(ok)[0]:
        pat.append(([i], 1, True))
    for i in np.where(fok)[0]:
        pat.append(([i], 1))
    for (i, k, a, b) in lc_meas:
        pat.append(([i, k], 3)); pat.append(([i, k], 1))
    pat.append(([0], 3)); pat.append(([0], 1))
    nr = sum(p[1] for p in pat)
    S = lil_matrix((nr, 4 * n + 1), dtype=int)
    r0 = 0
    for p in pat:
        idxs, cnt = p[0], p[1]
        for rr in range(cnt):
            for q in idxs:
                S[r0 + rr, 4 * q:4 * q + 4] = 1
            if len(p) > 2:
                S[r0 + rr, 4 * n] = 1
        r0 += cnt
    x0 = np.concatenate([np.c_[p0, psi0].ravel(), [phi0]])
    res0 = resid(x0)
    x, cost1, nit = _lm_solve(resid, x0, S)
    pos, psi, phi = unpack(x)
    sol = type("S", (), {"cost": cost1, "nfev": nit})
    # pin the frame: remove the global yaw/translation gauge freedom relative to keyframe 0
    info = dict(mode=mode, lc=len(lcs), cost0=float(0.5 * res0 @ res0), cost1=float(sol.cost),
                phi_deg=float(np.degrees(phi)), yaw_corr_deg_max=float(np.degrees(np.abs(psi - psi0).max())),
                pos_corr_m_max=float(np.linalg.norm(pos - p0, axis=1).max()), manhattan_obs=int(ok.sum()))
    log.info("posegraph[%s]: cost %.0f -> %.0f, max yaw corr %.2f deg, max pos corr %.3f m, LC=%d",
             mode, info["cost0"], info["cost1"], info["yaw_corr_deg_max"], info["pos_corr_m_max"], len(lcs))
    return dict(psi=psi, pos=pos, phi=float(phi), info=info, lcs=[(i, k) for i, k, _ in lcs])
