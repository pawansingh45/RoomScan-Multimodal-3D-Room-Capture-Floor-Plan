"""Small, dependency-light geometry helpers (NumPy/SciPy only)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def quat_to_R(qx, qy, qz, qw):
    return Rot.from_quat([qx, qy, qz, qw]).as_matrix()


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_y_batch(a):
    c, s = np.cos(a), np.sin(a)
    Rm = np.zeros(a.shape + (3, 3))
    Rm[..., 0, 0] = c; Rm[..., 0, 2] = s; Rm[..., 1, 1] = 1; Rm[..., 2, 0] = -s; Rm[..., 2, 2] = c
    return Rm


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_quarter(a):
    """Wrap an angle to (-pi/4, pi/4]: Manhattan (90-degree) periodicity."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def voxel_downsample(P, voxel, extra=None):
    """Keep one averaged point per voxel. P: (N,3). Returns (M,3) [and averaged extra (M,k)]."""
    if len(P) == 0:
        return (P, extra) if extra is not None else P
    q = np.floor(P / voxel).astype(np.int64)
    q -= q.min(0)
    dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out = np.zeros((len(uniq), 3))
    for d in range(3):
        out[:, d] = np.bincount(inv, weights=P[:, d]) / cnt
    if extra is not None:
        ex = np.zeros((len(uniq), extra.shape[1]))
        for d in range(extra.shape[1]):
            ex[:, d] = np.bincount(inv, weights=extra[:, d]) / cnt
        return out, ex
    return out


def fit_plane_lsq(P):
    """Total-least-squares plane through points. Returns (n, d) with n.x + d = 0, and RMS residual."""
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    d = -n @ c
    rms = s[-1] / np.sqrt(len(P))
    return n, d, rms


def ransac_plane(P, thresh=0.02, iters=300, rng=None, min_inliers=50):
    rng = rng or np.random.default_rng(0)
    n = len(P)
    if n < 3:
        return None
    best, best_in = None, 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = P[idx]
        nv = np.cross(b - a, c - a)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv /= nn
        d = -nv @ a
        cnt = int((np.abs(P @ nv + d) < thresh).sum())
        if cnt > best_in:
            best_in, best = cnt, (nv, d)
    if best is None or best_in < min_inliers:
        return None
    nv, d = best
    inl = np.abs(P @ nv + d) < thresh
    nv, d, rms = fit_plane_lsq(P[inl])
    return nv, d, inl, rms


def robust_mode_1d(x, bin_w=0.01, refine_w=0.03, weights=None):
    """Mode of a 1-D distribution with sub-bin refinement. Returns (mode, n_in_peak)."""
    if len(x) == 0:
        return np.nan, 0
    lo, hi = x.min(), x.max()
    if hi - lo < bin_w:
        return float(np.median(x)), len(x)
    edges = np.arange(lo, hi + bin_w, bin_w)
    h, e = np.histogram(x, bins=edges, weights=weights)
    # smooth with 3-bin box to avoid splitting a peak over two bins
    hs = np.convolve(h, np.ones(3), mode="same")
    k = int(np.argmax(hs))
    c = 0.5 * (e[k] + e[k + 1])
    m = np.abs(x - c) < refine_w
    if m.sum() == 0:
        return float(c), 0
    return float(np.median(x[m])), int(m.sum())


def line_from_points_2d(P):
    """TLS line fit in 2-D. Returns (point_on_line, direction_unit, normal_unit, rms)."""
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return c, d, n, s[-1] / np.sqrt(len(P))


def sim3_align(src, dst, with_scale=True):
    """Umeyama similarity alignment src->dst. Returns s,R,t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
