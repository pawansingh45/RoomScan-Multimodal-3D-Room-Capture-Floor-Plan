"""Whole-property assembly of independently reconstructed rooms (photo-tier requirement; also usable for any tier).

Each room is a rectilinear polygon in its OWN frame (axis-aligned, so only 90-degree rotations are ambiguous) plus its doors
(centre, outward normal, width). We search over door pairings with beam search:
    place room B so that its door b meets door a of an already placed room A (normals opposed, centres aligned, one wall
    thickness apart), reject placements that overlap anything, score = sum of width agreements of ALL coinciding door pairs
    (+ optional visual-adjacency prior from cross-folder image matches).
The top-K layouts give a posterior-like spread, used for the footprint interval instead of a fake single number.
"""
import itertools, math
import numpy as np
import cv2


def rot90(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s], [s, c]], float)


def _raster(poly, res=0.05, origin=None, shape=None):
    pts = np.round((np.asarray(poly) - origin) / res).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [pts[:, ::-1].copy()], 1)          # cv2: (x=col, y=row)
    return m.astype(bool)


class Room:
    def __init__(self, rid, polygon, doors):
        """doors: list of dict(center=[x,y], normal=[nx,ny] (outward, axis-aligned), width=w, kind='door'|'opening')"""
        self.id, self.poly = rid, np.asarray(polygon, float)
        self.doors = [dict(center=np.asarray(d["center"], float), normal=np.asarray(d["normal"], float), width=float(d["width"]), kind=d.get("kind", "door")) for d in doors]
        self.area = 0.5 * abs(np.dot(self.poly[:, 0], np.roll(self.poly[:, 1], -1)) - np.dot(self.poly[:, 1], np.roll(self.poly[:, 0], -1)))


def _transform(room, k, tau):
    R = rot90(k)
    poly = room.poly @ R.T + tau
    doors = [dict(center=R @ d["center"] + tau, normal=R @ d["normal"], width=d["width"], kind=d["kind"]) for d in room.doors]
    return poly, doors


def _overlap(polyA, polyB, res=0.05, tol_m2=0.06):
    amn, amx, bmn, bmx = polyA.min(0), polyA.max(0), polyB.min(0), polyB.max(0)
    if (amx < bmn + 0.1).any() or (bmx < amn + 0.1).any():       # bounding boxes cannot intersect: skip rasterising
        return False
    both = np.vstack([polyA, polyB]); o = both.min(0) - 0.2
    sh = tuple(np.ceil((both.max(0) + 0.2 - o) / res).astype(int)[::-1])
    a, b = _raster(polyA, res, o, sh), _raster(polyB, res, o, sh)
    from scipy import ndimage as ndi
    a, b = ndi.binary_erosion(a), ndi.binary_erosion(b)
    return float((a & b).sum() * res * res) > tol_m2


def _score_layout(placed, thickness, w_tol=0.15, prior=None):
    """Sum of agreement over every pair of doors that coincide geometrically (not just the ones we used to place)."""
    s, pairs = 0.0, []
    ids = list(placed)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            for da in placed[ids[i]][1]:
                for db in placed[ids[j]][1]:
                    if np.dot(da["normal"], db["normal"]) < -0.9 and abs(da["width"] - db["width"]) < w_tol:
                        d = db["center"] - da["center"]
                        along = np.dot(d, da["normal"]); across = np.linalg.norm(d - along * da["normal"])
                        if abs(along - thickness) < 0.12 and across < 0.12:
                            s += 1.0 - abs(da["width"] - db["width"]) / w_tol; pairs.append((ids[i], ids[j]))
    if prior:
        for (a, b), w in prior.items():
            if (a, b) in pairs or (b, a) in pairs:
                s += w
    return s, pairs


def _walls(poly):
    n = len(poly); out = []
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]; d = b - a; L = np.linalg.norm(d)
        if L < 0.5:
            continue
        t = d / L
        out.append(dict(a=a, b=b, c=0.5 * (a + b), n=np.array([t[1], -t[0]]), L=L, t=t))     # CCW polygon: outward = right of travel
    return out


def _contact_candidates(B, placed, thickness):
    """Weaker evidence: abut a wall of B against an anti-parallel wall of a placed room (3 sliding offsets)."""
    for aid, (_, _, polyA, _) in placed.items():
        for wa in _walls(polyA):
            for k in range(4):
                R = rot90(k)
                for wb in _walls(B.poly):
                    nb = R @ wb["n"]
                    if np.dot(nb, wa["n"]) > -0.99:
                        continue
                    tb = R @ wb["t"]
                    for sh in (0.0, 0.5 * (wa["L"] - wb["L"]), -0.5 * (wa["L"] - wb["L"])):
                        cb = R @ wb["c"]
                        tau = wa["c"] + wa["n"] * thickness + wa["t"] * sh - cb
                        # overlap of the two wall segments along the shared tangent
                        pa = np.array([np.dot(wa["a"], wa["t"]), np.dot(wa["b"], wa["t"])]); pb_ = np.array([np.dot(R @ wb["a"] + tau, wa["t"]), np.dot(R @ wb["b"] + tau, wa["t"])])
                        ov = min(pa.max(), pb_.max()) - max(pa.min(), pb_.min())
                        if ov > 0.5:
                            yield k, tau, ov


def assemble(rooms, thickness=0.15, beam=10, top_k=5, w_tol=0.15, prior=None, contact_fallback=True):
    """Returns list of layouts sorted by score: dict(score, placement{rid:(k,tau)}, pairs, unplaced, footprint_m2)."""
    rooms = sorted(rooms, key=lambda r: -r.area)
    by_id = {r.id: r for r in rooms}
    first = rooms[0]
    states = [dict(placed={first.id: (0, np.zeros(2), first.poly.copy(), first.doors)}, tf={first.id: (0, np.zeros(2))})]
    for _ in range(len(rooms) - 1):
        nxt = []
        for st in states:
            placed = st["placed"]
            cand = []
            for B in rooms:
                if B.id in placed:
                    continue
                for aid, (_, _, polyA, doorsA) in placed.items():
                    for da in doorsA:
                        for db in B.doors:
                            if abs(da["width"] - db["width"]) > w_tol:
                                continue
                            for k in range(4):
                                if np.dot(rot90(k) @ db["normal"], da["normal"]) > -0.9:
                                    continue
                                tau = da["center"] + da["normal"] * thickness - rot90(k) @ db["center"]
                                polyB, doorsB = _transform(B, k, tau)
                                if any(_overlap(polyB, p[2]) for p in placed.values()):
                                    continue
                                newp = dict(placed); newp[B.id] = (k, tau, polyB, doorsB)
                                sc, _ = _score_layout({i: (None, v[3]) for i, v in newp.items()}, thickness, w_tol, prior)
                                cand.append((sc, B.id, newp))
            if not cand and contact_fallback:
                for B in rooms:
                    if B.id in placed:
                        continue
                    for k, tau, ov in _contact_candidates(B, placed, thickness):
                        polyB, doorsB = _transform(B, k, tau)
                        if any(_overlap(polyB, p[2]) for p in placed.values()):
                            continue
                        newp = dict(placed); newp[B.id] = (k, tau, polyB, doorsB)
                        sc, _ = _score_layout({i: (None, v[3]) for i, v in newp.items()}, thickness, w_tol, prior)
                        cand.append((sc + 0.25 * min(ov, 2.0) / 2.0 - 0.01, B.id, newp))
            if not cand:
                nxt.append(st); continue
            for sc, bid, newp in cand:
                nxt.append(dict(placed=newp, tf={i: (v[0], v[1]) for i, v in newp.items()}))
        # dedupe by (placed ids, scores) and keep top beam
        scored = []
        seen = set()
        for st in nxt:
            sc, _ = _score_layout({i: (None, v[3]) for i, v in st["placed"].items()}, thickness, w_tol, prior)
            key = tuple(sorted((i, v[0], round(float(v[1][0]), 2), round(float(v[1][1]), 2)) for i, v in st["placed"].items()))
            if key in seen:
                continue
            seen.add(key); scored.append((len(st["placed"]), sc, st))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        states = [s[2] for s in scored[:beam]]
    out = []
    for st in states[:top_k]:
        sc, pairs = _score_layout({i: (None, v[3]) for i, v in st["placed"].items()}, thickness, w_tol, prior)
        allp = np.vstack([v[2] for v in st["placed"].values()]); o = allp.min(0) - 0.2
        sh = tuple(np.ceil((allp.max(0) + 0.2 - o) / 0.05).astype(int)[::-1])
        u = np.zeros(sh, bool)
        for v in st["placed"].values():
            u |= _raster(v[2], 0.05, o, sh)
        out.append(dict(score=sc, placement={i: (v[0], v[1].tolist()) for i, v in st["placed"].items()}, pairs=pairs,
                        unplaced=[r.id for r in rooms if r.id not in st["placed"]], footprint_m2=float(sum(r.area for r in rooms)), bbox_placed_m2=float(u.sum() * 0.0025),
                        polys={i: v[2] for i, v in st["placed"].items()}))
    return out


def footprint_interval(layouts, level=0.9, rel_floor_door=0.03, rel_floor_contact=0.09):
    """Footprint interval from the ALTERNATIVE layouts (unweighted range; softmax weights collapse onto the top-1 and
    under-cover), floored by how weak the evidence was: contact-only placements are ambiguous, so the floor is wider."""
    if not layouts:
        return None
    f = np.array([l["footprint_m2"] for l in layouts]); top = layouts[0]
    n_rooms = len(top["placement"])
    door_pairs = len({tuple(sorted(p)) for p in top["pairs"]})
    contact_only = (door_pairs < n_rooms - 1) or bool(top["unplaced"])
    m = float(f[0])
    # footprint = sum of room areas: independent of arrangement, so layout ambiguity does not widen it; only per-room area error does
    half = rel_floor_door * m
    return dict(value=m, lo=m - half, hi=m + half, n_layouts=len(layouts), top_score=float(top["score"]), evidence="contact_only" if contact_only else "door_confirmed",
                unplaced=top["unplaced"], level=level)


def rooms_from_plan(plan, rng=None, shuffle=True):
    """Shred a finished plan into independent rooms (own frames, random 90-deg rotation + offset) -> (rooms, truth)."""
    rng = rng or np.random.default_rng(0)
    rooms, truth = [], {}
    for r in plan["rooms"]:
        poly = np.array(r["polygon"]); c0 = poly.mean(0)
        k = int(rng.integers(0, 4)) if shuffle else 0; off = rng.uniform(-5, 5, 2) if shuffle else np.zeros(2)
        R = rot90(k)
        doors = []
        for o in r["openings"]:
            if o["kind"] not in ("door", "opening"):
                continue
            w = next(w for w in r["walls"] if w["id"] == o["wall_id"])
            c = 0.5 * (np.array(o["start"]) + np.array(o["end"]))
            doors.append(dict(center=R @ (c - c0) + off, normal=R @ np.array(w["outward"], float), width=o["width"]["value"], kind=o["kind"]))
        rooms.append(Room(r["id"], (poly - c0) @ R.T + off, doors)); truth[r["id"]] = c0
    return rooms, truth


def placement_error(layout, truth):
    """RMS centroid error after best rigid (rotation+translation, no scale, proper rotations only) alignment to the truth."""
    ids = [i for i in layout["placement"] if i in truth]
    if len(ids) < 2:
        return None
    P = np.array([layout["polys"][i].mean(0) for i in ids]); Q = np.array([truth[i] for i in ids])
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq)); S = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return float(np.sqrt(np.mean(np.sum(((P - mp) @ R.T + mq - Q) ** 2, 1))))
