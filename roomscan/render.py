"""Render the stitched plan (PNG + SVG): rooms, walls (observed solid / inferred dashed), openings, dimensions."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Arc


def _fmt(iv):
    if iv is None or iv.get("value") is None:
        return "n/a"
    return f"{iv['value']:.2f}"


def render_plan(plan, path_png, path_svg=None, title=None):
    rooms = plan["rooms"]
    fig, ax = plt.subplots(figsize=(11, 9))
    cols = plt.cm.Pastel1(np.arange(9))
    allp = []
    for ri, r in enumerate(rooms):
        P = np.array(r["polygon"]); allp.append(P)
        ax.add_patch(MPoly(P, closed=True, fc=cols[ri % 9], ec="none", alpha=0.55 if r["kind"] == "room" else 0.3))
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            ax.plot([a[0], b[0]], [a[1], b[1]], "-" if w["observed"] else "--", c="k" if w["observed"] else "0.5", lw=2.2 if w["observed"] else 1.2, zorder=3)
        for o in r["openings"]:
            a, b = np.array(o["start"]), np.array(o["end"])
            c = {"door": "#d62728", "window": "#17becf", "opening": "#9467bd"}[o["kind"]]
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", c="white", lw=6, zorder=4)
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", c=c, lw=2, zorder=5)
            m = 0.5 * (a + b)
            ax.text(m[0], m[1], f" {o['width']['value']*100:.0f}", fontsize=7, color=c, zorder=6)
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            if np.linalg.norm(b - a) > 0.7:
                m = 0.5 * (a + b); d = (b - a) / np.linalg.norm(b - a); nrm = np.array([-d[1], d[0]]) * 0.12
                ax.text(*(m + nrm), f"{w['length']['value']:.2f}", fontsize=6.5, ha="center", va="center", color="#333", zorder=6)
        c = np.array(r["centroid"])
        ch = r["ceiling_height"]
        ctext = f"h {ch['value']:.2f} m" if ch.get("value") else "ceiling n/a"
        ax.text(c[0], c[1], f"{r['name']}\n{r['floor_area']['value']:.1f} m²\n{ctext}", ha="center", va="center", fontsize=8, weight="bold", zorder=7)
    if allp:
        A = np.vstack(allp); pad = 0.5
        ax.set_xlim(A[:, 0].min() - pad, A[:, 0].max() + pad); ax.set_ylim(A[:, 1].min() - pad, A[:, 1].max() + pad)
        x0, y0 = A[:, 0].min(), A[:, 1].min() - 0.35
        ax.plot([x0, x0 + 1], [y0, y0], "k-", lw=3); ax.text(x0 + 0.5, y0 - 0.15, "1 m", ha="center", fontsize=8)
    ax.set_aspect("equal"); ax.axis("off")
    sp = plan.get("stitched_plan", {})
    fa = sp.get("footprint_area", {})
    ttl = title or "Floor plan"
    sub = f"tier: {plan['capture']['tier']} | rooms {sp.get('n_rooms')} | footprint {fa.get('value', 0):.1f} m² [{fa.get('lo', 0):.1f}, {fa.get('hi', 0):.1f}] ({int(fa.get('level', .9)*100)}% CI)"
    ax.set_title(ttl + "\n" + sub, fontsize=11)
    ax.text(0.01, 0.01, "solid = observed wall   dashed = inferred   red/cyan/violet = door/window/opening (cm)",
            transform=ax.transAxes, fontsize=7, color="0.3")
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    if path_svg:
        fig.savefig(path_svg)
    plt.close(fig)
