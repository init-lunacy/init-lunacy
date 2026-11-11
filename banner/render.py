"""Profile banner: one beam through a prism, split into a stream per project.

Each stream is that project's last 52 weeks of commits, drawn as signal: a quiet line in
weeks with nothing, a swing in weeks with work. Renders banner-dark.png and banner-light.png
next to this file. Run by .github/workflows/banner.yml every few hours.

  python banner/render.py            fetch fresh counts, then render
  python banner/render.py --offline  render from banner/data.json only
  add --force to redraw even when nothing changed

Needs numpy + Pillow. Fetching private repos needs GH_TOKEN with read access to them.
"""
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
DATA = HERE / "data.json"

# label, repo, hue (dark theme), hue (light theme)
STREAMS = [
    ("MDMESH", "MDMesh-app/MDMesh", (232, 98, 82), (186, 62, 48)),
    ("CREDPLOT", "init-lunacy/credplot", (236, 196, 92), (160, 112, 10)),
    ("BIOLAIR", "init-lunacy/biolayer", (96, 178, 222), (32, 112, 160)),
]
THEMES = {
    "dark": dict(bg=(18, 19, 18), ink=(239, 238, 233), faint=(143, 146, 140), hue=2),
    "light": dict(bg=(245, 244, 239), ink=(23, 24, 23), faint=(109, 111, 105), hue=3),
}
S = 2                       # render scale; GitHub shows READMEs ~880 px wide
W, H = 880 * S, 150 * S
WEEKS = 52
BAYER8 = (np.array([
    0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
    12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
    3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
    15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
]).reshape(8, 8) + .5) / 64


# --------------------------------------------------------------------------- data

def fetch_weeks(repo, token):
    """Weekly commit totals for the last 52 weeks. GitHub computes these lazily and answers
    202 until they're ready, so retry for a while before giving up."""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/stats/commit_activity",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "profile-banner",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    for _ in range(10):
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status == 200:
                return [w["total"] for w in json.load(r)][-WEEKS:]
        time.sleep(6)
    raise RuntimeError(f"{repo}: commit stats not ready")


def refresh():
    """Update data.json; returns True if any count changed. Keeps old numbers for a repo
    that can't be read this time, so one failed request never blanks a stream."""
    old = json.loads(DATA.read_text()) if DATA.exists() else {"weeks": {}, "updated": None}
    token = os.environ.get("GH_TOKEN")
    weeks = dict(old["weeks"])
    for label, repo, *_ in STREAMS:
        try:
            weeks[label] = fetch_weeks(repo, token)
        except Exception as e:  # noqa: BLE001 - keep the last good data and say why
            print(f"warning: {repo}: {e}", file=sys.stderr)
    changed = weeks != old["weeks"]
    if changed:
        DATA.write_text(json.dumps({"weeks": weeks, "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d")}, indent=1) + "\n")
    return changed


# --------------------------------------------------------------------------- drawing

def font(name, px, width=None):
    f = ImageFont.truetype(str(HERE / "fonts" / name), px * S)
    if width:
        f.set_variation_by_axes([400, width])  # Archivo: weight, width
    return f


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def dither(arr, mask, tone, color):
    """Write `color` into arr through an ordered dither wherever mask is set; tone is 0..1 per pixel."""
    ys, xs = np.nonzero(mask)
    on = tone[ys, xs] > BAYER8[(ys // S) & 7, (xs // S) & 7]
    arr[ys[on], xs[on]] = color


def poly_mask(pts):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.asarray(m) > 0


def beam_tone(a, b, w0, w1, t0, t1):
    """Tone field for a ray from a to b: brightest on its axis, fading t0 -> t1 along it."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    u = np.clip(((xx - a[0]) * ux + (yy - a[1]) * uy) / L, 0, 1)
    across = np.abs(-(xx - a[0]) * uy + (yy - a[1]) * ux) / (w0 / 2 + (w1 - w0) / 2 * u + 1e-6)
    return (t0 + (t1 - t0) * u) * (1 - .5 * across ** 2)


def beam(arr, a, b, w0, w1, t0, t1, color):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L
    pts = [(a[0] + nx * w0 / 2, a[1] + ny * w0 / 2), (b[0] + nx * w1 / 2, b[1] + ny * w1 / 2),
           (b[0] - nx * w1 / 2, b[1] - ny * w1 / 2), (a[0] - nx * w0 / 2, a[1] - ny * w0 / 2)]
    dither(arr, poly_mask(pts), beam_tone(a, b, w0, w1, t0, t1), color)


def render(theme, weeks, updated):
    t = THEMES[theme]
    bg, ink, faint = t["bg"], t["ink"], t["faint"]
    arr = np.zeros((H, W, 3), np.uint8); arr[:] = bg

    # Geometry: a prism with some depth; the beam enters its left face, leaves its right.
    cx, cy, size = 468 * S, 80 * S, 100 * S
    h = size * math.sqrt(3) / 2
    tri = [(cx, cy - h * 2 / 3), (cx + size / 2, cy + h / 3), (cx - size / 2, cy + h / 3)]
    depth = (24 * S, -14 * S)
    back = [(x + depth[0], y + depth[1]) for x, y in tri]
    entry = ((tri[0][0] + tri[2][0]) / 2, (tri[0][1] + tri[2][1]) / 2)
    exitp = ((tri[0][0] + tri[1][0]) / 2 + depth[0] * .35, (tri[0][1] + tri[1][1]) / 2 + depth[1] * .35)
    slope = .05                                        # the beam rises gently toward the prism
    src = (40 * S, entry[1] + slope * (entry[0] - 40 * S))

    beam(arr, src, entry, 6 * S, 6 * S, .95, .95, ink)
    beam(arr, entry, exitp, 5 * S, 3 * S, .35, .45, ink)   # faintly visible inside the glass

    # Streams: each project's year as a wave, drawn as a dense dithered ribbon in its colour.
    # The ribbon widens away from the prism, more for projects with more commits that year.
    x_end = W - 40 * S
    fan = np.linspace(-.12, .12, len(STREAMS))
    peak = max([max(weeks.get(s[0], [0]) or [0]) for s in STREAMS] + [1])
    big = max([sum(weeks.get(s[0], [])) for s in STREAMS] + [1])
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    ribbons, ends = [], []
    for (label, _, *hues), ang in zip(STREAMS, fan):
        hue = hues[0] if theme == "dark" else hues[1]
        series = (weeks.get(label) or [0] * WEEKS)[-WEEKS:]
        share = math.sqrt(sum(series) / big)
        end = (x_end, exitp[1] + math.tan(ang) * (x_end - exitp[0]))
        L = math.hypot(end[0] - exitp[0], end[1] - exitp[1])
        ux, uy = (end[0] - exitp[0]) / L, (end[1] - exitp[1]) / L
        nx, ny = -uy, ux
        m = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(m)
        prev = None
        for i in range(1200):
            u = i / 1199
            v = series[min(WEEKS - 1, int(u * WEEKS))]
            a = math.log1p(v) / math.log1p(peak) if v else 0
            wob = math.sin(u * 260) * a * 8 * S * (.35 + .65 * u)
            p = (exitp[0] + ux * L * u + nx * wob, exitp[1] + uy * L * u + ny * wob)
            if prev:
                md.line([prev, p], fill=255, width=max(S, round((1.5 + (2 + 5 * share) * u) * S)))
            prev = p
        mask = np.asarray(m) > 0
        u_px = np.clip(((xx - exitp[0]) * ux + (yy - exitp[1]) * uy) / L, 0, 1)
        dither(arr, mask, .95 - .4 * u_px, hue)
        ribbons.append(mask)
        ends.append((label, hue, end))

    # The prism, over the streams: dithered faces, crisp front edges.
    for face, tone in (([tri[0], back[0], back[1], tri[1]], .1), ([tri[2], back[2], back[1], tri[1]], .06)):
        dither(arr, poly_mask(face), np.full((H, W), tone), ink)
    top, bot = tri[0][1], tri[1][1]
    dither(arr, poly_mask(tri), .06 + .16 * np.clip((yy - top) / (bot - top), 0, 1), ink)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for a, b in zip(tri, back):
        d.line([a, b], fill=mix(bg, ink, .5), width=1)
    d.line([back[0], back[1], back[2], back[0]], fill=mix(bg, ink, .35), width=1)
    d.line([tri[0], tri[1], tri[2], tri[0]], fill=ink, width=S)
    d.line([entry, ((entry[0] + exitp[0]) / 2, (entry[1] + exitp[1]) / 2)], fill=mix(bg, ink, .55), width=S)

    # The name rides the incoming beam, tilted to its angle.
    name = font("Archivo-Variable.ttf", 25, width=125)
    mono = font("IBMPlexMono-Regular.ttf", 9)
    text = "Mohamed M. Elhassan"
    tw = int(d.textlength(text, font=name))
    pad = 40 * S
    tm = Image.new("L", (tw + 2 * pad, 3 * pad), 0)
    ImageDraw.Draw(tm).text((pad, 2 * pad), text, font=name, fill=255, anchor="ls")
    angle = math.degrees(math.atan(slope))
    tm = tm.rotate(angle, resample=Image.BICUBIC, center=(pad, 2 * pad))
    # Baseline sits just above the beam's top edge, starting where the beam starts.
    bx = src[0] + 2 * S
    by = src[1] - slope * (bx - src[0]) - 3 * S - 7 * S
    img.paste(Image.new("RGB", tm.size, ink), (int(bx - pad), int(by - 2 * pad)), tm)
    d = ImageDraw.Draw(img)
    name_mask = np.zeros((H, W), bool)
    x0, y0 = int(bx - pad), int(by - 2 * pad)
    tmv = np.asarray(tm) > 0
    ys, xs = np.nonzero(tmv)
    ok = (ys + y0 >= 0) & (ys + y0 < H) & (xs + x0 >= 0) & (xs + x0 < W)
    name_mask[ys[ok] + y0, xs[ok] + x0] = True

    # Labels: try spots along each stream; take the first that touches no stream, prism, name or label.
    clear = Image.new("L", (W, H), 0)
    ImageDraw.Draw(clear).polygon(back + tri, fill=255)
    taken = (np.asarray(clear) > 0) | name_mask
    for mask in ribbons:
        grown = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
        taken |= np.asarray(grown) > 0

    def free(box):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if x0 < 0 or y0 < 2 * S or x1 > W or y1 > H - 2 * S:
            return False
        return not taken[y0:y1, x0:x1].any()

    th = 11 * S
    for label, hue, end in ends:
        lw = d.textlength(label, font=mono)
        cands = []
        for u in (1.0, .92, .84, .76, .68):     # from the end back toward the prism
            px = exitp[0] + (end[0] - exitp[0]) * u
            py = exitp[1] + (end[1] - exitp[1]) * u
            for gap in (3, 6, 10, 14):          # above, then below, as close as fits
                g = gap * S
                cands.append((px - lw, py - g - th, px, py - g))
                cands.append((px - lw, py + g, px, py + g + th))
        box = next((b for b in cands if free(b)), cands[0])
        d.text((box[0], box[1]), label, font=mono, fill=hue)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        taken[max(0, y0 - 4 * S):y1 + 4 * S, max(0, x0 - 4 * S):x1 + 4 * S] = True

    stamp = f"52 weeks of commits · updated {updated}" if updated else "52 weeks of commits"
    d.text((src[0], src[1] + 9 * S), stamp, font=mono, fill=faint)

    # Transparent background, so it sits on GitHub's own page colour in either theme.
    a = np.asarray(img)
    alpha = np.where((a == bg).all(axis=2), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([a, alpha]), "RGBA")


def main():
    changed = refresh() if "--offline" not in sys.argv else False
    data = json.loads(DATA.read_text())
    # Re-render only when the counts or this script changed: Pillow output can differ between
    # machines, and redrawing identical data would otherwise commit a "new" image every run.
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    outs = [HERE / f"banner-{theme}.png" for theme in THEMES]
    if not changed and data.get("render") == version and all(o.exists() for o in outs) and "--force" not in sys.argv:
        print("No change.")
        return
    for theme, out in zip(THEMES, outs):
        render(theme, data["weeks"], data.get("updated")).save(out, optimize=True)
        print(out.name, out.stat().st_size, "bytes")
    data["render"] = version
    DATA.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
