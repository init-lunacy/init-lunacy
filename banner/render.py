"""Profile banner: one beam through a prism, split into a stream per project.

Each stream is that project's last 52 weeks of commits, drawn as signal: a quiet line in
weeks with nothing, a swing in weeks with work. Writes banner-dark.svg and banner-light.svg
next to this file: the rendered image plus CSS static (see animate()). Run by
.github/workflows/banner.yml every few hours.

  python banner/render.py            fetch fresh counts, then render
  python banner/render.py --offline  render from banner/data.json only
  add --force to redraw even when nothing changed

Needs numpy + Pillow. Fetching private repos needs GH_TOKEN with read access to them.
"""
import base64
import hashlib
import io
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
POKES = HERE / "pokes.json"

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


# --------------------------------------------------------------------------- pokes
# Visitors poke the prism by opening an issue titled "poke" (see .github/workflows/poke.yml).
# Each counted poke adds energy that decays over hours; energy turns up the banner's static.
# Only a hash of the login is kept, to limit each person to one counted poke an hour.

POKE_GAP = 3600          # seconds between counted pokes from the same person
POKE_DECAY = 6 * 3600    # energy e-folds over six hours


def load_pokes():
    return json.loads(POKES.read_text()) if POKES.exists() else {"count": 0, "log": []}


def poke(login, now=None):
    """Record a poke from `login`. Returns True if it counted."""
    now = now or time.time()
    p = load_pokes()
    who = hashlib.sha256(login.lower().encode()).hexdigest()[:12]
    if any(e["who"] == who and now - e["t"] < POKE_GAP for e in p["log"]):
        return False
    p["count"] += 1
    p["log"] = [e for e in p["log"] if now - e["t"] < 7 * POKE_DECAY] + [{"t": int(now), "who": who}]
    POKES.write_text(json.dumps(p, indent=1) + "\n")
    return True


def boost(now=None):
    """How much recent pokes turn up the static: 1 (calm) to 2.5, in quarter steps so the
    image only changes when the level does."""
    now = now or time.time()
    energy = sum(math.exp(-(now - e["t"]) / POKE_DECAY) for e in load_pokes()["log"])
    return round(min(2.5, 1 + .3 * energy) * 4) / 4


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


def render(theme, weeks, updated, pokes=0):
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
    geo = {"size": (W // S, H // S), "src": src, "entry": entry, "exit": exitp, "prism": back + tri, "streams": [], "labels": []}
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
        path = []
        for i in range(1200):
            u = i / 1199
            v = series[min(WEEKS - 1, int(u * WEEKS))]
            a = math.log1p(v) / math.log1p(peak) if v else 0
            wob = math.sin(u * 260) * a * 8 * S * (.35 + .65 * u)
            p = (exitp[0] + ux * L * u + nx * wob, exitp[1] + uy * L * u + ny * wob)
            if i % 8 == 0 or i == 1199:
                path.append(p)
            if prev:
                md.line([prev, p], fill=255, width=max(S, round((1.5 + (2 + 5 * share) * u) * S)))
            prev = p
        mask = np.asarray(m) > 0
        u_px = np.clip(((xx - exitp[0]) * ux + (yy - exitp[1]) * uy) / L, 0, 1)
        before = arr.copy()
        dither(arr, mask, .95 - .4 * u_px, hue)
        geo.setdefault("stream_masks", []).append((arr != before).any(axis=2))
        ribbons.append(mask)
        ends.append((label, hue, end))
        geo["streams"].append({"label": label, "hue": hue, "path": path, "width": (1.5 + 2 + 5 * share) * S})

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

    # The name rides the incoming beam, tilted to its angle; the stamp hangs just under it.
    name = font("Archivo-Variable.ttf", 25, width=125)
    mono = font("IBMPlexMono-Regular.ttf", 9)
    angle = math.degrees(math.atan(slope))

    def along_beam(text, fnt, color, anchor, gap):
        """Draw text parallel to the beam: `anchor` "ls" sits on the beam's top edge, "lt" hangs
        under its bottom edge, `gap` px away. Returns a mask of the pixels it covered."""
        tw = int(ImageDraw.Draw(img).textlength(text, font=fnt))
        pad = 40 * S
        tm = Image.new("L", (tw + 2 * pad, 3 * pad), 0)
        ImageDraw.Draw(tm).text((pad, 1.5 * pad), text, font=fnt, fill=255, anchor=anchor)
        tm = tm.rotate(angle, resample=Image.BICUBIC, center=(pad, 1.5 * pad))
        bx = src[0] + 2 * S
        edge = src[1] - slope * (bx - src[0])
        by = edge - 3 * S - gap if anchor == "ls" else edge + 3 * S + gap
        x0, y0 = int(bx - pad), int(by - 1.5 * pad)
        img.paste(Image.new("RGB", tm.size, color), (x0, y0), tm)
        cover = np.zeros((H, W), bool)
        ys, xs = np.nonzero(np.asarray(tm) > 0)
        ok = (ys + y0 >= 0) & (ys + y0 < H) & (xs + x0 >= 0) & (xs + x0 < W)
        cover[ys[ok] + y0, xs[ok] + x0] = True
        return cover

    name_mask = along_beam("Mohamed M. Elhassan", name, ink, "ls", 7 * S)
    stamp = f"52 weeks of commits · updated {updated}" if updated else "52 weeks of commits"
    if pokes:
        stamp += f" · poked {pokes:,}×"
    name_mask |= along_beam(stamp, mono, faint, "lt", 6 * S)
    d = ImageDraw.Draw(img)

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
        geo["labels"].append(box)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        taken[max(0, y0 - 4 * S):y1 + 4 * S, max(0, x0 - 4 * S):x1 + 4 * S] = True


    # Transparent background, so it sits on GitHub's own page colour in either theme.
    a = np.asarray(img)
    alpha = np.where((a == bg).all(axis=2), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([a, alpha]), "RGBA"), geo


# --------------------------------------------------------------------------- animation

STATIC = 1.6   # overall amount of static; the rates below are relative to it


def flicker(rng, xs, ys, colour, count, weights=None, fast=False):
    """`count` cells picked from (xs, ys) that blink `colour` on and off on their own clocks."""
    out = []
    if not len(xs) or count <= 0:
        return out
    for i in rng.choice(len(xs), size=min(count, len(xs)), replace=False, p=weights):
        dur = (.3 + rng.random() * .5) if fast else (.35 + rng.random() * .6)
        out.append(f'<rect class="n" x="{xs[i]}" y="{ys[i]}" width="{S}" height="{S}" fill="rgb{tuple(colour)}" '
                   f'style="animation-name:f{rng.integers(8)};animation-duration:{dur:.2f}s;animation-delay:{-rng.random():.2f}s"/>')
    return out


def animate(theme, weeks, updated, pokes=0, level=1.0):
    """The banner as an SVG: the rendered image, plus static that crackles in place. Light builds
    along the beam from the second "s" of Elhassan into the prism, the entry edge crackles where
    the beam strikes, and the coloured lanes carry a quieter, even static along their length.
    CSS only (GitHub strips scripts), and it all stops for prefers-reduced-motion."""
    img, geo = render(theme, weeks, updated, pokes)
    a = np.asarray(img)
    amount = STATIC * level
    bg, ink = THEMES[theme]["bg"], THEMES[theme]["ink"]
    rng = np.random.default_rng(3)                     # fixed, so the same data gives the same file
    gy, gx = np.mgrid[0:H, 0:W]
    grid = (gy % S == 0) & (gx % S == 0)               # one sample per dither cell
    lit = (a[..., :3] == ink).all(axis=2) & (a[..., 3] > 0)
    inside = poly_mask(geo["prism"])
    cells = []

    # Lanes: even static along each whole lane, in its own colour.
    for st, m in zip(geo["streams"], geo["stream_masks"]):
        m = m & ~inside
        halo = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0
        for pool, colour, rate in ((m & grid, bg, .07), (halo & ~m & grid & ~inside, st["hue"], .032)):
            ys, xs = np.nonzero(pool)
            cells += flicker(rng, xs, ys, colour, int(len(xs) * rate * amount))

    # Beam: starts after the second "s" of Elhassan (the name rides the beam), builds into the glass.
    (x0, y0), (x1, y1) = geo["src"], geo["entry"]
    L = math.hypot(x1 - x0, y1 - y0)
    dist = np.abs((gx - x0) * (y1 - y0) - (gy - y0) * (x1 - x0)) / L
    along = ((gx - x0) * (x1 - x0) + (gy - y0) * (y1 - y0)) / L
    lead = ImageDraw.Draw(Image.new("L", (1, 1))).textlength("Mohamed M. Elhass", font=font("Archivo-Variable.ttf", 25, width=125))
    start = 2 * S + lead
    for pool, colour, rate in ((lit & (dist < 3.2 * S), bg, .12), (~lit & (dist >= 3.2 * S) & (dist < 4.6 * S), ink, .07)):
        ys, xs = np.nonzero(pool & grid & (along > start) & (along < L + 2 * S))
        if len(xs):
            w = np.clip((along[ys, xs] - start) / (L - start), 0, 1) ** 2.4 + .005
            cells += flicker(rng, xs, ys, colour, int(len(xs) * rate * amount), w / w.sum())

    # Entry edge: crackles where the beam strikes, fading up and down the edge.
    tri = geo["prism"][3:]
    (ax, ay), (bx, by) = tri[0], tri[2]                 # top vertex to bottom-left vertex
    EL = math.hypot(bx - ax, by - ay)
    edist = np.abs((gx - ax) * (by - ay) - (gy - ay) * (bx - ax)) / EL
    ealong = ((gx - ax) * (bx - ax) + (gy - ay) * (by - ay)) / EL
    hit = ((x1 - ax) * (bx - ax) + (y1 - ay) * (by - ay)) / EL
    for pool, colour, rate in ((lit & (edist < 1.6 * S), bg, .35), (~lit & (edist < 3.2 * S), ink, .18)):
        ys, xs = np.nonzero(pool & grid & (ealong > 0) & (ealong < EL))
        if len(xs):
            w = np.exp(-((ealong[ys, xs] - hit) / (22 * S)) ** 2) + .002
            cells += flicker(rng, xs, ys, colour, int(len(xs) * rate * amount), w / w.sum(), fast=True)

    # Eight on/off patterns, mostly off, stepped so cells snap rather than fade.
    css = [".n{opacity:0;animation-timing-function:steps(1,end);animation-iteration-count:infinite}"]
    for t in range(8):
        on = rng.random(16) < .3
        css.append(f"@keyframes f{t}{{" + "".join(f"{i * 100 / 16:.2f}%{{opacity:{int(v)}}}" for i, v in enumerate(on)) + "}")
    css.append("@media (prefers-reduced-motion: reduce){.n{display:none;animation:none}}")

    buf = io.BytesIO(); img.save(buf, "PNG", optimize=True)
    base = base64.b64encode(buf.getvalue()).decode()
    w, h = geo["size"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{w}" height="{h}" shape-rendering="crispEdges">'
            f'<title>Mohamed M. Elhassan</title><style>{"".join(css)}</style>'
            f'<image href="data:image/png;base64,{base}" width="{W}" height="{H}"/>' + "".join(cells) + "</svg>")


def main():
    counted = None
    if "--poke" in sys.argv:
        counted = poke(sys.argv[sys.argv.index("--poke") + 1])
        print("poke counted" if counted else "poke not counted (same person within the hour)")
        if out := os.environ.get("GITHUB_OUTPUT"):
            with open(out, "a") as f:
                f.write(f"counted={'true' if counted else 'false'}\n")
    changed = refresh() if "--offline" not in sys.argv else False
    data = json.loads(DATA.read_text())
    pokes, level = load_pokes()["count"], boost()
    # Redraw only when something visible changed: the counts, the pokes, the poke energy level,
    # or this script. Pillow output differs between machines, so redrawing identical inputs
    # would otherwise commit a "new" image every run.
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    state = {"render": version, "pokes": pokes, "level": level}
    outs = [HERE / f"banner-{theme}.svg" for theme in THEMES]
    if not changed and data.get("state") == state and all(o.exists() for o in outs) and "--force" not in sys.argv:
        print("No change.")
        return
    for theme, out in zip(THEMES, outs):
        out.write_text(animate(theme, data["weeks"], data.get("updated"), pokes, level))
        print(out.name, out.stat().st_size, "bytes", f"(static x{level})")
    data.pop("render", None)
    data["state"] = state
    DATA.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
