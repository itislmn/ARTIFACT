"""
glare.py - Radar-Anchored Glare Removal: the radar says which bright spot is the lamp, the veil L = b + a rho^-p is
fitted around it and subtracted, the clipped core is cut out.

    python -m backend.glare selftest
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

STEP = 4
N_RINGS = 24


@dataclass
class Glare:
    source: int
    centre_uv: tuple
    core_px: float
    fits: list
    ms: float
    profile: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict, repr=False)
    scores: dict = field(default_factory=dict)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def band(shape, bar, hw: float) -> np.ndarray:
    h, w = shape
    m = np.zeros(shape, bool)
    for (u0, v0), (u1, v1) in zip(bar[:-1], bar[1:]):
        x0, x1 = int(max(min(u0, u1) - hw, 0)), int(min(max(u0, u1) + hw + 1, w))
        y0, y1 = int(max(min(v0, v1) - hw, 0)), int(min(max(v0, v1) + hw + 1, h))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def bar_mask(shape, bars: list, half_width_px: list) -> np.ndarray:
    m = np.zeros(shape, bool)
    for bar, hw in zip(bars, half_width_px):
        if bar is not None:
            m |= band(shape, np.asarray(bar, float), hw)
    return m


def candidates(clipped: np.ndarray, bars: list, half_width_px: list) -> list:
    out = []
    for i, (bar, hw) in enumerate(zip(bars, half_width_px)):
        if bar is None:
            continue
        ys, xs = np.nonzero(clipped & band(clipped.shape, np.asarray(bar, float), hw))
        if ys.size:
            cu, cv = xs.mean(), ys.mean()
            out.append((i, (cu, cv), float(np.hypot(xs - cu, ys - cv).max())))
    return out


def fit_veil(channel_s: np.ndarray, usable_s: np.ndarray, centre, core_px: float):
    h, w = channel_s.shape
    ys, xs = np.mgrid[0:h, 0:w] * STEP
    rho = np.hypot(xs - centre[0], ys - centre[1])
    ok = usable_s & (rho > max(core_px, 2 * STEP))
    vals, rho = channel_s[ok], rho[ok]
    edges = np.geomspace(max(core_px, 2 * STEP), rho.max() + 1, N_RINGS + 1)
    ring = np.digitize(rho, edges) - 1
    radii, levels = [], []
    for k in range(N_RINGS):
        v = vals[ring == k]
        if v.size >= 20:
            radii.append(np.sqrt(edges[k] * edges[k + 1]))
            levels.append(np.median(v))
    radii, levels = np.array(radii), np.array(levels)
    best = None
    for p in np.arange(0.5, 4.0001, 0.05):
        A = np.c_[np.ones_like(radii), radii ** -p]
        (b, a), *_ = np.linalg.lstsq(A, levels, rcond=None)
        err = np.sum((A @ [b, a] - levels) ** 2)
        if a >= 0 and (best is None or err < best[0]):
            best = (err, float(b), float(a), float(p))
    if best is None:
        return (float(np.median(levels)), 0.0, 2.0), radii, levels
    return best[1:], radii, levels


def fit_at(rgb: np.ndarray, clipped: np.ndarray, source: int, centre, core_px: float, exclude: np.ndarray) -> Glare:
    t0 = time.perf_counter()
    usable_s = (~clipped & ~exclude)[::STEP, ::STEP]
    rgb_s = rgb[::STEP, ::STEP].astype(np.float32)
    fits, prof = [], {"radii": None, "levels": []}
    for c in range(3):
        fit, radii, levels = fit_veil(rgb_s[..., c], usable_s, centre, core_px)
        fits.append(fit)
        prof["radii"] = radii
        prof["levels"].append(levels)
    return Glare(source, tuple(centre), core_px, fits, 1e3 * (time.perf_counter() - t0), prof)


def locate_source(uv, objects: list, bars: list, proj, cfg: dict) -> dict | None:
    z_lo, z_hi = cfg["alignment"]["bar_z_m"]
    best = None
    for i, (o, bar) in enumerate(zip(objects, bars)):
        if bar is None:
            continue
        lo, hi = z_lo, z_hi
        for _ in range(30):
            z = (lo + hi) / 2
            v = proj.point(o["range_m"], o["azimuth_deg"], z)[1]
            lo, hi = (lo, z) if v < uv[1] else (z, hi)
        u_bar = proj.point(o["range_m"], o["azimuth_deg"], z)[0]
        tol = 3 * proj.loo_rms_px + proj.f * cfg["light"]["source_size_m"] / 2 / o["range_m"]
        du = abs(uv[0] - u_bar)
        if du <= tol and (best is None or du < best["du_px"]):
            y = o["range_m"] * np.sin(np.radians(o["azimuth_deg"]))
            x = float(np.sqrt(max(o["range_m"] ** 2 - y ** 2 - z ** 2, 0.0)))
            best = {"object": i, "range_m": o["range_m"], "azimuth_deg": o["azimuth_deg"], "z_m": round(float(z), 3),
                    "xyz_m": [round(x, 3), round(float(y), 3), round(float(z), 3)],
                    "anchor_uv": [round(float(u_bar), 1), round(float(uv[1]), 1)], "blob_uv": [round(uv[0], 1), round(uv[1], 1)],
                    "du_px": round(float(du), 1), "tolerance_px": round(float(tol), 1)}
    return best


def analyse(rgb: np.ndarray, clipped: np.ndarray, bars: list, half_width_px: list) -> Glare | None:
    t0 = time.perf_counter()
    cands = candidates(clipped, bars, half_width_px)
    if not cands:
        return None
    usable_s = (~clipped & ~bar_mask(clipped.shape, bars, half_width_px))[::STEP, ::STEP]
    rgb_s = rgb[::STEP, ::STEP].astype(np.float32)
    Y_s = luma(rgb_s)

    def veil_at_100px(c):
        (b, a, p), _, _ = fit_veil(Y_s, usable_s, c[1], c[2])
        return a * 100.0 ** -p

    scores = {c[0]: float(veil_at_100px(c)) for c in cands}
    i, centre, core = max(cands, key=lambda c: scores[c[0]])
    g = fit_at(rgb, clipped, i, centre, core, bar_mask(clipped.shape, bars, half_width_px))
    g.scores, g.ms = scores, 1e3 * (time.perf_counter() - t0)
    return g


def remove(rgb: np.ndarray, g: Glare, clipped: np.ndarray, scale: float = 1.0) -> np.ndarray:
    h, w = clipped.shape
    key = ((h, w), scale)
    if key not in g.cache:
        yy, xx = np.ogrid[0:h, 0:w]
        rho = np.maximum(np.hypot(xx / scale - g.centre_uv[0], yy / scale - g.centre_uv[1]), max(g.core_px, 1.0))
        veil = np.stack([a * rho ** -p for b, a, p in g.fits], axis=2).astype(np.float32)
        g.cache[key] = (veil, ((xx + yy) // 6) % 2 == 0)
    veil, hatch = g.cache[key]
    out = np.clip(rgb.astype(np.float32) - veil, 0, 255).astype(np.uint8)
    out[clipped & hatch] = 200
    out[clipped & ~hatch] = 235
    return out


def clean(rgb: np.ndarray, g: Glare, clipped: np.ndarray, edge_px: int = 3) -> np.ndarray:
    h, w = clipped.shape
    yy, xx = np.ogrid[0:h, 0:w]
    rho = np.maximum(np.hypot(xx - g.centre_uv[0], yy - g.centre_uv[1]), max(g.core_px, 1.0))
    veil = np.stack([a * rho ** -p for b, a, p in g.fits], axis=2).astype(np.float32)
    out = np.clip(rgb.astype(np.float32) - veil, 0, 255)
    cut = clipped.copy()
    for _ in range(edge_px):
        cut[1:] |= cut[:-1].copy()
        cut[:-1] |= cut[1:].copy()
        cut[:, 1:] |= cut[:, :-1].copy()
        cut[:, :-1] |= cut[:, 1:].copy()
    out[cut] = 255.0
    return out.astype(np.uint8)


def selftest() -> bool:
    from backend.pipeline import correct_chars
    from backend.settings import load_config
    from readers.glyphs import TEST_FONTS, add_veil, word_scene
    from readers.metricsight import read
    cfg = load_config()
    rng = np.random.default_rng(11)
    f, R, text = 950.0, 1.0, "ARTIFACT"
    c_text = (640.0, 380.0)
    src = (640.0 - f * 0.30 / R, 380.0 - 40)
    ok = True
    for level in (0, 60, 120, 200):
        hits = 0
        p_fit = []
        for k in range(4):
            Y = add_veil(word_scene(text, TEST_FONTS[k % 2], int(f * 0.05 / R), centre=c_text, rng=rng), src, level, c_text)
            rgb = np.repeat(Y[..., None], 3, axis=2).astype(np.uint8)
            clipped = Y >= 250
            hits += correct_chars(read(rgb, c_text, R, f, cfg).text, text)
            bars = [((src[0], src[1] - 80), (src[0], src[1] + 80)), ((c_text[0], c_text[1] - 80), (c_text[0], c_text[1] + 80))]
            g = analyse(rgb, clipped, bars, [f * 0.15 / R, f * 0.15 / R])
            if g is not None and level > 0:
                ok &= g.source == 0
                p_fit.append(g.fits[0][2])
        extra = f", fitted exponent p {np.median(p_fit):.2f} (true 2)" if p_fit else ""
        print(f"  veil {level:3d} grey levels at the text: characters read {hits}/32{extra}")
        if level > 0 and p_fit:
            ok &= abs(np.median(p_fit) - 2) < 0.3
    print(f"  {'PASS' if ok else 'FAIL'}  light source identified as the radar object with the clipped core, "
          f"veil exponent recovered")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest"])
    ap.parse_args()
    raise SystemExit(0 if selftest() else 1)


if __name__ == "__main__":
    main()
