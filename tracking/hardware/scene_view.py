"""
scene_view.py - the picture of the experiment, the same live and in the videos: the camera with the prediction
regions in the classifier's colours, the radar top view, the ball crop and three lines in Computer Modern.

    img = SceneView().frame(rgb, boxes, crop_rect, gt, top, lines, banner)
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
CHIRP = {"search": "search chirp, 10 frames/s", "track": "track chirp, 20 frames/s",
         "track_fast": "fast chirp, 20 frames/s"}
COL = {"ball": (213, 94, 0), "person": (0, 114, 178), "object": (127, 127, 127)}


def _classifier_colours():
    try:
        import sys

        from matplotlib.colors import to_rgb
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from material_classification.plots import MATERIAL_COLORS, MOTION_COLORS
        conv = lambda d: {k: tuple(int(round(255 * c)) for c in to_rgb(v)) for k, v in d.items()}
        return conv(MATERIAL_COLORS), conv(MOTION_COLORS)
    except Exception:                                                     # noqa: BLE001  (same values, written out)
        return ({"metal": (0, 114, 178), "dielectric": (230, 159, 0), "waterlike": (0, 158, 115), "none": (153, 153, 153)},
                {"static": (0, 0, 0), "rigid": (213, 94, 0), "articulated": (204, 121, 167), "distributed": (86, 180, 233),
                 "none": (153, 153, 153)})


MATERIAL_COL, MOTION_COL = _classifier_colours()
MOTION_COL = {**MOTION_COL, "moving": (96, 96, 96)}


def scene_style(kind: str) -> dict:
    return {"color": COL[kind], "color2": None, "tags": [(kind, COL[kind])] if kind != "object" else []}


def classifier_style(tr: dict) -> dict:
    c = tr.get("classes") or {}
    mat = (c.get("material") or {}).get("label")
    mot = (c.get("motion") or {}).get("label")
    if not mat and not mot:
        g = MATERIAL_COL["none"]
        return {"color": g, "color2": None, "tags": [("not yet classified", g)]}
    cm = MATERIAL_COL.get(mat, MATERIAL_COL["none"]) if mat else MATERIAL_COL["none"]
    cn = MOTION_COL.get(mot, MOTION_COL["none"]) if mot else MOTION_COL["none"]
    return {"color": cm, "color2": cn, "tags": [(mat or "?", cm), (mot or "?", cn)]}


def ink(rgb) -> tuple:
    return (0, 0, 0) if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 160 else (255, 255, 255)
CAM = (6, 8, 800, 450)
CROP = (6, 494, 220, 220)
TOP_X0, TOP_Y0, K = 878, 44, 128.0
LAT0, LAT1, RNG1 = -2.0, 1.0, 3.2
HALF_FOV = 29.9


def _font_file(name: str) -> str | None:
    if shutil.which("kpsewhich"):
        try:
            p = subprocess.run(["kpsewhich", name], capture_output=True, text=True, timeout=10).stdout.strip()
            if p and Path(p).exists():
                return p
        except Exception:                                # noqa: BLE001
            pass
    import matplotlib
    alt = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / ("cmr10.ttf" if "rm" in name else "cmmi10.ttf")
    return str(alt) if alt.exists() else None


def math_italic(s: str) -> str:
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append("ℎ" if c == "h" else chr(0x1D44E + ord(c) - ord("a")))
        elif "A" <= c <= "Z":
            out.append(chr(0x1D434 + ord(c) - ord("A")))
        elif c == "α":
            out.append(chr(0x1D6FC))
        else:
            out.append(c)
    return "".join(out)


class Fonts:
    def __init__(self):
        rm, mt = _font_file("cmunrm.otf"), _font_file("latinmodern-math.otf")
        self.cmu = rm is not None and rm.endswith(".otf") and mt is not None and mt.endswith(".otf")
        self._rm, self._it, self._cache = rm, (mt if self.cmu else rm), {}

    def get(self, size: int, italic: bool = False):
        key = (size, italic)
        if key not in self._cache:
            path = self._it if italic else self._rm
            self._cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        return self._cache[key]

    def text(self, draw: ImageDraw.ImageDraw, xy, s: str, size: int, fill=(0, 0, 0), anchor="la", halo: int = 0):
        if not self.cmu:
            s = s.replace("α", "alpha")
        parts = s.split("$")
        if self.cmu:
            parts = [math_italic(q) if i % 2 == 1 else q for i, q in enumerate(parts)]
        widths = [draw.textlength(p, font=self.get(size, i % 2 == 1)) for i, p in enumerate(parts)]
        x, y = xy
        if anchor[0] == "m":
            x -= sum(widths) / 2
        for i, (p, w) in enumerate(zip(parts, widths)):
            if p:
                draw.text((x, y), p, font=self.get(size, i % 2 == 1), fill=fill, anchor="l" + anchor[1],
                          stroke_width=halo, stroke_fill=(255, 255, 255))
            x += w
        return sum(widths)


def px(lat: float, rng: float):
    return TOP_X0 + (lat - LAT0) * K, TOP_Y0 + (RNG1 - rng) * K


def wedge(lo, hi, n: int = 12) -> np.ndarray:
    th = np.linspace(lo[1], hi[1], n)
    return np.array([(r * np.sin(a), r * np.cos(a)) for r, a in [(lo[0], a) for a in th] + [(hi[0], a) for a in th[::-1]]])


class SceneView:
    def __init__(self, note: str = ""):
        self.fonts = Fonts()
        self.note = note
        self.bg = self._background()

    def _background(self) -> np.ndarray:
        img = np.full((H, W, 3), 255, np.uint8)
        x1, y1 = px(LAT1, 0.0)
        for r in (1, 2, 3):
            pts = np.array([px(r * np.sin(a), r * np.cos(a)) for a in np.radians(np.linspace(-70, 70, 120))])
            ok = (pts[:, 0] >= TOP_X0) & (pts[:, 0] <= x1) & (pts[:, 1] >= TOP_Y0)
            cv2.polylines(img, [pts[ok].astype(np.int32)], False, (220, 220, 220), 1, cv2.LINE_AA)
        for s in (-1, 1):
            for d in np.arange(0.05, 3.4, 0.08):
                a, b = px(d * np.sin(np.radians(s * HALF_FOV)), d * np.cos(np.radians(s * HALF_FOV))), \
                    px((d + 0.035) * np.sin(np.radians(s * HALF_FOV)), (d + 0.035) * np.cos(np.radians(s * HALF_FOV)))
                if TOP_X0 <= a[0] <= x1 and a[1] >= TOP_Y0:
                    cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (150, 150, 150), 1, cv2.LINE_AA)
        ox, oy = px(0, 0)
        cv2.fillPoly(img, [np.array([(ox - 7, oy), (ox + 7, oy), (ox, oy - 12)], np.int32)], (0, 0, 0), cv2.LINE_AA)
        cv2.rectangle(img, (TOP_X0, TOP_Y0), (int(x1), int(y1)), (77, 77, 77), 1)
        for lat in np.arange(LAT0, LAT1 + 1e-9, 0.5):
            x = int(px(lat, 0)[0])
            cv2.line(img, (x, int(y1)), (x, int(y1) + 5), (77, 77, 77), 1)
        for rng in np.arange(0, RNG1 + 1e-9, 0.5):
            y = int(px(LAT0, rng)[1])
            cv2.line(img, (TOP_X0 - 5, y), (TOP_X0, y), (77, 77, 77), 1)
        lx, ly = TOP_X0 + 20, int(y1) + 66
        cv2.circle(img, (lx, ly), 4, (110, 110, 110), -1, cv2.LINE_AA)
        ov = img.copy()
        cv2.rectangle(ov, (lx - 9, ly + 22), (lx + 9, ly + 32), (90, 90, 90), -1)
        img[:] = cv2.addWeighted(ov, 0.3, img, 0.7, 0)
        cv2.line(img, (lx + 175, ly), (lx + 205, ly), (90, 90, 90), 2, cv2.LINE_AA)
        cv2.circle(img, (lx + 190, ly), 5, (90, 90, 90), -1, cv2.LINE_AA)
        for d in range(lx + 175, lx + 205, 6):
            cv2.line(img, (d, ly + 27), (d + 3, ly + 27), (150, 150, 150), 1, cv2.LINE_AA)
        pil = Image.fromarray(img)
        dr = ImageDraw.Draw(pil)
        f = self.fonts
        for lat in np.arange(LAT0, LAT1 + 1e-9, 0.5):
            f.text(dr, (px(lat, 0)[0], y1 + 9), f"{lat:.1f}".replace("-", "−"), 15, anchor="mt")
        for rng in np.arange(0, RNG1 + 1e-9, 0.5):
            w = dr.textlength(f"{rng:.1f}", font=f.get(15))
            f.text(dr, (TOP_X0 - 9 - w, px(LAT0, rng)[1]), f"{rng:.1f}", 15, anchor="lm")
        f.text(dr, ((TOP_X0 + x1) / 2, y1 + 32), "lateral (m)", 17, anchor="mt")
        f.text(dr, ((TOP_X0 + x1) / 2, TOP_Y0 - 30), "radar, top view", 18, anchor="mt")
        lab = Image.new("RGB", (120, 24), "white")
        ImageDraw.Draw(lab).text((60, 12), "range (m)", font=f.get(17), fill=(0, 0, 0), anchor="mm")
        lab = lab.rotate(90, expand=True)
        pil.paste(lab, (TOP_X0 - 72, int((TOP_Y0 + y1) / 2) - 60))
        for (dx, dy), s in (((14, 0), "detection"), ((14, 27), "prediction (90 %)"), ((222, 0), "track, last 3 s"),
                            ((222, 27), "camera view")):
            f.text(dr, (lx + dx, ly + dy), s, 15, anchor="lm")
        if self.note:
            f.text(dr, (lx - 9, ly + 56), self.note, 15, fill=(60, 60, 60), anchor="lm")
        return np.asarray(pil).copy()

    def frame(self, rgb, boxes, crop_rect, gt, top, lines, banner=None) -> np.ndarray:
        img = self.bg.copy()
        hc, wc = rgb.shape[:2]
        cx, cy, cw, ch = CAM
        s = cw / wc
        img[cy:cy + ch, cx:cx + cw] = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
        cam = img[cy:cy + ch, cx:cx + cw]
        labels = []
        for box in boxes:
            u0, v0, u1, v1 = box["rect"]
            a = (int(np.clip(u0 * s, -10, cw + 10)), int(np.clip(v0 * s, -10, ch + 10)))
            b = (int(np.clip(u1 * s, -10, cw + 10)), int(np.clip(v1 * s, -10, ch + 10)))
            if b[0] <= 0 or a[0] >= cw or b[1] <= 0 or a[1] >= ch:
                continue
            w1 = 4 if box.get("thick") else 3
            cv2.rectangle(cam, a, b, box["color"], w1, cv2.LINE_AA)
            if box.get("color2") is not None:
                d = w1
                cv2.rectangle(cam, (a[0] + d, a[1] + d), (b[0] - d, b[1] - d), box["color2"], 2, cv2.LINE_AA)
            if box.get("tags"):
                labels.append((box["tags"], (cx + max(a[0], 0) + 2, cy + max(a[1], 0) + 2)))
        if crop_rect is not None:
            u0, v0, u1, v1 = (int(c * s) for c in crop_rect)
            for (xa, ya), (xb, yb) in (((u0, v0), (u1, v0)), ((u1, v0), (u1, v1)), ((u1, v1), (u0, v1)), ((u0, v1), (u0, v0))):
                n = max(int(np.hypot(xb - xa, yb - ya) / 12), 1)
                for k in range(n):
                    p = (xa + (xb - xa) * k / n, ya + (yb - ya) * k / n)
                    q = (xa + (xb - xa) * (k + 0.55) / n, ya + (yb - ya) * (k + 0.55) / n)
                    cv2.line(cam, tuple(map(int, p)), tuple(map(int, q)), (0, 0, 0), 2, cv2.LINE_AA)
        if gt is not None:
            u, v, r = (c * s for c in gt)
            for a0 in range(0, 360, 20):
                cv2.ellipse(cam, (int(u), int(v)), (int(r), int(r)), 0, a0, a0 + 9, (255, 255, 255), 1, cv2.LINE_AA)
        x1, y1 = px(LAT1, 0.0)
        sub = img[TOP_Y0:int(y1), TOP_X0:int(x1)]
        ov = sub.copy()
        off = np.array([TOP_X0, TOP_Y0])
        for poly, colour in top.get("regions", []):
            pts = (np.array([px(a, b) for a, b in poly]) - off).astype(np.int32)
            cv2.fillPoly(ov, [pts], colour, cv2.LINE_AA)
        sub[:] = cv2.addWeighted(ov, 0.25, sub, 0.75, 0)
        for tr in top.get("tracks", []):
            if tr.get("trail") is not None and len(tr["trail"]) > 1:
                pts = (np.array([px(a, b) for a, b in tr["trail"]]) - off).astype(np.int32)
                cv2.polylines(sub, [pts], False, tr["color"], 2, cv2.LINE_AA)
        for lat, rng, colour in top.get("dets", []):
            p = (np.array(px(lat, rng)) - off).astype(int)
            cv2.circle(sub, tuple(p), 4, colour if colour is not None else (110, 110, 110), -1, cv2.LINE_AA)
        tlabels = []
        for tr in top.get("tracks", []):
            p = (np.array(px(tr["lat"], tr["rng"])) - off).astype(int)
            ring = tr.get("color2") if tr.get("color2") is not None else (255, 255, 255)
            cv2.circle(sub, tuple(p), 8, ring, -1, cv2.LINE_AA)
            cv2.circle(sub, tuple(p), 5, tr["color"], -1, cv2.LINE_AA)
            if tr.get("tags"):
                tlabels.append((tr["tags"], (TOP_X0 + p[0] + 11, TOP_Y0 + p[1] - 10)))
        x, y, w, h = CROP
        if crop_rect is not None:
            u0, v0, u1, v1 = crop_rect
            side = max(u1 - u0, v1 - v0, 8)
            cu, cv_ = (u0 + u1) / 2, (v0 + v1) / 2
            xs = np.clip((cu - side / 2 + np.arange(w) * side / w).astype(int), 0, wc - 1)
            ys = np.clip((cv_ - side / 2 + np.arange(h) * side / h).astype(int), 0, hc - 1)
            img[y:y + h, x:x + w] = rgb[ys][:, xs]
        else:
            img[y:y + h, x:x + w] = 236
        pil = Image.fromarray(img)
        dr = ImageDraw.Draw(pil)
        f = self.fonts
        placed = []
        for tags, (lx, ly) in labels:
            ws = [dr.textlength(t, font=f.get(18)) + 8 for t, _ in tags]
            box = [lx, ly, lx + sum(ws) + 2 * (len(ws) - 1), ly + 24]
            while any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3] for q in placed):
                box[1], box[3] = box[1] + 26, box[3] + 26
            placed.append(box)
            x0 = box[0]
            for (t, c), wt in zip(tags, ws):
                dr.rectangle((x0, box[1], x0 + wt, box[3]), fill=c)
                f.text(dr, (x0 + 4, box[1] + 12), t, 18, fill=ink(c), anchor="lm")
                x0 += wt + 2
        x_end = px(LAT1, 0.0)[0]
        tplaced = []
        for tags, (lx, ly) in tlabels:
            text = " / ".join(t for t, _ in tags)
            wl = dr.textlength(text, font=f.get(17))
            if lx + wl > x_end:
                lx -= wl + 22
            box = [lx, ly - 10, lx + wl, ly + 10]
            while any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3] for q in tplaced):
                box[1], box[3], ly = box[1] + 20, box[3] + 20, ly + 20
            tplaced.append(box)
            for i, (t, c) in enumerate(tags):
                if i:
                    lx += f.text(dr, (lx, ly), " / ", 17, fill=(80, 80, 80), anchor="lm", halo=3)
                lx += f.text(dr, (lx, ly), t, 17, fill=c, anchor="lm", halo=3)
        f.text(dr, (x + w / 2, y - 12), "ball crop" if crop_rect is not None else "no crop (not sure)", 17, anchor="mm")
        for i, line in enumerate(lines[:3]):
            f.text(dr, (262, 530 + 62 * i), line, 25, anchor="lm")
        if banner:
            wb = dr.textlength(banner, font=f.get(24))
            bx, by = cx + cw / 2 - wb / 2, cy + 14
            dr.rectangle((bx - 10, by - 4, bx + wb + 10, by + 32), fill=COL["ball"])
            f.text(dr, (bx, by + 14), banner, 24, fill=(255, 255, 255), anchor="lm")
        return np.asarray(pil)


def selftest() -> bool:
    import time
    v = SceneView()
    rgb = np.full((720, 1280, 3), 200, np.uint8)
    ball = classifier_style({"classes": {"material": {"label": "dielectric"}, "motion": {"label": "rigid"}}})
    person = classifier_style({"classes": {"material": {"label": "waterlike"}, "motion": {"label": "articulated"}}})
    top = {"dets": [(0.0, 1.55, ball["color"]), (-0.8, 2.3, None)],
           "regions": [(wedge((1.5, -0.05), (1.6, 0.05)), ball["color"]), (wedge((2.2, -0.4), (2.5, -0.25)), person["color"])],
           "tracks": [dict(lat=0.0, rng=1.55, trail=np.array([(-0.1, 1.55), (0.1, 1.56)]), **ball),
                      dict(lat=-0.8, rng=2.35, trail=None, **person)]}
    t0 = time.perf_counter()
    for _ in range(20):
        img = v.frame(rgb, [dict(rect=(500, 100, 700, 300), thick=True, **ball), dict(rect=(100, 50, 400, 700), **person)],
                      (450, 60, 750, 340),
                      (600, 200, 70), top, ["$t$ = 1.0 s     search chirp, 10 frames/s",
                                            "3 detections (FDR level $α$ = 0.10)     objects: ball, person",
                                            "crop: 12 % of the image (3.1 bit)     read: ARTIFA"], "radar blind: chirp change")
    ms = 1e3 * (time.perf_counter() - t0) / 20
    ok = img.shape == (720, 1280, 3) and ms < 60
    print(f"  {'PASS' if ok else 'FAIL'}  frame {img.shape}, {ms:.1f} ms, Computer Modern: {v.fonts.cmu}")
    Image.fromarray(img).save(Path(__file__).resolve().parent.parent / "results" / "scene_view_selftest.png")
    return ok


if __name__ == "__main__":
    selftest()
