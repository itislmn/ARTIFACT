"""
view.py - the live picture, the same on screen and in the recorded video, in tracking's style: the camera with each
object's region box (frame = material, inner frame = motion, two tags) and where each measured object is, the scene
from above with the detections, every track's prediction region and path in its material colour, the measured
objects (dashed), the empty path, the rail's path, what the planner waits for or steps around, and the colour key.

    from backend.view import ObstacleView
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


W, H = 1280, 720
CAM = (6, 8, 800, 450)
CROP = (6, 494, 220, 220)
TOP_X0, TOP_Y0, K = 872, 46, 150.0
LAT0, LAT1, FWD0, FWD1 = -1.2, 1.2, -0.3, 2.6
OKABE = {"black": (0, 0, 0), "orange": (230, 159, 0), "sky": (86, 180, 233), "green": (0, 158, 115),
         "blue": (0, 114, 178), "vermillion": (213, 94, 0), "purple": (204, 121, 167), "grey": (127, 127, 127)}
MATERIAL_COL = {"metal": (0, 114, 178), "dielectric": (230, 159, 0), "waterlike": (0, 158, 115), "none": (153, 153, 153)}
MOTION_COL = {"static": (0, 0, 0), "rigid": (213, 94, 0), "articulated": (204, 121, 167), "distributed": (86, 180, 233),
              "none": (153, 153, 153)}
STATE_COL = {"wait": OKABE["vermillion"], "blocked": OKABE["black"], "avoid": OKABE["blue"], "verify": OKABE["purple"],
             "return": OKABE["orange"], "approach": OKABE["sky"], "cruise": OKABE["green"], "done": OKABE["black"],
             "idle": OKABE["grey"], "decide": OKABE["sky"], "observe": OKABE["grey"]}
FOCUS = OKABE["vermillion"]
NORM = (34, 34, 34)


def dashed(img, a, b, col, t: int = 1, on: int = 7, off: int = 5):
    (x0, y0), (x1, y1) = a, b
    for p, q in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        for s0 in np.arange(0.0, n, on + off):
            s1 = min(s0 + on, n)
            c0 = (int(p[0] + (q[0] - p[0]) * s0 / max(n, 1)), int(p[1] + (q[1] - p[1]) * s0 / max(n, 1)))
            c1 = (int(p[0] + (q[0] - p[0]) * s1 / max(n, 1)), int(p[1] + (q[1] - p[1]) * s1 / max(n, 1)))
            cv2.line(img, c0, c1, col, t, cv2.LINE_AA)


def px(lat: float, fwd: float):
    return TOP_X0 + (lat - LAT0) * K, TOP_Y0 + (FWD1 - fwd) * K


def ink(rgb) -> tuple:
    return (0, 0, 0) if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 160 else (255, 255, 255)


def style(o: dict) -> dict:
    if o.get("known"):
        g = (150, 150, 150)
        return {"color": g, "color2": None, "tags": [("room", g)]}
    if "classes" in o:
        c = o.get("classes") or {}
        o = {"material": (c.get("material") or {}).get("label"), "motion": (c.get("motion") or {}).get("label")}
    mat, mot = o.get("material"), o.get("motion")
    if not mat and not mot:
        g = MATERIAL_COL["none"]
        return {"color": g, "color2": None, "tags": [("not yet classified", g)]}
    cm, cn = MATERIAL_COL.get(mat or "none", MATERIAL_COL["none"]), MOTION_COL.get(mot or "none", MOTION_COL["none"])
    return {"color": cm, "color2": cn, "tags": [(mat or "?", cm), (mot or "?", cn)]}


def plates_width(dr, fonts, tags, size: int) -> float:
    return sum(dr.textlength(t, font=fonts.get(size)) + 10 for t, _ in tags) - 2


def plates(dr, fonts, xy, tags, size: int) -> list:
    x, y = xy
    x0 = x
    for t, c in tags:
        w = dr.textlength(t, font=fonts.get(size)) + 8
        dr.rectangle((x, y, x + w, y + size + 7), fill=c)
        fonts.text(dr, (x + 4, y + (size + 7) / 2), t, size, fill=ink(c), anchor="lm")
        x += w + 2
    return [x0, y, x - 2, y + size + 7]


def free(box, placed, step: int, limit: float) -> list:
    box = list(box)
    while box[3] < limit and any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3] for q in placed):
        box[1], box[3] = box[1] + step, box[3] + step
    return box


class ObstacleView:
    def __init__(self, cfg: dict, platform):
        self.fonts = Fonts()
        self.pf = platform
        self.goal = float(platform.limits[0, 1])
        self.r_max = float(cfg["radar"]["r_max_m"])
        self.clear = float(cfg["planner"]["clearance_m"])
        self.bg = self._background()

    def _background(self) -> np.ndarray:
        img = np.full((H, W, 3), 255, np.uint8)
        x1, y1 = px(LAT1, FWD0)
        for f in np.arange(0.0, FWD1 + 1e-9, 0.5):
            cv2.line(img, tuple(map(int, px(LAT0, f))), tuple(map(int, px(LAT1, f))), (232, 232, 232), 1)
        for lat in np.arange(-1.0, 1.0 + 1e-9, 0.5):
            cv2.line(img, tuple(map(int, px(lat, FWD0))), tuple(map(int, px(lat, FWD1))), (232, 232, 232), 1)
        a, b = px(-0.004, 0.0), px(0.004, self.goal)
        cv2.rectangle(img, (int(a[0]), int(b[1])), (int(b[0]), int(a[1])), (190, 190, 190), -1)
        cv2.rectangle(img, (TOP_X0, TOP_Y0), (int(x1), int(y1)), (77, 77, 77), 1)
        pil = Image.fromarray(img)
        dr = ImageDraw.Draw(pil)
        f = self.fonts
        for lat in np.arange(-1.0, 1.0 + 1e-9, 0.5):
            f.text(dr, (px(lat, FWD0)[0], y1 + 6), f"{lat:.1f}".replace("-", "−"), 14, anchor="mt")
        for fw in np.arange(0.0, FWD1 + 1e-9, 0.5):
            w = dr.textlength(f"{fw:.1f}", font=f.get(14))
            f.text(dr, (TOP_X0 - 7 - w, px(LAT0, fw)[1]), f"{fw:.1f}", 14, anchor="lm")
        f.text(dr, ((TOP_X0 + x1) / 2, y1 + 24), "sideways $y$ (m)", 15, anchor="mt")
        f.text(dr, ((TOP_X0 + x1) / 2, TOP_Y0 - 28), "scene from above", 17, anchor="mt")
        lab = Image.new("RGB", (150, 24), "white")
        ImageDraw.Draw(lab).text((75, 12), "forward x (m)", font=f.get(15), fill=(0, 0, 0), anchor="mm")
        pil.paste(lab.rotate(90, expand=True), (TOP_X0 - 66, int((TOP_Y0 + y1) / 2) - 75))
        ly = int(y1) + 52
        f.text(dr, (TOP_X0 - 60, ly), "material", 14, anchor="lm")
        x = TOP_X0 + 12
        for name in ("metal", "dielectric", "waterlike"):
            x = plates(dr, f, (x, ly - 10), [(name, MATERIAL_COL[name])], 14)[2] + 6
        plates(dr, f, (x, ly - 10), [("not yet classified", MATERIAL_COL["none"])], 14)
        ly += 26
        f.text(dr, (TOP_X0 - 60, ly), "motion", 14, anchor="lm")
        x = TOP_X0 + 12
        for name in ("static", "rigid", "articulated", "distributed"):
            x = plates(dr, f, (x, ly - 10), [(name, MOTION_COL[name])], 14)[2] + 6
        ly += 30
        items = (("prediction region (90 %)", "reg"), ("track, last 3 s", "trk"), ("detection", "det"),
                 ("obstacle box", "box"), ("measured object", "meas"), ("empty path (reference)", "room"),
                 ("rail path", "rail"), ("platform", "head"), ("waiting for / avoiding", "focus"))
        for i, (s, kind) in enumerate(items):
            lx, yy = TOP_X0 - 52 + (i % 2) * 200, ly + (i // 2) * 23
            if kind == "reg":
                dr.polygon([(lx - 9, yy + 6), (lx - 3, yy - 7), (lx + 3, yy - 7), (lx + 9, yy + 6)], fill=(205, 222, 238))
            elif kind == "trk":
                dr.line((lx - 10, yy + 3, lx + 4, yy - 2), fill=(90, 90, 90), width=2)
                dr.ellipse((lx + 1, yy - 6, lx + 9, yy + 2), fill=(90, 90, 90))
            elif kind == "det":
                dr.ellipse((lx - 3, yy - 3, lx + 3, yy + 3), fill=(130, 130, 130))
            elif kind == "box":
                dr.rectangle((lx - 8, yy - 6, lx + 8, yy + 6), outline=MATERIAL_COL["metal"], width=2)
            elif kind == "meas":
                for x0 in range(lx - 8, lx + 8, 6):
                    dr.line((x0, yy - 6, x0 + 3, yy - 6), fill=NORM)
                    dr.line((x0, yy + 6, x0 + 3, yy + 6), fill=NORM)
                dr.line((lx - 8, yy - 3, lx - 8, yy + 1), fill=NORM)
                dr.line((lx + 8, yy - 3, lx + 8, yy + 1), fill=NORM)
            elif kind == "room":
                dr.rectangle((lx - 8, yy - 6, lx + 8, yy + 6), fill=(215, 215, 215))
            elif kind == "rail":
                dr.line((lx - 10, yy, lx + 10, yy), fill=(60, 60, 60), width=2)
            elif kind == "focus":
                dr.ellipse((lx - 8, yy - 8, lx + 8, yy + 8), outline=FOCUS, width=3)
            else:
                dr.rectangle((lx - 8, yy - 6, lx + 8, yy + 6), fill=(60, 60, 60))
            f.text(dr, (lx + 14, yy), s, 14, anchor="lm")
        return np.asarray(pil).copy()

    def frame(self, rgb, cam_boxes, crop_rect, light_uv, scene: dict, lines, banner=None, banner_col=None,
              crop_title="attention", light_tag=None, light_r=0.0, measured=()) -> np.ndarray:
        img = self.bg.copy()
        hc, wc = rgb.shape[:2]
        cx, cy, cw, ch = CAM
        s = cw / wc
        img[cy:cy + ch, cx:cx + cw] = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
        cam = img[cy:cy + ch, cx:cx + cw]
        labels, names = [], []
        for name, (u0, v0, u1, v1) in measured:
            c = (int((u0 + u1) / 2 * s), int((v0 + v1) / 2 * s))
            if 0 <= c[0] < cw and 0 <= c[1] < ch:
                cv2.circle(cam, c, 6, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(cam, c, 6, NORM, 1, cv2.LINE_AA)
                names.append((name, (cx + c[0] + 10, cy + c[1])))
        for box in cam_boxes:
            u0, v0, u1, v1 = box["rect"]
            if not (0 <= (u0 + u1) / 2 < wc and 0 <= (v0 + v1) / 2 < hc):
                continue
            a = (int(np.clip(u0 * s, -10, cw + 10)), int(np.clip(v0 * s, -10, ch + 10)))
            b = (int(np.clip(u1 * s, -10, cw + 10)), int(np.clip(v1 * s, -10, ch + 10)))
            w1 = 3 if box.get("thick") else 2
            if box.get("focus"):
                cv2.rectangle(cam, (a[0] - 4, a[1] - 4), (b[0] + 4, b[1] + 4), FOCUS, 3, cv2.LINE_AA)
            cv2.rectangle(cam, a, b, box["color"], w1, cv2.LINE_AA)
            if box.get("color2") is not None:
                cv2.rectangle(cam, (a[0] + w1, a[1] + w1), (b[0] - w1, b[1] - w1), box["color2"], 2, cv2.LINE_AA)
            labels.append((box["tags"], (cx + max(a[0], 0) + 3, cy + max(a[1], 0) + 3)))
        if light_uv is not None:
            c, r = (int(light_uv[0] * s), int(light_uv[1] * s)), int(max(light_r * s, 12))
            for a in range(0, 360, 30):
                cv2.ellipse(cam, c, (r, r), 0, a, a + 18, OKABE["orange"], 2, cv2.LINE_AA)
            cv2.drawMarker(cam, c, OKABE["orange"], cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            labels.append(([("flash (camera)", OKABE["orange"])], (cx + c[0] + r + 4, cy + c[1] - 11)))
        tlabels, focus_label = self._top(img, scene)
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
        nplaced = []
        for name, (lx, ly) in names:
            wt = dr.textlength(name, font=f.get(13))
            lx = min(lx, cx + cw - wt - 2)
            box = free([lx, ly - 8, lx + wt, ly + 8], nplaced, 17, cy + ch)
            nplaced.append(box)
            f.text(dr, (box[0], box[1] + 8), name, 13, fill=NORM, anchor="lm", halo=2)
        placed = [(cx, cy, cx + cw, cy + 50)] if banner else []
        for tags, (lx, ly) in labels:
            wt = plates_width(dr, f, tags, 15)
            lx = min(lx, cx + cw - wt - 2)
            box = free([lx, ly, lx + wt, ly + 22], placed, 24, cy + ch)
            if box[3] <= cy + ch:
                placed.append(plates(dr, f, box[:2], tags, 15))
        x_end, y_end, tplaced = px(LAT1, 0.0)[0], px(LAT0, FWD0)[1], []
        if focus_label is not None:
            text, (lx, ly) = focus_label
            wt = dr.textlength(text, font=f.get(15)) + 10
            lx = min(max(lx - wt / 2, TOP_X0 + 2), x_end - wt - 2)
            tplaced.append(plates(dr, f, (lx, max(ly, TOP_Y0 + 2)), [(text, FOCUS)], 15))
        for tags, (lx, ly) in tlabels:
            wt = plates_width(dr, f, tags, 13)
            if lx + wt > x_end:
                lx -= wt + 22
            box = free([lx, max(ly, TOP_Y0 + 2), lx + wt, max(ly, TOP_Y0 + 2) + 20], tplaced, 21, y_end)
            if box[3] <= y_end:
                tplaced.append(plates(dr, f, box[:2], tags, 13))
        f.text(dr, (x, y - 12), crop_title, 17, anchor="lm")
        for i, line in enumerate(lines[:4]):
            while line and dr.textlength(line.replace("$", ""), font=f.get(17)) > TOP_X0 - 70 - 244:
                line = line[:-2].rstrip() + "…"
            f.text(dr, (244, 516 + 54 * i), line, 17, anchor="lm")
        if light_tag:
            tag = [(light_tag, OKABE["orange"])]
            plates(dr, f, (cx + cw - plates_width(dr, f, tag, 18) - 8, cy + ch - 34), tag, 18)
        if banner:
            while dr.textlength(banner, font=f.get(24)) > cw - 40:
                banner = banner[:-2].rstrip() + "…"
            wb = dr.textlength(banner, font=f.get(24))
            bx, by = cx + cw / 2 - wb / 2, cy + 14
            dr.rectangle((bx - 10, by - 4, bx + wb + 10, by + 32), fill=banner_col or FOCUS)
            f.text(dr, (bx, by + 14), banner, 24, fill=(255, 255, 255), anchor="lm")
        return np.asarray(pil)

    def _top(self, img, scene: dict):
        x1, y1 = px(LAT1, FWD0)
        sub = img[TOP_Y0:int(y1), TOP_X0:int(x1)]
        off = np.array([TOP_X0, TOP_Y0])

        def P(lat, fwd):
            return tuple((np.array(px(lat, fwd)) - off).astype(int))

        def rect(lo, hi, col, fill=None, t=2, alpha=0.3):
            a, b = P(max(lo[1], LAT0), min(hi[0], FWD1)), P(min(hi[1], LAT1), max(lo[0], FWD0))
            if fill is not None:
                ov = sub.copy()
                cv2.rectangle(ov, a, b, fill, -1)
                sub[:] = cv2.addWeighted(ov, alpha, sub, 1 - alpha, 0)
            cv2.rectangle(sub, a, b, col, t, cv2.LINE_AA)

        pose = np.asarray(scene.get("pose", [0, 0, 0]), float)
        plan = scene.get("plan", {})
        for i in range(-1, 2, 2):
            for d in np.arange(0.1, self.r_max, 0.08):
                a = np.radians(60) * i
                p0, p1 = (pose[1] + d * np.sin(a), pose[0] + d * np.cos(a)), (pose[1] + (d + 0.035) * np.sin(a),
                                                                              pose[0] + (d + 0.035) * np.cos(a))
                if FWD0 < p0[1] < FWD1 and LAT0 < p0[0] < LAT1:
                    cv2.line(sub, P(*p0), P(*p1), (165, 165, 165), 1, cv2.LINE_AA)
        for rr in (1.0, 2.0):
            arc = np.array([P(pose[1] + rr * np.sin(a), pose[0] + rr * np.cos(a)) for a in np.radians(np.linspace(-60, 60, 49))])
            cv2.polylines(sub, [arc.astype(np.int32)], False, (222, 222, 222), 1, cv2.LINE_AA)
        x_end = min(plan.get("x_free", self.goal), self.goal)
        hw = self.pf.head[1]
        for yy in (pose[1] + hw[0] - self.clear, pose[1] + hw[1] + self.clear):
            for d in np.arange(pose[0] + self.pf.head[0, 1], x_end + self.pf.head[0, 1], 0.05):
                cv2.line(sub, P(yy, d), P(yy, min(d + 0.025, x_end + self.pf.head[0, 1])), (0, 158, 115), 1, cv2.LINE_AA)
        if plan.get("limit_obj") is not None and x_end < self.goal - 0.01:
            cv2.line(sub, P(pose[1] + hw[0] - self.clear, x_end + self.pf.head[0, 1]),
                     P(pose[1] + hw[1] + self.clear, x_end + self.pf.head[0, 1]), FOCUS, 3, cv2.LINE_AA)
        for o in scene.get("objects", []):
            lo, hi = np.asarray(o["lo"], float), np.asarray(o["hi"], float)
            if o.get("known"):
                rect(lo, hi, (215, 215, 215), fill=(215, 215, 215), t=1, alpha=0.6)
            elif o["kind"] == "static":
                rect(lo, hi, style(o)["color"], t=2)
        for lo, hi in scene.get("measured", []):
            a, b = P(max(lo[1], LAT0), min(hi[0], FWD1)), P(min(hi[1], LAT1), max(lo[0], FWD0))
            dashed(sub, (a[0] - 2, a[1] - 2), (b[0] + 2, b[1] + 2), NORM, 1, 4, 3)
        tracks = [q for q in scene.get("tracks", []) if q["confirmed"]]
        ov = sub.copy()
        for tr in tracks:
            if tr.get("poly") is not None:
                cv2.fillPoly(ov, [np.array([P(a, b) for a, b in tr["poly"]], np.int32)], tr["color"], cv2.LINE_AA)
        sub[:] = cv2.addWeighted(ov, 0.18, sub, 0.82, 0)
        col = {tr["id"]: tr["color"] for tr in tracks}
        for y, x, tid in scene.get("det_hist", []):
            if tid in col and FWD0 < x < FWD1 and LAT0 < y < LAT1:
                cv2.circle(sub, P(y, x), 2, col[tid], -1, cv2.LINE_AA)
        for d in scene.get("dets", []):
            w = d.get("world")
            if w is None or not (FWD0 < w[0] < FWD1 and LAT0 < w[1] < LAT1):
                continue
            if col.get(d.get("track")) is not None and d.get("label") in ("core", "part", "birth"):
                cv2.circle(sub, P(w[1], w[0]), 4, col[d["track"]], -1, cv2.LINE_AA)
                cv2.circle(sub, P(w[1], w[0]), 4, (255, 255, 255), 1, cv2.LINE_AA)
            elif d.get("label") == "ghost":
                cv2.circle(sub, P(w[1], w[0]), 3, (175, 175, 175), 1, cv2.LINE_AA)
            else:
                cv2.circle(sub, P(w[1], w[0]), 2, (130, 130, 130), -1, cv2.LINE_AA)
        for tr in tracks:
            if tr.get("trail") is not None and len(tr["trail"]) > 1:
                cv2.polylines(sub, [np.array([P(a, b) for a, b in tr["trail"]], np.int32)], False, tr["color"], 2, cv2.LINE_AA)
        for tr in scene.get("tracks", []):
            if not tr["confirmed"]:
                cv2.circle(sub, P(tr["lat"], tr["fwd"]), 3, (170, 170, 170), -1, cv2.LINE_AA)
        tlabels = []
        for tr in sorted(tracks, key=lambda q: q["dist"]):
            p = P(tr["lat"], tr["fwd"])
            cv2.circle(sub, p, 9, tr.get("color2") or (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(sub, p, 6, tr["color"], -1, cv2.LINE_AA)
            if 0 <= p[0] < sub.shape[1] and 0 <= p[1] < sub.shape[0]:
                tlabels.append((tr["tags"], (TOP_X0 + p[0] + 12, TOP_Y0 + p[1] - 10)))
        trail = scene.get("trail")
        if trail is not None and len(trail) > 1:
            cv2.polylines(sub, [np.array([P(p[1], p[0]) for p in trail], np.int32)], False, (60, 60, 60), 2, cv2.LINE_AA)
        car_lo = self.pf.carriage[:, 0] + np.r_[pose[0], 0, 0]
        car_hi = self.pf.carriage[:, 1] + np.r_[pose[0], 0, 0]
        rect(car_lo, car_hi, (110, 110, 110), t=1)
        lo, hi = self.pf.head_box(pose)
        a, b = P(lo[1], hi[0]), P(hi[1], lo[0])
        cv2.rectangle(sub, a, b, (60, 60, 60), -1)
        if plan.get("target") is not None and np.hypot(plan["target"][1] - pose[1], plan["target"][2] - pose[2]) > 0.003:
            cv2.arrowedLine(sub, P(pose[1], pose[0] + self.pf.head[0, 1]),
                            P(plan["target"][1], pose[0] + self.pf.head[0, 1]), OKABE["blue"], 3, cv2.LINE_AA, tipLength=0.4)
        focus_label = None
        fo = scene.get("focus")
        if fo is not None:
            p = P(np.clip(fo["lat"], LAT0, LAT1), np.clip(fo["fwd"], FWD0, FWD1))
            cv2.circle(sub, p, 18, FOCUS, 3, cv2.LINE_AA)
            focus_label = (fo["text"], (TOP_X0 + p[0], TOP_Y0 + p[1] - 44))
        return tlabels, focus_label
