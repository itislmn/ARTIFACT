"""
make_video.py - the reading video of a recorded text take: the camera video with the live radar overlays and a side
panel that animates both readers on that recording (radar top view, MetricSight steps, the CNN's feature maps).

    python make_video.py recordings/CV1/text_085c          (-> output/videos/text_085c_reading.mp4)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from backend import pipeline  # noqa: E402
from backend.imgio import ffmpeg_exe, read_video  # noqa: E402
from backend.settings import VIDEOS, load_config  # noqa: E402
from metrics import cer  # noqa: E402
from readers import cnn_reader  # noqa: E402
from readers.metricsight import ALPHABET  # noqa: E402

UI, LIGHT, RADAR, OK, BAD, WHITE = "#B0B0B0", "#FFD400", "#FF3B30", "#4CD964", "#FF3B30", "#F2F2F2"
PANEL_BG = (0.0, 0.0, 0.0, 1.0)
MONO = "DejaVu Sans Mono"
PHASES = {"radar": "radar: infini-rex detections, top view", "new": "reader 1: MetricSight",
          "cnn": "reader 2: cnn (lenet-5)", "result": "summary"}


class Processor:

    def __init__(self, objects, proj, cfg, truth):
        self.objects, self.proj, self.cfg, self.truth = objects, proj, cfg, truth
        self.res, self.rows, self.rd, self.cr = None, [], None, None

    def __call__(self, rgb):
        cfg = self.cfg
        self.res = pipeline.analyse(rgb, self.objects, self.proj, cfg, "text", keep_debug=True)
        self.rd = self.cr = None
        if self.res.readings:
            i = pipeline.text_object(self.res.readings)
            self.rd = self.res.readings[i]
            o = self.objects[i]
            c = self.proj.point(o["range_m"], o["azimuth_deg"], cfg["text"]["centre_z_m"])
            self.cr = cnn_reader.read(rgb, c, o["range_m"], self.proj.f, cfg, keep_debug=True)
        row = {}
        for name, r in (("new", self.rd), ("cnn", self.cr)):
            if r is not None:
                row[f"{name}_cer"], row[f"{name}_ms"] = 100 * cer(self.truth, r.text), r.ms
        self.rows.append(row)

    def accuracy(self, name):
        v = [r[f"{name}_cer"] for r in self.rows if f"{name}_cer" in r]
        return (float(np.mean(v)), len(v)) if v else (np.nan, 0)


def frames_looped(path: Path, n_out: int):
    k = 0
    while True:
        got = False
        for f in read_video(path):
            got = True
            yield f
            k += 1
            if k >= n_out:
                return
        if not got:
            raise SystemExit(f"no frames in {path}")


def _outline():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2, foreground="black")]


def hud_text(fig, x, y, s, size=9, color=WHITE, weight="normal", **kw):
    return fig.text(x, y, s, fontsize=min(size, 11), color=color, family=MONO, weight="normal",
                    path_effects=_outline(), **kw)


def panel_axes(fig, rect, title=None):
    ax = fig.add_axes(rect)
    ax.set_facecolor((0, 0, 0, 0))
    ax.tick_params(colors=UI, labelsize=7)
    for s in ax.spines.values():
        s.set_color(UI)
        s.set_linewidth(0.6)
    if title:
        ax.set_title(title.lower(), fontsize=7, color=UI, family=MONO, loc="left")
    return ax


def draw_video_overlays(ax, P, phase, t):
    res, objs = P.res, P.objects
    for i, (o, bar) in enumerate(zip(objs, res.bars)):
        if bar is None:
            continue
        n = len(bar)
        grow = min(1.0, 2 * t) if phase == "radar" else 1.0
        k = max(1, int(round(grow * n / 2)))
        part = bar[max(n // 2 - k, 0):n // 2 + k + 1]
        ax.plot(part[:, 0], part[:, 1], color=RADAR, lw=3, solid_capstyle="butt")
        seen = bar[(bar[:, 1] >= 0) & (bar[:, 1] < P.proj.size_wh[1])]
        u1, v1 = seen[np.argmin(seen[:, 1])] if len(seen) else bar[-1]
        v_lab = v1 + 14 if i % 2 == 0 else v1 + 34
        left = u1 > 0.655 * P.proj.size_wh[0] - 230
        ax.text(u1 - 6 if left else u1 + 6, v_lab, f"obj{i + 1} {o['range_m']:.2f}m {o['azimuth_deg']:+.1f}deg",
                ha="right" if left else "left", color=RADAR, fontsize=7, family=MONO, path_effects=_outline())
    rd = P.rd if phase != "cnn" else P.cr
    if rd is not None and rd.line_box is not None and phase in ("new", "cnn", "result"):
        import matplotlib.patches as mp
        x0, y0, x1, y1 = rd.line_box
        c = 0.15 * (y1 - y0)
        for (cx, cy, dx, dy) in ((x0 - c, y0 - c, 1, 1), (x1 + c, y0 - c, -1, 1), (x0 - c, y1 + c, 1, -1),
                                 (x1 + c, y1 + c, -1, -1)):
            ax.plot([cx, cx + dx * 2 * c], [cy, cy], color=UI, lw=2.5)
            ax.plot([cx, cx], [cy, cy + dy * 2 * c], color=UI, lw=2.5)
        cur = min(len(rd.boxes) - 1, int(t * len(rd.boxes))) if phase in ("new", "cnn") else -1
        for k, (a, b, cc, d) in enumerate(rd.boxes):
            ax.add_patch(mp.Rectangle((a, b), cc - a, d - b, fill=False, color=WHITE if k == cur else OK,
                                      lw=2 if k == cur else 1))


def readout(fig, P):
    hud_text(fig, 0.012, 0.115, "read" + (f"   (truth {P.truth})" if P.truth else ""), 8, UI)
    for row, (name, label, r) in enumerate((("new", "MetricSight", P.rd), ("cnn", "cnn", P.cr))):
        y = 0.075 - row * 0.032
        hud_text(fig, 0.012, y, label, 9, WHITE)
        if r is None:
            continue
        for k, ch in enumerate(r.text[:12]):
            col = (OK if (P.truth and k < len(P.truth) and ch == P.truth[k]) else BAD) if P.truth else WHITE
            hud_text(fig, 0.11 + 0.012 * k, y, ch, 11, col)
        acc, n = P.accuracy(name)
        hud_text(fig, 0.27, y, f"{r.ms:4.0f} ms" + (f"  cer {acc:3.0f}% so far" if n else ""), 8, UI)


def panel_radar(fig, P, t, radar_frames):
    ax = panel_axes(fig, [0.70, 0.40, 0.27, 0.40], "top view (x fwd, y right) [m]")
    k = int(t * 3 * len(radar_frames)) % len(radar_frames)
    ax.plot([0, 2.6 * np.sin(np.radians(-60))], [0, 2.6 * np.cos(np.radians(-60))], color=UI, lw=0.5, alpha=0.5)
    ax.plot([0, 2.6 * np.sin(np.radians(60))], [0, 2.6 * np.cos(np.radians(60))], color=UI, lw=0.5, alpha=0.5)
    d = radar_frames[k]
    if len(d):
        ax.plot(d[:, 0] * np.sin(np.radians(d[:, 1])), d[:, 0] * np.cos(np.radians(d[:, 1])), "o", color=RADAR, ms=6,
                alpha=0.85, label=f"radar frame {k + 1}/{len(radar_frames)}")
    for i, o in enumerate(P.objects):
        y, x = o["range_m"] * np.sin(np.radians(o["azimuth_deg"])), o["range_m"] * np.cos(np.radians(o["azimuth_deg"]))
        ax.plot(y, x, "s", mfc="none", color=WHITE, ms=11)
        ax.text(y + 0.06, x, f"obj{i + 1}", color=WHITE, fontsize=7, family=MONO)
    ax.plot(0, 0, "^", color=UI, ms=9)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.1, 2.6)
    ax.set_aspect("equal")
    ax.grid(color=UI, alpha=0.15)
    ax.legend(loc="upper left", fontsize=7, labelcolor=WHITE, frameon=False)
    hud_text(fig, 0.665, 0.33, f"doa: {P.cfg['radar']['method']} (fdr-controlled)", 9, UI)
    hud_text(fig, 0.665, 0.305, f"chirps/frame: {P.cfg['radar']['chirps']}", 9, UI)
    for i, o in enumerate(P.objects):
        hud_text(fig, 0.665, 0.27 - 0.026 * i, f"obj{i + 1}  r={o['range_m']:.3f}m  az={o['azimuth_deg']:+6.2f}deg  "
                 f"in {100 * o['rate']:.0f}% of frames", 9, WHITE)
    hud_text(fig, 0.665, 0.12, "", 8, UI)


def panel_new(fig, P, t):
    rd = P.rd
    if rd is None or not rd.debug:
        hud_text(fig, 0.668, 0.6, "no text line at the radar objects", 9, LIGHT)
        return
    d = rd.debug
    stage = min(3, int(t * 4))
    mm = P.cfg["text"]["mm_per_px"]
    imgs = [("1 region resampled to mm (radar range)", d["luma"], "gray", (0, 255)),
            ("2 local contrast", d["norm"], "gray", (-2.5, 2.5)),
            ("3 ink, line, cuts", d["ink"], "gray", (0, 1))]
    for k, (title, im, cm, lim) in enumerate(imgs[:stage + 1]):
        ax = panel_axes(fig, [0.668, 0.70 - 0.165 * k, 0.32, 0.13], title)
        ax.imshow(im, cmap=cm, vmin=lim[0], vmax=lim[1], aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 2:
            a, b = d["line"]
            ax.axhline(a, color=UI, lw=1)
            ax.axhline(b, color=UI, lw=1)
            for c0, c1 in d["cols"]:
                ax.axvspan(c0, c1, ymin=0, ymax=1, color=OK, alpha=0.18)
    a, b = d["line"]
    hud_text(fig, 0.668, 0.875, f"{mm} mm per pixel at every range | line height {(b - a) * mm:.0f} mm", 8, UI)
    if stage == 3 and len(d["cols"]):
        n = len(d["cols"])
        slot = min(n - 1, int((t - 0.75) / 0.25 * n))
        ax = panel_axes(fig, [0.668, 0.10, 0.07, 0.12], f"char {slot + 1}")
        ax.imshow(d["crops"][slot], cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = panel_axes(fig, [0.765, 0.10, 0.22, 0.17], "hog -> ridge scores")
        sc = d["scores"][slot]
        order = np.argsort(sc)[::-1][:8]
        ax.bar(range(8), sc[order], color=[OK] + [UI] * 7)
        ax.set_xticks(range(8), [ALPHABET[i] for i in order], fontsize=9, color=WHITE, family=MONO)
        hud_text(fig, 0.668, 0.035, "read: " + rd.text[:slot + 1], 11, OK)


def panel_cnn(fig, P, t):
    cr = P.cr
    if cr is None:
        return
    import matplotlib.patches as mp
    acts = cr.debug.get("acts") if cr.debug else None
    if not acts:
        hud_text(fig, 0.668, 0.6, "no characters in this frame", 9, LIGHT)
        return
    n = len(acts)
    slot = min(n - 1, int(t * n))
    ts = t * n - slot
    a = acts[slot]
    ax = fig.add_axes([0.665, 0.08, 0.325, 0.76])
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    layers = [("in\n32x32", [a["x"][0]]), ("conv6\n5x5", list(a["r1"])), ("pool", list(a["q1"])),
              ("conv12\n3x3", list(a["r2"])), ("pool", list(a["q2"])), ("fc\n64", None), ("out\n36", None)]
    xs = np.arange(len(layers)) + 0.5
    lit = ts * (len(layers) + 1)
    pos = []
    for li, (name, maps) in enumerate(layers):
        on = lit >= li
        col = WHITE if on else (0.4, 0.5, 0.6)
        ax.text(xs[li], 9.75, name, ha="center", va="top", color=col, fontsize=7, family=MONO)
        if maps is not None:
            m = len(maps)
            hh = min(0.8, 8.2 / m - 0.1)
            ys = np.linspace(8.3 - hh / 2, 0.3 + hh / 2, m) if m > 1 else [4.5]
            for mp_img, y in zip(maps, ys):
                ww = hh
                if on:
                    ax.imshow(mp_img, cmap="inferno" if li else "gray", extent=(xs[li] - ww / 2, xs[li] + ww / 2, y - hh / 2, y + hh / 2),
                              aspect="auto", interpolation="nearest")
                ax.add_patch(mp.Rectangle((xs[li] - ww / 2, y - hh / 2), ww, hh, fill=False, color=col, lw=0.5))
            pos.append(list(zip([xs[li]] * m, ys)))
        elif name.startswith("fc"):
            h = a["h"]
            ys = np.linspace(8.3, 0.4, 32)
            vals = h[::2] / (h.max() + 1e-9)
            ax.scatter([xs[li]] * 32, ys, s=14, c=vals if on else [0.2] * 32, cmap="viridis", vmin=0, vmax=1)
            pos.append(list(zip([xs[li]] * 32, ys)))
        else:
            pr = a["prob"]
            top = np.argsort(pr)[::-1][:6]
            ys = np.linspace(7.8, 2.0, 6)
            for k, (ci, y) in enumerate(zip(top, ys)):
                ax.add_patch(mp.Rectangle((xs[li] - 0.35, y - 0.3), 0.8 * pr[ci] if on else 0, 0.6, color=OK if k == 0 else UI))
                ax.text(xs[li] - 0.45, y, ALPHABET[ci], ha="right", va="center", color=WHITE, fontsize=9, family=MONO)
            pos.append(list(zip([xs[li]] * 6, ys)))
    rng = np.random.default_rng(slot)
    for li in range(len(layers) - 1):
        A, B = pos[li], pos[li + 1]
        pairs = [(A[i], B[j]) for i in rng.choice(len(A), min(len(A), 8), replace=False)
                 for j in rng.choice(len(B), min(len(B), 6), replace=False)]
        active = li <= lit < li + 1
        for (p0, p1) in pairs:
            ax.plot([p0[0] + 0.25, p1[0] - 0.25], [p0[1], p1[1]], color=UI, lw=0.4, alpha=0.6 if lit > li else 0.12)
            if active:
                f = lit - li
                ax.plot(p0[0] + 0.25 + f * (p1[0] - p0[0] - 0.5), p0[1] + f * (p1[1] - p0[1]), "o", color=LIGHT, ms=3)
    hud_text(fig, 0.668, 0.875, f"char {slot + 1}/{n}  forward pass  "
             f"P = {a['prob'].max():.2f}", 8, UI)
    hud_text(fig, 0.668, 0.03, "read: " + cr.text[:slot + (1 if lit >= len(layers) else 0)], 11, OK)


def panel_result(fig, P, t):
    rows = [("NEW MetricSight", "new", P.rd), ("CNN LeNet-5", "cnn", P.cr)]
    hud_text(fig, 0.668, 0.80, "reader            read       frames  cer       ms", 8.5, UI)
    for k, (label, name, r) in enumerate(rows):
        acc, n = P.accuracy(name)
        ms = [x[f"{name}_ms"] for x in P.rows if f"{name}_ms" in x]
        hud_text(fig, 0.668, 0.76 - 0.04 * k, f"{label:<17} {(r.text[:10] if r else '-'):<10} {n:6d} {acc:6.1f}% "
                 f"{(np.median(ms) if ms else np.nan):5.0f}", 8.5, WHITE)
    ax = panel_axes(fig, [0.70, 0.30, 0.27, 0.28], "character error rate, all processed frames [%]")
    accs = [P.accuracy(n)[0] for _, n, _ in rows]
    ax.bar(range(2), np.nan_to_num(accs), color=[OK, UI])
    ax.set_xticks(range(2), ["MetricSight", "CNN"], color=WHITE, family=MONO)
    ax.set_ylim(0, max(105, np.nanmax(np.nan_to_num(accs)) + 5))
    hud_text(fig, 0.668, 0.17, "Same radar region, same character cutting.\n"
             "MetricSight: metric image + censored ink + ridge.\nCNN: camera pixels + Sauvola + LeNet-5.\n"
             "Live: only MetricSight (CPU shared with the radar).", 8, UI)


def render(video: Path, objects, radar_frames, proj, cfg, truth, phase_s, out: Path, fps=30.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    plt.rcParams.update({"animation.ffmpeg_path": ffmpeg_exe(), "font.family": MONO})
    phases = ["radar", "new", "cnn", "result"]
    n_frames = int(len(phases) * phase_s * fps)
    P = Processor(objects, proj, cfg, truth)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = anim.FFMpegWriter(fps=fps, codec="h264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"])
    t_start = time.perf_counter()
    with writer.saving(fig, str(out), dpi=100):
        for k, rgb in enumerate(frames_looped(video, n_frames)):
            P(rgb)
            pi = min(len(phases) - 1, int(k / (phase_s * fps)))
            phase, t = phases[pi], (k / (phase_s * fps)) - pi
            fig.clf()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.set_zorder(-10)
            ax.imshow(rgb, extent=(0, rgb.shape[1], rgb.shape[0], 0))
            ax.set_xlim(0, rgb.shape[1])
            ax.set_ylim(rgb.shape[0], 0)
            draw_video_overlays(ax, P, phase, t)
            fig.patches.append(mp.Rectangle((0.655, 0.0), 0.345, 1.0, transform=fig.transFigure, color=PANEL_BG, zorder=-5))
            hud_text(fig, 0.012, 0.972, f"{video.parent.name}  t={k / fps:06.2f}s  frame {k:05d}  "
                     f"cam iw-1000-1 1280x720  radar awr2944 infini-rex", 8, WHITE)
            hud_text(fig, 0.665, 0.972, PHASES[phase], 8, UI)
            {"radar": lambda: panel_radar(fig, P, t, radar_frames), "new": lambda: panel_new(fig, P, t),
             "cnn": lambda: panel_cnn(fig, P, t), "result": lambda: panel_result(fig, P, t)}[phase]()
            readout(fig, P)
            writer.grab_frame()
            if k % 60 == 0:
                print(f"  frame {k}/{n_frames}  ({time.perf_counter() - t_start:.0f} s)", flush=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path, help="text take folder recordings/<session>/<take>")
    ap.add_argument("--phase-s", type=float, default=3.5, help="seconds per phase")
    args = ap.parse_args()
    cfg = load_config()
    from backend.alignment import Projector
    from backend.radar_link import live_log
    summary = json.loads((args.take / "summary.json").read_text())
    objects, radar_frames, _ = live_log(args.take, cfg["radar"])
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out = VIDEOS / f"{args.take.name}_reading.mp4"
    render(args.take / "camera.mp4", objects, radar_frames, Projector(), cfg, summary.get("truth", ""), args.phase_s, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
