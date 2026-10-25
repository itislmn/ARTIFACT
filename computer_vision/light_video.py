"""
light_video.py - the light experiment as a video: the recording played twice (raw frame, then after Radar-Anchored Glare
Removal) with the live radar, the lamp's radar and camera position and both readers, from output/predictions.csv.

    python light_video.py recordings/CV1/light_3          (after python run_all.py; -> output/videos/)
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

import dataset  # noqa: E402
from backend import glare as gl  # noqa: E402
from backend import pipeline  # noqa: E402
from backend.imgio import ffmpeg_exe  # noqa: E402
from backend.settings import VIDEOS, load_config  # noqa: E402
from evaluate import PREDICTIONS  # noqa: E402
from make_video import BAD, LIGHT, MONO, OK, PANEL_BG, RADAR, UI, WHITE, _outline, hud_text, panel_axes  # noqa: E402


def records(name: str) -> list[dict]:
    rows = [r for r in dataset.read(dataset.TABLE) if r["take"] == name]
    preds = {}
    for p in dataset.read(PREDICTIONS):
        if p["take"] == name:
            preds[int(p["k"]), "new" if p["reader"] == "MetricSight" else "cnn", "raw" if p["condition"] == "raw" else "filt"] = p
    recs = []
    for r in rows:
        k = int(r["k"])
        rec = {"k": k, "t": r["t_s"], "light": int(r["lamp"]), "objects": json.loads(r["objects"]),
               "dets": np.array(json.loads(r["dets"]), float).reshape(-1, 2),
               "source": (r["source_range_m"], r["source_azimuth_deg"]) if r["source_range_m"] != "" else None,
               "glare": ((r["glare_u"], r["glare_v"]), r["glare_core_px"], json.loads(r["glare_fit"])) if r["glare_u"] != "" else None}
        for reader in ("new", "cnn"):
            for cond in ("raw", "filt"):
                p = preds.get((k, reader, cond))
                if p is not None:
                    rec[f"{reader}_{cond}"] = "" if p["text"] == "" else str(p["text"])
                    rec[f"{reader}_{cond}_boxes"] = json.loads(p["boxes"])
                    rec[f"{reader}_{cond}_cer"] = 100 * p["cer"]
        recs.append(rec)
    return recs


def cer(recs, key, light):
    v = [r[key] for r in recs if key in r and r["light"] == light and (not light or r.get("new_filt_cer") is not None)]
    return float(np.mean(v)) if v else float("nan")


def render(take: Path, recs, truth, proj, cfg, fps_out: float, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    plt.rcParams.update({"animation.ffmpeg_path": ffmpeg_exe(), "font.family": MONO})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = anim.FFMpegWriter(fps=fps_out, codec="h264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"])
    n = len(recs)
    t_start = time.perf_counter()
    with writer.saving(fig, str(out), dpi=100):
        for pi, cond in enumerate(("raw", "filt")):
            seen = []
            for j, (rec, rgb) in enumerate(zip(recs, frames_at(take, [r["k"] for r in recs]))):
                seen.append(rec)
                fig.clf()
                ax = fig.add_axes([0, 0, 1, 1])
                ax.set_axis_off()
                ax.set_zorder(-10)
                img = rgb
                if cond == "filt" and rec["glare"] is not None:
                    g = gl.Glare(-1, rec["glare"][0], rec["glare"][1], rec["glare"][2], 0.0)
                    img = gl.clean(rgb, g, pipeline.fast_luma(rgb) >= cfg["camera"]["clip_level"])
                ax.imshow(img, extent=(0, rgb.shape[1], rgb.shape[0], 0))
                ax.set_xlim(0, rgb.shape[1])
                ax.set_ylim(rgb.shape[0], 0)
                for i, o in enumerate(rec["objects"]):
                    bar = proj.bar(o["range_m"], o["azimuth_deg"], cfg["alignment"]["bar_z_m"])
                    if bar is None:
                        continue
                    src = rec["source"] and abs(o["range_m"] - rec["source"][0]) < 0.05
                    ax.plot(bar[:, 0], bar[:, 1], color=LIGHT if src else RADAR, lw=2.5)
                    vis = bar[(bar[:, 1] >= 0) & (bar[:, 1] < rgb.shape[0])]
                    u1, v1 = vis[np.argmin(vis[:, 1])] if len(vis) else bar[-1]
                    ax.text(u1 + 5, v1 + 12, f"obj{i + 1} {o['range_m']:.2f}m {o['azimuth_deg']:+.1f}deg" + (" lamp" if src else ""),
                            color=LIGHT if src else RADAR, fontsize=7, family=MONO, path_effects=_outline())
                if rec["glare"] is not None:
                    (gu, gv), core, _ = rec["glare"]
                    ax.plot(gu, gv, "+", color=LIGHT, ms=18, mew=2)
                    ax.text(gu + 10, gv - 10, "flash (camera)", color=LIGHT, fontsize=7, family=MONO, path_effects=_outline())
                    if rec["source"]:
                        r_s, az_s = rec["source"]
                        bu = proj.u_at_row(r_s, az_s, gv, cfg["alignment"]["bar_z_m"])
                        ax.plot([bu, gu], [gv, gv], color=LIGHT, lw=1, ls=":")
                        ax.text((bu + gu) / 2, gv + 18, f"radar-camera {abs(gu - bu) * r_s / proj.f * 100:.1f} cm",
                                color=LIGHT, fontsize=7, family=MONO, ha="center", path_effects=_outline())
                    if cond == "filt":
                        ax.add_patch(mp.Circle((gu, gv), max(core, 5), fill=False, color=LIGHT, lw=1, ls="--"))
                for (a, b, c2, d) in rec.get(f"new_{cond}_boxes", []):
                    ax.add_patch(mp.Rectangle((a, b), c2 - a, d - b, fill=False, color=OK, lw=1))
                fig.patches.append(mp.Rectangle((0.655, 0.0), 0.345, 1.0, transform=fig.transFigure, color=PANEL_BG, zorder=-5))
                hud_text(fig, 0.012, 0.972, f"{take.name}  t={rec['t']:06.2f}s  frame {rec['k']:05d}  "
                         f"pass {pi + 1}/2: {'raw frame' if cond == 'raw' else 'radar-anchored glare removal (veil subtracted, clipped core cut)'}", 8, WHITE)
                hud_text(fig, 0.665, 0.972, "light experiment: raw vs. glare-removed reading", 8, UI)
                axr = panel_axes(fig, [0.70, 0.60, 0.27, 0.33], "radar top view (x fwd, y right) [m]")
                d = rec["dets"]
                if len(d):
                    axr.plot(d[:, 0] * np.sin(np.radians(d[:, 1])), d[:, 0] * np.cos(np.radians(d[:, 1])), "o", color=RADAR, ms=5)
                for i, o in enumerate(rec["objects"]):
                    y, x = o["range_m"] * np.sin(np.radians(o["azimuth_deg"])), o["range_m"] * np.cos(np.radians(o["azimuth_deg"]))
                    src = rec["source"] and abs(o["range_m"] - rec["source"][0]) < 0.05
                    axr.plot(y, x, "s", mfc="none", color=LIGHT if src else WHITE, ms=10)
                    axr.text(y + 0.04, x, f"obj{i + 1}", color=WHITE, fontsize=7, family=MONO)
                axr.plot(0, 0, "^", color=UI, ms=8)
                axr.set_xlim(-0.6, 0.6)
                axr.set_ylim(-0.05, 1.15)
                axr.set_aspect("equal")
                axr.grid(color=UI, alpha=0.15)
                if rec["light"]:
                    where = (f"radar obj at {rec['source'][0]:.2f}m {rec['source'][1]:+.1f}deg" if rec["source"]
                             else "camera core only (no radar object)")
                    hud_text(fig, 0.665, 0.545, f"light: ON   source: {where}", 8, LIGHT)
                    if rec["glare"] is not None:
                        b, a, p = rec["glare"][2][1]
                        hud_text(fig, 0.665, 0.52, f"veil fit L = b + a*rho^-p   p={p:.2f}", 8, UI)
                else:
                    hud_text(fig, 0.665, 0.545, "light: off", 8, UI)
                hud_text(fig, 0.665, 0.47, "reading now           raw          filtered", 8, UI)
                for row_i, name in enumerate(("new", "cnn")):
                    y = 0.44 - 0.03 * row_i
                    hud_text(fig, 0.665, y, "MetricSight" if name == "new" else "cnn", 8, WHITE)
                    for ci, cnd in enumerate(("raw", "filt")):
                        txt = rec.get(f"{name}_{cnd}", "")
                        col = (OK if txt == truth else BAD) if truth and txt else UI
                        if cnd == cond or pi == 1:
                            hud_text(fig, 0.79 + 0.105 * ci, y, (txt or "-")[:11], 8, col)
                axb = panel_axes(fig, [0.70, 0.08, 0.27, 0.25], "character error rate while the lamp is on [%]")
                vals, labels, cols = [], [], []
                for name, lab in (("new", "MetricSight"), ("cnn", "cnn")):
                    for cnd in ("raw", "filt"):
                        v = cer(seen if pi == 0 or cnd == cond else recs, f"{name}_{cnd}_cer", 1) if (cnd == "raw" or pi == 1) else float("nan")
                        vals.append(0 if np.isnan(v) else v)
                        labels.append(lab[:5] + chr(10) + ("raw" if cnd == "raw" else "removed"))
                        cols.append(OK if cnd == "filt" else UI)
                axb.bar(range(4), vals, color=cols)
                for x, v in enumerate(vals):
                    if v:
                        axb.text(x, v + 2, f"{v:.0f}", ha="center", color=WHITE, fontsize=7, family=MONO)
                axb.set_xticks(range(4), labels, fontsize=7, color=WHITE)
                axb.set_ylim(0, max(110, max(vals) + 10))
                off = cer(seen, f"new_{cond}_cer", 0)
                hud_text(fig, 0.665, 0.37, f"MetricSight CER, lamp off so far: {off:5.1f} %" if not np.isnan(off) else "", 8, UI)
                hud_text(fig, 0.012, 0.115, f"read ({'raw' if cond == 'raw' else 'filtered'})" + (f"   (truth {truth})" if truth else ""), 8, UI)
                for row_i, (name, lab) in enumerate((("new", "MetricSight"), ("cnn", "cnn"))):
                    y = 0.075 - row_i * 0.032
                    hud_text(fig, 0.012, y, lab, 9, WHITE)
                    txt = rec.get(f"{name}_{cond}", "")
                    for q, ch in enumerate(txt[:12]):
                        col = (OK if (q < len(truth) and ch == truth[q]) else BAD) if truth else WHITE
                        hud_text(fig, 0.11 + 0.012 * q, y, ch, 11, col)
                writer.grab_frame()
                if j % 60 == 0:
                    print(f"  pass {pi + 1}: frame {j}/{n}  ({time.perf_counter() - t_start:.0f} s)", flush=True)
    plt.close(fig)


def frames_at(take: Path, ks):
    for _, rgb in dataset.video_frames(take, set(ks)):
        yield rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path, help="light take folder recordings/<session>/<take>")
    args = ap.parse_args()
    from backend.alignment import Projector
    recs = records(args.take.name)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out = VIDEOS / f"{args.take.name}_light.mp4"
    render(args.take, recs, dataset.TRUTH, Projector(), load_config(), dataset.FPS / dataset.STEP, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
