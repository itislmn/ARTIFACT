"""
plots.py - the tutorial's figures from output/cache/results.pkl: the timing race, the virtual array, raw samples and
their spectrum, the radar cube of the recording, the synthetic targets and the UART point cloud; PDFs and readme.svg.

    python plots.py            (python plots.py cube: only this figure)
"""
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial import dataset  # noqa: E402
from radar_tutorial.evaluate import CACHE, U  # noqa: E402
from visualization import style  # noqa: E402

FIGURES = dataset.OUTPUT / "figures"
R_SHOW = 7.0


def db(x, ref=None):
    x = np.asarray(x, float)
    return 10 * np.log10(x / (ref if ref is not None else x.max()) + 1e-12)


def timing(res, pdf):
    fig, ax = plt.subplots(figsize=style.size(style.SINGLE, 0.75), layout="constrained")
    for i, t in enumerate(res["protocols"]["timing"]):
        ax.barh(i + 0.18, t["cfg_seconds"], height=0.32, color=style.ACCENT if t["packets"] == 0 else style.PRIMARY)
        ax.barh(i - 0.18, t["window_seconds"], height=0.32, color=style.MUTED)
        ax.text(max(t["cfg_seconds"], t["window_seconds"]) + 0.5, i, f"{t['packets']} packets", va="center", fontsize=7)
    handles = [Patch(color=style.PRIMARY, label=".cfg Sent over UART, Data inside the Window"),
               Patch(color=style.ACCENT, label=".cfg Sent over UART, Data after the Window"),
               Patch(color=style.MUTED, label="DCA1000 Receive Window")]
    ax.set_yticks([0, 1], [r"$\mathtt{read(400)}$, 1 s Timeout", "Quiet-Period Read"])
    ax.set_xlabel("Seconds after Arming the DCA1000")
    ax.set_xlim(0, 1.45 * max(max(t["cfg_seconds"], t["window_seconds"]) for t in res["protocols"]["timing"]))
    style.title(ax, "UART and UDP Race")
    fig.legend(handles=handles, loc="outside lower center", ncols=1)
    style.save(pdf, fig)


def array(res, pdf):
    r = res["radar"]
    pos = r.virtual_positions
    fig, ax = plt.subplots(figsize=style.size(style.DOUBLE, 0.3), layout="constrained")
    markers = ["o", "s", "^", "D"]
    slot = {t: s for t, s in enumerate(r.tx_reorder)}
    for t in range(r.n_tx):
        for rx in range(r.n_rx):
            v = t * r.n_rx + rx
            ax.plot(pos[v, 0] * r.d_az, -pos[v, 1] * r.d_el + pos[:, 1].max() * r.d_el, markers[rx], ms=7,
                    color=style.SERIES[t], label=f"TX{t} (Fires in Slot {slot[t]})" if rx == 0 else None)
            ax.text(pos[v, 0] * r.d_az, -pos[v, 1] * r.d_el + pos[:, 1].max() * r.d_el + 0.22, str(v), ha="center",
                    fontsize=6)
    for rx in range(r.n_rx):
        ax.plot([], [], markers[rx], color=style.NORM, mfc="white", label=f"RX{rx}")
    az = r.azimuth_channels
    ax.axhspan(-0.15, 0.15, color=style.GOOD, alpha=0.25, lw=0, label=f"Azimuth Row ({az.size} Channels)")
    ax.set_xlabel(r"Horizontal Position (Wavelengths, $d = \lambda/2$)")
    ax.set_ylabel("Height (Wavelengths)")
    ax.set_ylim(-0.5, 1.4)
    ax.set_xlim(-0.4, pos[:, 0].max() * r.d_az + 0.4)
    style.title(ax, "Virtual Array")
    style.legend_below(fig, [ax], ncols=5)
    style.save(pdf, fig)


def samples(res, pdf):
    r, m = res["radar"], res["maps"]
    fig, axs = plt.subplots(1, 3, figsize=style.size(style.DOUBLE, 0.32), layout="constrained")
    t = np.arange(r.n_samples) / r.fs_hz * 1e6
    axs[0].plot(t, m["samples"], color=style.PRIMARY, lw=0.6)
    axs[0].set_xlabel(r"Time in the Chirp ($\mu$s)")
    axs[0].set_ylabel("ADC Value (int16)")
    style.title(axs[0], "One Chirp, TX0 RX0 (Real ADC)")
    spec = np.abs(m["spectrum"]) ** 2
    k = np.arange(r.n_samples)
    axs[1].plot(k[: r.n_range_bins] * r.range_res_m, db(spec[: r.n_range_bins]), color=style.PRIMARY, lw=0.7,
                label=r"Bins $k < N_s/2$: Range")
    axs[1].plot(k[r.n_range_bins:] * r.range_res_m, db(spec[r.n_range_bins:], spec.max()), color=style.MUTED, lw=0.7,
                label=r"Bins $k \geq N_s/2$: Mirror ($X_{N-k} = X_k^*$)")
    axs[1].set_xlabel(r"$k\,\Delta R$ (m)")
    axs[1].set_ylabel("Power (dB)")
    axs[1].set_ylim(-80, 3)
    style.title(axs[1], "Its FFT: Complex, Mirrored")
    snap = m["peak_snapshot"][r.azimuth_channels]
    axs[2].plot(np.arange(snap.size), np.unwrap(np.angle(snap)), "o-", color=style.PRIMARY, ms=3.5,
                label=f"Phase at {m['peak_bin'] * r.range_res_m:.2f} m")
    axs[2].set_xlabel(r"Azimuth Element $m$ ($d = \lambda/2$)")
    axs[2].set_ylabel("Phase (rad)")
    style.title(axs[2], r"Phase Across the Row: $2\pi d m \sin\theta$")
    style.legend_below(fig, axs, ncols=3)
    style.save(pdf, fig)


def cube(res, pdf):
    r, m = res["radar"], res["maps"]
    rng = r.range_axis_m
    show = rng <= R_SHOW
    fig, axs = plt.subplots(1, 3, figsize=style.size(style.DOUBLE, 0.36), layout="constrained")
    axs[0].plot(rng[show], db(m["range_profile"][show]), color=style.PRIMARY, lw=0.8, label="Mean over Loops, Channels")
    f0 = np.array([[p["range_m"], p["velocity_mps"], p["azimuth_deg"]] for p in res["raw"]
                   if p["frame"] == 0 and p["range_m"] <= R_SHOW])
    axs[0].set_xlabel("Range (m)")
    axs[0].set_ylabel("Power (dB)")
    style.title(axs[0], "Range Profile")
    im = axs[1].pcolormesh(rng[show], r.velocity_axis_mps, db(m["range_doppler"][:, show]), cmap=style.SEQUENTIAL,
                           vmin=-60, vmax=0, shading="nearest", rasterized=True)
    axs[1].plot(f0[:, 0], f0[:, 1], "o", mfc="none", color=style.ACCENT, ms=4, mew=0.8, label="CFAR Detection")
    axs[1].set_xlabel("Range (m)")
    axs[1].set_ylabel("Velocity (m/s)")
    style.title(axs[1], "Range-Doppler (Frame 0)")
    th = np.degrees(np.arcsin(U))
    axs[2].pcolormesh(th, rng[show], db(m["range_azimuth"][show]), cmap=style.SEQUENTIAL, vmin=-60, vmax=0,
                      shading="nearest", rasterized=True)
    axs[2].plot(f0[:, 2], f0[:, 0], "o", mfc="none", color=style.ACCENT, ms=4, mew=0.8)
    axs[2].set_xlim(-70, 70)
    axs[2].set_xlabel(r"Azimuth $\theta$ ($^\circ$)")
    axs[2].set_ylabel("Range (m)")
    style.title(axs[2], "Range-Azimuth (Frame 0)")
    fig.colorbar(im, ax=axs[1:], label="Power (dB)", shrink=0.9)
    style.legend_below(fig, axs, ncols=2)
    style.title(fig, "Recorded Radar Cube")
    style.save(pdf, fig)


def synthetic(res, pdf):
    s = res["synthetic"]
    fig, axs = plt.subplots(1, 3, figsize=style.size(style.DOUBLE, 0.33), layout="constrained")
    for ax, key, unit, name in ((axs[0], "range_m", "m", "Range"), (axs[1], "velocity_mps", "m/s", "Velocity"),
                                (axs[2], "azimuth_deg", r"$^\circ$", "Azimuth")):
        t = np.array([x["true_" + key] for x in s])
        e = np.array([x[key] for x in s])
        lo, hi = min(t.min(), np.nanmin(e)), max(t.max(), np.nanmax(e))
        pad = 0.08 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="-.", color=style.GOAL, lw=0.9, label="Truth")
        ax.plot(t, e, "o", color=style.PRIMARY, ms=5, label="Found from the Raw Bytes")
        ax.set_xlabel(f"True {name} ({unit})")
        ax.set_ylabel(f"Found {name} ({unit})")
        style.title(ax, name)
    style.legend_below(fig, axs, ncols=2)
    style.title(fig, "Synthetic Targets through the Chain")
    style.save(pdf, fig)


def pointcloud(res, pdf):
    frames = res["uart_frames"]
    pts = np.array([[p["x_m"], p["y_m"], p["velocity_mps"]] for p in res["points"]])
    fig, axs = plt.subplots(1, 3, figsize=style.size(style.DOUBLE, 0.36), layout="constrained",
                            gridspec_kw={"width_ratios": [1, 1, 1.2]})
    still = np.abs(pts[:, 2]) < 0.05
    axs[0].plot(pts[still, 1], pts[still, 0], ".", color=style.MUTED, ms=2, label=r"$|v| < 0.05$ m/s")
    sc = axs[0].scatter(pts[~still, 1], pts[~still, 0], c=pts[~still, 2], cmap=style.SEQUENTIAL, s=5,
                        label="Moving (Colour: $v$)")
    axs[0].set_xlabel("Sideways $y$ (m)")
    axs[0].set_ylabel("Forward $x$ (m)")
    axs[0].set_xlim(-6, 6)
    axs[0].set_ylim(0, 10)
    fig.colorbar(sc, ax=axs[0], label="Velocity (m/s)", shrink=0.8)
    style.title(axs[0], "Every Point of the Recording")
    n = np.array([f["n_obj"] for f in frames])
    axs[1].plot(np.arange(n.size) * 0.1, n, color=style.PRIMARY, lw=0.8, label="Points per Frame")
    axs[1].set_xlabel("Time (s, 10 Frames/s)")
    axs[1].set_ylabel("Points")
    style.title(axs[1], "Points per Frame")
    prof = np.array([f["range_profile"] for f in frames if f["range_profile"] is not None], float)
    axs[2].pcolormesh(np.arange(prof.shape[1]), np.arange(prof.shape[0]) * 0.1, 20 * np.log10(2) * prof,
                      cmap=style.SEQUENTIAL, shading="nearest", rasterized=True)
    axs[2].set_xlabel("Range Bin (On-Chip)")
    axs[2].set_ylabel("Time (s)")
    style.title(axs[2], "Range Profile TLV over Time (dB)")
    style.legend_below(fig, axs, ncols=3)
    style.title(fig, "On-Chip Point Cloud")
    style.save(pdf, fig)


def readme_figure(res, path):
    r, m = res["radar"], res["maps"]
    rng = r.range_axis_m
    show = rng <= R_SHOW
    f0 = np.array([[p["range_m"], p["azimuth_deg"]] for p in res["raw"] if p["frame"] == 0 and p["range_m"] <= R_SHOW])
    pts = np.array([[p["x_m"], p["y_m"]] for p in res["points"]])
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.4), layout="constrained")
    axs[0].pcolormesh(np.degrees(np.arcsin(U)), rng[show], db(m["range_azimuth"][show]), cmap=style.SEQUENTIAL,
                      vmin=-60, vmax=0, shading="nearest", rasterized=True)
    axs[0].plot(f0[:, 1], f0[:, 0], "o", mfc="none", color=style.ACCENT, ms=4, mew=0.8, label="CFAR Point (Ours)")
    axs[0].set_xlim(-70, 70)
    axs[0].set_xlabel(r"Azimuth ($^\circ$)")
    axs[0].set_ylabel("Range (m)")
    style.title(axs[0], "Raw ADC via DCA1000: Range-Azimuth")
    axs[1].plot(pts[:, 1], pts[:, 0], ".", color=style.PRIMARY, ms=2, label="On-Chip Point")
    axs[1].set_xlim(-6, 6)
    axs[1].set_ylim(0, 10)
    axs[1].set_xlabel("Sideways $y$ (m)")
    axs[1].set_ylabel("Forward $x$ (m)")
    style.title(axs[1], "On-Chip Point Cloud via UART")
    style.legend_below(fig, axs, ncols=2)
    style.save(path, fig)


FIGS = (("timing", timing), ("array", array), ("samples", samples), ("cube", cube), ("synthetic", synthetic),
        ("pointcloud", pointcloud))


def draw(res, only=()):
    style.use()
    for name, fn in FIGS:
        if not only or name in only:
            with style.pdf(FIGURES / f"{name}.pdf") as pdf:
                fn(res, pdf)
            print(f"  figures/{name}.pdf")
    if not only or "readme" in only:
        readme_figure(res, FIGURES / "readme.svg")
        print("  figures/readme.svg")


if __name__ == "__main__":
    draw(pickle.loads(CACHE.read_bytes()), sys.argv[1:])
