"""
STEP 3 - Turn the raw .bin into figures your professors will recognise.

Parses your sensor .cfg to derive the radar cube dimensions and the physical
range / velocity axes (no hard-coded magic numbers), reshapes the DCA1000
binary into (frames, chirps, rx, samples), then produces:

  - range profile with a correct metre axis
  - range-Doppler map in dB with a correct m/s axis
  - range-time intensity (micro-motion over the capture)
  - a simple CA-CFAR detection overlay

    python step3_process_adc.py --bin captures/adc_Raw_0.bin --cfg profile_lvds.cfg

If the reshape fails, read the printed "expected vs actual" numbers - that
mismatch is almost always numTx (MIMO / DDM) or complex-vs-real ADC format.
"""
import argparse
import os
import re

import numpy as np

C = 299792458.0


# ----------------------------------------------------------------------------
# cfg parsing
# ----------------------------------------------------------------------------
def parse_cfg(path):
    p = {"chirpTx": {}}
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        t = re.split(r"\s+", line)
        k, v = t[0], t[1:]
        if k == "profileCfg":
            p["startFreq_GHz"] = float(v[1])
            p["idleTime_us"] = float(v[2])
            p["rampEndTime_us"] = float(v[5])
            p["slope_MHz_us"] = float(v[7])
            p["numAdcSamples"] = int(v[9])
            p["sampleRate_ksps"] = float(v[10])
        elif k == "frameCfg":
            p["chirpStartIdx"] = int(v[0])
            p["chirpEndIdx"] = int(v[1])
            p["numLoops"] = int(v[2])
            p["numFrames"] = int(v[3])
            p["framePeriod_ms"] = float(v[4])
        elif k == "channelCfg":
            p["rxMask"] = int(v[0])
            p["txMask"] = int(v[1])
        elif k == "adcCfg":
            p["adcBits"] = int(v[0])
            p["adcOutputFmt"] = int(v[1])  # 1 = complex 1x, 2 = complex 2x, 0 = real
        elif k == "chirpCfg":
            p["chirpTx"][int(v[0])] = int(v[7])

    p["numRx"] = bin(p.get("rxMask", 15)).count("1")
    p["numTx"] = bin(p.get("txMask", 1)).count("1")
    if p.get("chirpTx"):
        p["numTx"] = len({m for m in p["chirpTx"].values() if m})
    p["chirpsPerLoop"] = p.get("chirpEndIdx", 0) - p.get("chirpStartIdx", 0) + 1
    p["chirpsPerFrame"] = p["chirpsPerLoop"] * p.get("numLoops", 1)
    p["isComplex"] = p.get("adcOutputFmt", 1) != 0
    return p


def derive_axes(p):
    N = p["numAdcSamples"]
    fs = p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12          # Hz/s
    bw = slope * (N / fs)
    range_res = C / (2 * bw)
    range_axis = np.arange(N) * (C * fs) / (2 * slope * N)

    lam = C / (p["startFreq_GHz"] * 1e9 + bw / 2)
    tc = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6 * p["numTx"]
    n_dop = p["chirpsPerFrame"] // p["numTx"]
    v_max = lam / (4 * tc)
    v_axis = np.linspace(-v_max, v_max, n_dop, endpoint=False)
    return {
        "bw_GHz": bw / 1e9, "range_res_m": range_res, "range_axis": range_axis,
        "max_range_m": range_axis[-1], "v_res_ms": 2 * v_max / n_dop,
        "v_max_ms": v_max, "v_axis": v_axis, "n_doppler": n_dop,
        "frame_rate_Hz": 1000.0 / p.get("framePeriod_ms", 100.0),
    }


# ----------------------------------------------------------------------------
# binary -> radar cube
# ----------------------------------------------------------------------------
def load_cube(bin_path, p, lvds_lanes=2):
    """DCA1000 non-interleaved complex, 2 LVDS lanes: int16 quads [I0 I1 Q0 Q1]."""
    adc = np.fromfile(bin_path, dtype=np.int16)
    if p["isComplex"]:
        adc = adc.reshape(-1, 4)
        iq = adc[:, 0:2] + 1j * adc[:, 2:4]
        data = iq.reshape(-1)
    else:
        data = adc.astype(np.complex64)

    N, nrx = p["numAdcSamples"], p["numRx"]
    per_chirp = N * nrx
    per_frame = per_chirp * p["chirpsPerFrame"]
    n_frames = data.size // per_frame

    print(f"  samples in file : {data.size}")
    print(f"  per frame       : {per_frame}  "
          f"({p['chirpsPerFrame']} chirps x {nrx} rx x {N} samples)")
    print(f"  complete frames : {n_frames}  (dropping {data.size % per_frame} trailing samples)")
    if n_frames == 0:
        raise SystemExit(
            "Zero complete frames. Your numTx/numRx/numAdcSamples do not match the file.\n"
            "Check adcbufCfg (interleaved vs non-interleaved) and whether DDM is on.")

    cube = data[: n_frames * per_frame].reshape(n_frames, p["chirpsPerFrame"], nrx, N)
    # de-interleave TDM-MIMO: chirp k belongs to tx (k % numTx)
    if p["numTx"] > 1:
        cube = cube.reshape(n_frames, -1, p["numTx"], nrx, N)
        cube = cube.transpose(0, 1, 2, 3, 4).reshape(n_frames, -1, p["numTx"] * nrx, N)
    return cube


# ----------------------------------------------------------------------------
# DSP
# ----------------------------------------------------------------------------
def range_doppler(frame, remove_static=True):
    """frame: (chirps, vrx, samples) -> (doppler, range) magnitude in dB."""
    w_r = np.hanning(frame.shape[-1])
    rfft = np.fft.fft(frame * w_r, axis=-1)
    if remove_static:
        rfft = rfft - rfft.mean(axis=0, keepdims=True)
    w_d = np.hanning(frame.shape[0])[:, None, None]
    dfft = np.fft.fftshift(np.fft.fft(rfft * w_d, axis=0), axes=0)
    mag = np.abs(dfft).sum(axis=1)
    return 20 * np.log10(mag + 1e-9)


def ca_cfar_2d(rd_db, guard=(2, 2), train=(6, 6), offset_db=8.0):
    from scipy.ndimage import uniform_filter
    gw = (2 * guard[0] + 1, 2 * guard[1] + 1)
    tw = (2 * (guard[0] + train[0]) + 1, 2 * (guard[1] + train[1]) + 1)
    lin = 10 ** (rd_db / 20.0)
    big = uniform_filter(lin, size=tw, mode="nearest") * (tw[0] * tw[1])
    small = uniform_filter(lin, size=gw, mode="nearest") * (gw[0] * gw[1])
    n_train = tw[0] * tw[1] - gw[0] * gw[1]
    noise = (big - small) / max(n_train, 1)
    thr = 20 * np.log10(noise + 1e-9) + offset_db
    return rd_db > thr


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)

    p = parse_cfg(args.cfg)
    ax = derive_axes(p)
    print("\n=== Derived radar parameters ===")
    print(f"  bandwidth        {ax['bw_GHz']:.3f} GHz")
    print(f"  range resolution {ax['range_res_m']*100:.1f} cm")
    print(f"  max range        {ax['max_range_m']:.1f} m")
    print(f"  velocity res     {ax['v_res_ms']:.3f} m/s   max +/-{ax['v_max_ms']:.2f} m/s")
    print(f"  frame rate       {ax['frame_rate_Hz']:.1f} Hz")
    print(f"  Tx={p['numTx']}  Rx={p['numRx']}  virtual={p['numTx']*p['numRx']}")

    print("\n=== Loading cube ===")
    cube = load_cube(args.bin, p)
    print(f"  cube shape {cube.shape}  (frames, chirps, vrx, samples)")

    f = cube[min(args.frame, cube.shape[0] - 1)]

    # 1. range profile
    rp = 20 * np.log10(np.abs(np.fft.fft(f * np.hanning(f.shape[-1]), axis=-1)).mean((0, 1)) + 1e-9)
    n_half = len(rp) // 2
    plt.figure(figsize=(8, 4))
    plt.plot(ax["range_axis"][:n_half], rp[:n_half])
    plt.xlabel("Range [m]"); plt.ylabel("Magnitude [dB]")
    plt.title("Range profile (AWR2944P raw ADC)"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{args.outdir}/range_profile.png", dpi=150)

    # 2. range-Doppler
    rd = range_doppler(f)
    plt.figure(figsize=(8, 5))
    plt.imshow(rd[:, :n_half], aspect="auto", origin="lower", cmap="viridis",
               extent=[ax["range_axis"][0], ax["range_axis"][n_half - 1],
                       ax["v_axis"][0], ax["v_axis"][-1]])
    plt.colorbar(label="dB"); plt.xlabel("Range [m]"); plt.ylabel("Velocity [m/s]")
    plt.title("Range-Doppler map (static clutter removed)")
    plt.tight_layout(); plt.savefig(f"{args.outdir}/range_doppler.png", dpi=150)

    # 3. CFAR overlay
    try:
        det = ca_cfar_2d(rd[:, :n_half])
        yy, xx = np.nonzero(det)
        plt.figure(figsize=(8, 5))
        plt.imshow(rd[:, :n_half], aspect="auto", origin="lower", cmap="gray",
                   extent=[ax["range_axis"][0], ax["range_axis"][n_half - 1],
                           ax["v_axis"][0], ax["v_axis"][-1]])
        plt.scatter(ax["range_axis"][xx], ax["v_axis"][yy], s=6, c="red")
        plt.xlabel("Range [m]"); plt.ylabel("Velocity [m/s]")
        plt.title(f"CA-CFAR detections ({det.sum()} cells)")
        plt.tight_layout(); plt.savefig(f"{args.outdir}/cfar.png", dpi=150)
    except ImportError:
        print("  (scipy not installed, skipping CFAR)")

    # 4. range-time intensity across all frames
    n_show = min(cube.shape[0], 400)
    rti = np.stack([
        20 * np.log10(np.abs(np.fft.fft(cube[i] * np.hanning(cube.shape[-1]),
                                        axis=-1)).mean((0, 1)) + 1e-9)[:n_half]
        for i in range(n_show)])
    rti -= rti.mean(axis=0, keepdims=True)
    plt.figure(figsize=(9, 5))
    plt.imshow(rti.T, aspect="auto", origin="lower", cmap="inferno",
               extent=[0, n_show / ax["frame_rate_Hz"], 0, ax["range_axis"][n_half - 1]])
    plt.colorbar(label="dB rel. mean"); plt.xlabel("Time [s]"); plt.ylabel("Range [m]")
    plt.title("Range-Time Intensity")
    plt.tight_layout(); plt.savefig(f"{args.outdir}/range_time.png", dpi=150)

    np.save(f"{args.outdir}/radar_cube.npy", cube)
    print(f"\nFigures + radar_cube.npy written to {args.outdir}/")
    print("radar_cube.npy is your ML-ready tensor: (frames, chirps, virtual_rx, samples)")


if __name__ == "__main__":
    main()