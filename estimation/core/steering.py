"""
steering.py - the steering vectors of all 16 virtual channels, the elevation from the TX1 row and the plane-wave
check of a measured snapshot.

    from estimation.core.steering import check, elevation_deg
"""
from __future__ import annotations

import numpy as np

from estimation.core.array import ArrayModel
from estimation.core.radar import RadarConfig, open_recording, range_fft
from estimation.core.truth import EL_SIGN

ALT_COLUMNS = {0: 0, 1: 1, 2: 2, 3: 3, 12: 4, 13: 5, 14: 6, 15: 7, 4: 8, 5: 9, 6: 10, 7: 11}


def all_channel_steering(cfg: RadarConfig, theta_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    a = np.exp(2j * np.pi * cfg.x_lam * np.sin(np.radians(theta_deg)))
    return a * np.exp(1j * EL_SIGN * 2 * np.pi * cfg.z_lam * np.sin(np.radians(elevation_deg)))


def elevation_deg(y16, cfg: RadarConfig, thetas_deg, az_sign: int = 1):
    rows = np.asarray(cfg.virt_rows)
    main, el = rows == rows.max(), rows != rows.max()
    u = az_sign * np.sin(np.radians(np.atleast_1d(np.asarray(thetas_deg, float))))
    A = np.exp(2j * np.pi * cfg.x_lam[:, None] * u[None, :])
    s = np.linalg.lstsq(A[main], y16[main], rcond=None)[0]
    c = np.linalg.lstsq(A[el], y16[el], rcond=None)[0]
    psi = np.angle(c / s)
    phi = np.degrees(np.arcsin(np.clip(EL_SIGN * psi / (2 * np.pi * cfg.z_lam[el].mean()), -1.0, 1.0)))
    return phi, np.abs(c / s)


def describe(cfg: RadarConfig, array: ArrayModel) -> str:
    lam_mm = cfg.wavelength_m * 1e3
    rows, cols = np.asarray(cfg.virt_rows), np.asarray(cfg.virt_cols)
    lines = [f"wavelength at the centre of the sampled band: lambda = {lam_mm:.3f} mm "
             f"(f_c = {299.792458 / lam_mm:.2f} GHz)",
             f"spacing d = {cfg.d_az} lambda = {cfg.d_az * lam_mm:.3f} mm (azimuth), elevation offset {cfg.d_el} lambda",
             "physical positions (azimuth x, elevation z) in lambda, from antGeometryCfg:"]
    rx_x = cols[: cfg.n_rx] * cfg.d_az
    lines.append("  RX0..3 : x = " + ", ".join(f"{x:.1f}" for x in rx_x))
    for t in range(cfg.n_tx):
        c0, r0 = cols[t * cfg.n_rx], rows[t * cfg.n_rx]
        lines.append(f"  TX{t}    : x = {c0 * cfg.d_az:.1f}, z = {(rows.max() - r0) * cfg.d_el:.1f}"
                     f"  (chirp slot {cfg.tx_reorder[t]})")
    lines.append("virtual channel v = 4 tx + rx -> (row, column): " +
                 ", ".join(f"{v}:({rows[v]},{cols[v]})" for v in range(cfg.n_virt)))
    lines.append(f"azimuth ULA: channels {[int(c) for c in array.channels]} at columns "
                 f"{[int(round(c)) for c in array.positions / array.d]}")
    A = array.steering(array.u_grid)
    coh = np.abs(np.sum(A[:, :-1].conj() * A[:, 1:], axis=0)) / array.n
    lines.append(f"grid: p = {array.p} angles, sin(theta_j) = j/{round(1 / array.du)}, theta in "
                 f"[{array.theta_grid_deg[0]:.1f}, {array.theta_grid_deg[-1]:.1f}] deg; adjacent-column coherence "
                 f"|a_j^H a_(j+1)|/N = {coh.mean():.3f}")
    lines.append(f"Rayleigh resolution 1/(Nd) = {array.rayleigh_u:.4f} in sin(theta) "
                 f"({np.degrees(np.arcsin(array.rayleigh_u)):.1f} deg at broadside); no grating lobes (d = lambda/2)")
    lines.append(f"Nyquist grid (BH/BY): sin(theta_k) = k/{array.n * array.d:g}, p = {array.nyquist_grid.size}, "
                 "orthogonal columns")
    lines.append("TDM: TX slots fire 62.5 us apart; static reflectors need no slot-phase compensation")
    return "\n".join(lines)


def take_snapshot(path, cfg, n_frames=None, cal=None, background=None, range_m=None, others_m=()):
    n, get = open_recording(path, cfg)
    n = min(n, n_frames or n)
    acc = None
    for i in range(n):
        X = range_fft(get(i), cfg, cal).mean(axis=1)
        if background is not None:
            X = X - background["rfft"]
        acc = X if acc is None else acc + X
    X = acc / n
    az = list(ArrayModel(cfg).channels)
    k0 = int(np.ceil(0.45 / cfg.range_res_m))
    if range_m is None:
        k = k0 + int(np.argmax(np.sum(np.abs(X[k0:, az]) ** 2, axis=1)))
    else:
        ks = np.arange(int((range_m - 0.15) / cfg.range_res_m), int((range_m + 0.15) / cfg.range_res_m) + 1)
        for o in others_m:
            ks = ks[np.abs(ks * cfg.range_res_m - range_m) < np.abs(ks * cfg.range_res_m - o)]
        k = int(ks[np.argmax(np.sum(np.abs(X[ks][:, az]) ** 2, axis=1))])
    return X[k], k * cfg.range_res_m


def plane_wave_fit(y12, columns, d=0.5):
    u = np.linspace(-1, 1, 4001)
    A = np.exp(2j * np.pi * d * np.asarray(columns, float)[:, None] * u[None, :])
    g = np.abs(A.conj().T @ y12) ** 2 / (len(columns) * np.vdot(y12, y12).real)
    i = int(np.argmax(g))
    return float(g[i]), float(np.degrees(np.arcsin(u[i])))


def check(y16, cfg, array, theta_true=None) -> dict:
    ch = list(array.channels)
    fit, th = plane_wave_fit(y16[ch], array.positions / array.d, array.d)
    alt = np.array(sorted(ALT_COLUMNS, key=ALT_COLUMNS.get))
    fit_alt, th_alt = plane_wave_fit(y16[alt], [ALT_COLUMNS[c] for c in alt], array.d)
    out = {"fit": fit, "theta_hat": th, "fit_alt": fit_alt, "theta_hat_alt": th_alt,
           "residual_db": 10 * np.log10(max(1 - fit, 1e-12))}
    if theta_true is not None:
        z = y16 / all_channel_steering(cfg, theta_true)
        ph = np.angle(z * np.conj(np.mean(z / np.abs(z))))
        out["phase_residual_deg"] = np.degrees(ph)
    return out
