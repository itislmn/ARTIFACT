"""
simulate.py - a simulated scene for dry runs: a swinging ball and a walking person as point scatterers, rendered as
raw ADC frames with the board's TDM timing.

    from hardware.simulate import Renderer, scenario
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

import paths
from estimation.core.radar import C0
from estimation.simulation import amplitude_for_snr
from estimation.core.truth import EL_SIGN

G = 9.81
RCS_REF, R_REF, SNR_REF_DB = 11.7, 2.0, 69.0


def snr_db(rcs_m2: float, r_m: np.ndarray) -> np.ndarray:
    return SNR_REF_DB + 10 * np.log10(rcs_m2 / RCS_REF) - 40 * np.log10(np.maximum(r_m, 0.05) / R_REF)


@dataclass
class Pendulum:
    name: str = "ball"
    pivot: tuple = (1.2, -0.3, 0.6)
    length_m: float = 0.6
    swing_dir_deg: float = 90.0
    amplitude_deg: float = 25.0
    rcs_m2: float = 0.01
    phase: float = 0.0

    def points(self, t):
        w = np.sqrt(G / self.length_m)
        phi = np.radians(self.amplitude_deg) * np.cos(w * np.asarray(t) + self.phase)
        e = np.array([np.cos(np.radians(self.swing_dir_deg)), np.sin(np.radians(self.swing_dir_deg)), 0.0])
        p = np.asarray(self.pivot)[None] + self.length_m * (np.sin(phi)[..., None] * e - np.cos(phi)[..., None] * [0, 0, 1])
        return p[:, None, :], np.array([self.rcs_m2])

    def centre(self, t):
        return self.points(np.atleast_1d(t))[0][:, 0]


@dataclass
class Walker:
    name: str = "person"
    a: tuple = (1.6, 0.9, -0.38)
    b: tuple = (3.4, 0.9, -0.38)
    peak_speed: float = 1.0
    gait_hz: float = 0.9
    phase: float = 0.0
    parts: tuple = ((1.10, 0.00, 0.00, 0, 0.60), (1.62, 0.00, 0.00, 0, 0.05),
                    (1.00, -0.22, 0.20, -1, 0.04), (1.00, 0.22, 0.20, 1, 0.04),
                    (0.45, -0.10, 0.30, 1, 0.08), (0.45, 0.10, 0.30, -1, 0.08))

    def _s(self, t):
        a, b = np.asarray(self.a), np.asarray(self.b)
        L = np.linalg.norm(b - a)
        T = np.pi * L / self.peak_speed
        arg = 2 * np.pi * np.asarray(t) / T + self.phase
        return a, (b - a) / L, L / 2 * (1 - np.cos(arg)), np.abs(np.sin(arg))

    def points(self, t):
        t = np.atleast_1d(t)
        a, e, s, frac = self._s(t)
        side = np.array([-e[1], e[0], 0.0])
        base = a[None] + s[:, None] * e[None]
        P, rcs = [], []
        for h, lat, amp, sign, sig in self.parts:
            swing = sign * amp * frac * np.sin(2 * np.pi * self.gait_hz * t)
            P.append(base + lat * side + h * np.array([0, 0, 1.0]) + swing[:, None] * e)
            rcs.append(sig)
        return np.stack(P, axis=1), np.array(rcs)

    def centre(self, t):
        return self.points(np.atleast_1d(t))[0][:, 0]


@dataclass
class Static:
    name: str = "clutter"
    pos: np.ndarray = field(default_factory=lambda: np.array(
        [[1.05, -0.75, -0.30], [1.05, 0.15, -0.30],
         [2.30, -1.30, -0.10], [3.80, 0.00, 0.20], [3.70, -1.50, 0.00],
         [0.06, 0.00, 0.00]]))
    rcs: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.3, 2.0, 1.0, 0.004]))

    def points(self, t):
        t = np.atleast_1d(t)
        return np.broadcast_to(self.pos, (t.size,) + self.pos.shape), self.rcs


@dataclass
class RadarPose:
    velocity: tuple = (0.0, 0.0, 0.0)
    yaw_rate_dps: float = 0.0
    t_start: float = 1e9
    t_end: float = 1e9

    def at(self, t):
        t = np.asarray(t, float)
        dt = np.clip(t, self.t_start, self.t_end) - self.t_start
        half = (self.t_end - self.t_start) / 2
        mov = np.where(dt <= half, dt, 2 * half - dt) if np.isfinite(half) else dt
        return np.asarray(self.velocity)[None] * mov[..., None], np.radians(self.yaw_rate_dps) * mov

    def velocity_at(self, t):
        t = np.asarray(t, float)
        half = (self.t_end - self.t_start) / 2
        inside = (t >= self.t_start) & (t <= self.t_end)
        sign = np.where(t - self.t_start <= half, 1.0, -1.0)
        return (np.asarray(self.velocity)[None] * (inside * sign)[..., None],
                np.radians(self.yaw_rate_dps) * inside * sign)


def to_radar(p_world, pose: RadarPose, t):
    pos, yaw = pose.at(t)
    d = p_world - pos[:, None, :]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], axis=-1)


def scenario(name: str, seed: int = 1):
    rng = np.random.default_rng(seed)
    objs = [Pendulum(phase=float(rng.uniform(0, 2 * np.pi))), Walker(phase=float(rng.uniform(0, 0.5)))]
    pose = RadarPose()
    if name == "ball_person":
        pass
    elif name == "moving_radar":
        pose = RadarPose(velocity=(0.0, 0.25, 0.0), yaw_rate_dps=8.0, t_start=8.0, t_end=14.0)
    elif name == "crossing":
        objs = [Pendulum(phase=float(rng.uniform(0, 2 * np.pi))),
                Walker(a=(2.2, -1.4, -0.38), b=(2.2, 1.2, -0.38), peak_speed=0.9)]
    elif name == "empty":
        objs = []
    else:
        raise SystemExit(f"unknown scenario {name}: ball_person, moving_radar, crossing, empty")
    return objs, Static(), pose


class Renderer:
    def __init__(self, objects, static: Static, pose: RadarPose, seed: int = 0,
                 cal_phase_std_deg: float = 4.0, clutter_residual_db: float = -20.0):
        self.objects, self.static, self.pose = objects, static, pose
        self.rng = np.random.default_rng(seed)
        self.delta = np.radians(cal_phase_std_deg) * self.rng.standard_normal(16)
        self.delta -= self.delta[0]
        self.eps = 10 ** (clutter_residual_db / 20)
        self.obj_phase = {o.name: self.rng.uniform(0, 2 * np.pi, 8) for o in objects}
        self.static_phase = self.rng.uniform(0, 2 * np.pi, len(static.rcs))
        self._static_cache = {}

    def _times(self, wf, t0):
        c = wf.cfg
        k = np.arange(c.n_loops)[:, None]
        slot = np.arange(c.n_tx)[None, :]
        within = c.idle_s + c.adc_start_s + 0.5 * c.n_samples / c.fs_hz
        return t0 + k * wf.loop_s + slot * wf.slot_s + within

    def _analytic(self, wf, P_r, rcs, phases):
        c = wf.cfg
        tx_of_v = np.arange(c.n_virt) // c.n_rx
        slot_of_v = np.asarray(c.tx_reorder)[tx_of_v]
        P = P_r[:, slot_of_v]
        R = np.linalg.norm(P, axis=-1)
        uy, uz = P[..., 1] / R, P[..., 2] / R
        x, z = c.x_lam[None, :, None], c.z_lam[None, :, None]
        el = EL_SIGN * 2 * np.pi * z * uz
        ph = (4 * np.pi * R / wf.wavelength_m + 2 * np.pi * x * uy + el
              + phases[None, None, :] + self.delta[None, :, None])
        a0 = amplitude_for_snr(0.0, c)
        amp = np.stack([a0 * 10 ** (snr_db(s, R[..., i]) / 20) for i, s in enumerate(rcs)], -1)
        fb = 2 * c.slope_hz_per_s * R.mean(axis=1) / C0
        n = (np.arange(c.n_samples) - c.n_samples / 2)[:, None]
        out = np.zeros((c.n_samples, c.n_loops, c.n_virt), np.complex64)
        for i in range(R.shape[-1]):
            tone = np.exp(2j * np.pi * fb[None, :, i] * n / c.fs_hz).astype(np.complex64)
            out += tone[:, :, None] * (amp[..., i] * np.exp(1j * ph[..., i])).astype(np.complex64)[None]
        return out

    def frame(self, wf, t0, noise: bool = True):
        c = wf.cfg
        T = self._times(wf, t0)
        flat = T.ravel()
        z = np.zeros((c.n_samples, c.n_loops, c.n_virt), np.complex64)
        for o in self.objects:
            pw, rcs = o.points(flat)
            P = to_radar(pw, self.pose, flat).reshape(T.shape + pw.shape[1:])
            z += self._analytic(wf, P, rcs, self.obj_phase[o.name][:len(rcs)])
        moving = self.pose.t_start <= t0 + c.n_loops * wf.loop_s and t0 <= self.pose.t_end
        if not moving and wf.name in self._static_cache:
            zs = self._static_cache[wf.name]
        else:
            pw, rcs = self.static.points(flat)
            P = to_radar(pw, self.pose, flat).reshape(T.shape + pw.shape[1:])
            zs = self._analytic(wf, P, rcs, self.static_phase)
            if not moving:
                self._static_cache[wf.name] = zs
        g = 1 + self.eps * (self.rng.standard_normal() + 1j * self.rng.standard_normal()) / np.sqrt(2)
        x = np.real(z + np.complex64(g) * zs).astype(np.float32)
        if noise:
            x += self.rng.standard_normal(x.shape, dtype=np.float32)
        return x, self.truth(wf, t0)

    def background(self, wf, n_frames: int = 20):
        from estimation.core.frame import background_from_recording
        saved, self.objects = self.objects, []
        pose, self.pose = self.pose, RadarPose()
        frames = [self.frame(wf, -10.0 - i * wf.frame_period_s)[0] for i in range(n_frames)]
        self.objects, self.pose = saved, pose
        return background_from_recording(n_frames, lambda i: frames[i], wf.cfg)

    def truth(self, wf, t0):
        tc = t0 + 0.5 * wf.cfg.n_loops * wf.loop_s
        out = []
        for o in self.objects:
            p = to_radar(o.centre(np.array([tc - 1e-3, tc, tc + 1e-3]))[:, None, :], self.pose,
                         np.array([tc - 1e-3, tc, tc + 1e-3]))[:, 0]
            R = np.linalg.norm(p, axis=1)
            vel = (p[2] - p[0]) / 2e-3
            out.append({"name": o.name, "xyz": p[1].tolist(), "range_m": float(R[1]),
                        "theta_deg": float(np.degrees(np.arcsin(p[1, 1] / R[1]))),
                        "vr_mps": float((R[2] - R[0]) / 2e-3), "vel": vel.tolist(),
                        "x_slant": float(np.sqrt(R[1] ** 2 - p[1, 1] ** 2)), "y": float(p[1, 1])})
        v, w = self.pose.velocity_at(np.array([tc]))
        return {"t": tc, "objects": out, "radar_velocity": v[0].tolist(), "radar_yaw_rate": float(w[0])}


def main():
    from methods.radar_model import waveform_library
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="ball_person")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--waveform", default="track")
    args = ap.parse_args()
    wf = waveform_library(paths.load_config())[args.waveform]
    objs, static, pose = scenario(args.scenario)
    sim = Renderer(objs, static, pose)
    t, ms = 0.0, []
    while t < args.seconds:
        t0 = time.perf_counter()
        x, truth = sim.frame(wf, t)
        ms.append(1e3 * (time.perf_counter() - t0))
        if len(ms) % 10 == 1:
            print(f"t {truth['t']:5.2f} s: " + "; ".join(f"{o['name']} R {o['range_m']:.2f} m th {o['theta_deg']:+.1f} "
                                                       f"vr {o['vr_mps']:+.2f}" for o in truth["objects"]))
        t += wf.frame_period_s
    print(f"{len(ms)} frames, {np.median(ms):.0f} ms per frame to render")


if __name__ == "__main__":
    main()
