"""
simulation.py - static point reflectors with exact ground truth rendered as raw real ADC frames, random scenes for
the Monte Carlo studies, and a whole session plan as simulated recordings to rehearse a measurement day.

    python simulation.py --session SIM_PLAN
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path as _Path

import numpy as np

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from estimation.core.radar import C0, RadarConfig, range_window  # noqa: E402
from estimation.core.truth import EL_SIGN, complete  # noqa: E402

SCENARIOS = ("single", "multi")


@dataclass
class Reflector:
    range_m: float
    azimuth_deg: float
    snr_db: float
    phase: float = 0.0
    elevation_deg: float = 0.0


@dataclass
class Scene:
    scenario: str
    snr_db: float
    reflectors: list = field(default_factory=list)
    cal_phase_error_rad: np.ndarray | None = None

    @property
    def truth(self) -> np.ndarray:
        return np.array([[r.range_m, r.azimuth_deg] for r in self.reflectors], float).reshape(-1, 2)


def draw_scene(scenario: str, snr_db: float, rng: np.random.Generator, cfg: RadarConfig,
               cal_phase_std_deg: float = 0.0) -> Scene:
    dR = cfg.range_res_m
    phase = lambda: float(rng.uniform(0.0, 2.0 * np.pi))
    if scenario == "single":
        refl = [Reflector(float(rng.uniform(1.0, 4.0)), float(rng.uniform(-45.0, 45.0)), snr_db, phase())]
    elif scenario == "multi":
        r1, th1 = float(rng.uniform(1.2, 3.5)), float(rng.uniform(-35.0, 35.0))
        sep = float(rng.uniform(15.0, 30.0)) * float(rng.choice([-1.0, 1.0]))
        th2 = th1 + sep if abs(th1 + sep) <= 50.0 else th1 - sep
        r2 = r1 + float(rng.uniform(-0.25, 0.25)) * dR
        while True:
            r3 = float(rng.uniform(1.0, 4.0))
            if abs(r3 - r1) >= 4 * dR:
                break
        th3 = float(rng.uniform(-45.0, 45.0))
        refl = [Reflector(r1, th1, snr_db, phase()), Reflector(r2, th2, snr_db - 3.0, phase()),
                Reflector(r3, th3, snr_db - 6.0, phase())]
    else:
        raise ValueError(f"unknown scenario '{scenario}', use one of {SCENARIOS}")
    err = None
    if cal_phase_std_deg > 0:
        err = rng.normal(0.0, np.radians(cal_phase_std_deg), cfg.n_virt)
    return Scene(scenario, float(snr_db), refl, err)


def amplitude_for_snr(snr_db: float, cfg: RadarConfig, sigma_w: float = 1.0) -> float:
    w = range_window(cfg.n_samples).astype(float)
    gain = (w.sum() ** 2 / 4.0) / (w ** 2).sum()
    return float(np.sqrt(10 ** (snr_db / 10) * sigma_w ** 2 / (gain * cfg.n_loops)))


def _signal(scene: Scene, cfg: RadarConfig, az_sign: int, sigma_w: float) -> np.ndarray:
    n = np.arange(cfg.n_samples)[:, None]
    cols = cfg.x_lam[None, :]
    x = np.zeros((cfg.n_samples, cfg.n_virt))
    for r in scene.reflectors:
        a = amplitude_for_snr(r.snr_db, cfg, sigma_w)
        fb = 2.0 * cfg.slope_hz_per_s * r.range_m / C0
        psi = 2 * np.pi * az_sign * cols * np.sin(np.radians(r.azimuth_deg))
        psi = psi + cfg.z_lam[None, :] * EL_SIGN * 2 * np.pi * np.sin(np.radians(r.elevation_deg))
        if scene.cal_phase_error_rad is not None:
            psi = psi + scene.cal_phase_error_rad[None, :]
        x += a * np.cos(2 * np.pi * fb * n / cfg.fs_hz + r.phase + psi)
    return x


def render_frame(scene: Scene, cfg: RadarConfig, rng: np.random.Generator, az_sign: int = 1,
                 sigma_w: float = 1.0) -> np.ndarray:
    s = _signal(scene, cfg, az_sign, sigma_w).astype(np.float32)
    noise = rng.standard_normal((cfg.n_samples, cfg.n_loops, cfg.n_virt), dtype=np.float32)
    return s[:, None, :] + np.float32(sigma_w) * noise


def render_chirp_average(scene: Scene, cfg: RadarConfig, rng: np.random.Generator,
                         az_sign: int = 1, sigma_w: float = 1.0) -> np.ndarray:
    s = _signal(scene, cfg, az_sign, sigma_w)
    return s + sigma_w / np.sqrt(cfg.n_loops) * rng.standard_normal(s.shape)


def write_take(stem, scene: Scene | None, cfg: RadarConfig, n_frames: int, rng: np.random.Generator,
               scenario: str, adc_scale: float = 40.0, notes: str = "simulated take", targets: list | None = None):
    import json
    from pathlib import Path
    stem = Path(stem)
    empty = Scene(scenario, 0.0, [])
    with open(stem.with_suffix(".bin"), "wb") as f:
        for _ in range(n_frames):
            x = render_frame(scene or empty, cfg, rng)
            q = np.clip(np.round(x * adc_scale), -32768, 32767).astype("<i2")
            c = q.transpose(1, 2, 0).reshape(cfg.n_loops, cfg.n_tx, cfg.n_rx, cfg.n_samples)
            slot = np.empty_like(c)
            slot[:, list(cfg.tx_reorder)] = c
            s = slot.ravel()
            g = np.empty((s.size // 2, 4), "<i2")
            g[:, :2], g[:, 2], g[:, 3] = s.reshape(-1, 2), -1, 0
            f.write(g.tobytes())
    if targets is None:
        targets = [] if scene is None else [{"range_m": r.range_m, "azimuth_deg": r.azimuth_deg, "elevation_deg": r.elevation_deg,
                                             "label": f"simulated, SNR {r.snr_db:g} dB"} for r in scene.reflectors]
    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"name": stem.name, "scenario": scenario, "targets": targets, "notes": notes,
                   "frames": n_frames, "simulated": True}, f, indent=2)


def simulate_session(plan_path, folder, cfg: RadarConfig, seed: int = 20261007, snr_small_db: float = 30.0,
                     snr_big_db: float = 42.0, n_frames: int | None = None, phase_error_deg: float = 0.0):
    import json
    from pathlib import Path
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    err = rng.normal(0.0, np.radians(phase_error_deg), cfg.n_virt) if phase_error_deg > 0 else None
    for take in plan["takes"]:
        targets = [complete(t, plan) for t in take.get("targets", [])]
        refl = [Reflector(t["range_m"], t["azimuth_deg"],
                          (snr_big_db if "big" in t.get("label", "") else snr_small_db) + 40 * np.log10(2.0 / t["range_m"]),
                          float(rng.uniform(0, 2 * np.pi)), t["elevation_deg"]) for t in targets]
        scene = Scene(take["scenario"], max([r.snr_db for r in refl], default=0.0), refl, err) if refl else None
        write_take(folder / take["name"], scene, cfg, n_frames or cfg.n_frames, rng, take["scenario"],
                   notes=f"simulated: {take.get('setup', '')}", targets=targets)
        print(f"  {take['name']:20s} {take['scenario']:12s} " +
              ", ".join(f"{r.range_m:.2f} m {r.azimuth_deg:+.0f} deg {r.snr_db:.1f} dB" for r in refl))


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="simulate a whole recording session from a plan (dry run)")
    ap.add_argument("--plan", default="config/session_plan.json")
    ap.add_argument("--session", default="SIM_PLAN", help="folder name under data/simulated/")
    ap.add_argument("--cfg", default="config/awr2944_doa_perfect.cfg")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--phase-error-deg", type=float, default=0.0)
    ap.add_argument("--snr-small", type=float, default=30.0, help="SNR of a small CR at 2 m [dB] (S2 measured: 62)")
    ap.add_argument("--snr-big", type=float, default=42.0, help="SNR of the big CR at 2 m [dB] (S2 measured: 69)")
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args()
    cfg_ = RadarConfig.from_cfg(a.cfg)
    out = Path(__file__).resolve().parent / "data" / "simulated" / a.session
    print(f"simulated session -> {out}")
    simulate_session(a.plan, out, cfg_, seed=a.seed, snr_small_db=a.snr_small, snr_big_db=a.snr_big,
                     n_frames=a.frames, phase_error_deg=a.phase_error_deg)
