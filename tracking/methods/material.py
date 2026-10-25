"""
material.py - the earlier per-track material head, kept as the fallback when a session has no empty-room take for
M2-SRC.

    from methods.material import MaterialHead
"""
from __future__ import annotations

import json

import numpy as np

import paths  # noqa: F401

RCS_REF_DBSM, SNR_REF_DB, R_REF = 10 * np.log10(11.7), 69.0, 2.0
N_SUB = 4
WINDOW_FEATURES = ["rcs_dbsm", "doppler_spread_mps", "range_extent_m", "angle_extent_m", "n_scatterers",
                   "speed_mps", "flicker", "separability", "spectral_slope_db_ghz", "spectral_ripple_db"]
MATERIAL_INPUTS = ["rcs_dbsm", "flicker", "separability", "spectral_slope_db_ghz", "spectral_ripple_db",
                   "range_extent_m", "n_scatterers"]
MOTION_INPUTS = ["doppler_spread_mps", "speed_mps", "flicker", "separability", "range_extent_m", "angle_extent_m",
                 "n_scatterers"]


def rank_one_nmf(M: np.ndarray, iters: int = 60):
    M = np.maximum(np.asarray(M, float), 0.0)
    h = M.mean(axis=1) + 1e-12
    g = np.ones(M.shape[1])
    for _ in range(iters):
        g = np.maximum(M.T @ h / (h @ h), 0.0)
        h = np.maximum(M @ g / max(g @ g, 1e-30), 0.0)
    s = h.mean() + 1e-30
    h, g = h / s, g * s
    res = float(np.linalg.norm(M - np.outer(h, g)) / max(np.linalg.norm(M), 1e-30))
    return h, g, res


class MaterialHead:
    def __init__(self, wf, model_path=paths.MODELS / "material_head.json", window_s: float = 1.6):
        self.wf = wf
        self.window_s = window_s
        self.models = None
        if model_path is not None and model_path.exists():
            from material_classification.classifier import SparseClassifier
            doc = json.loads(model_path.read_text())
            self.models = {k: SparseClassifier.from_state(v) for k, v in doc["heads"].items()}
            self.inputs = doc["inputs"]
        self._sub_cache = {}

    def _subband_dft(self, f_beat: float):
        c = self.wf.cfg
        seg = c.n_samples // N_SUB
        n = np.arange(seg)
        key = round(f_beat, 0)
        if key not in self._sub_cache:
            self._sub_cache = {key: (np.hanning(seg) * np.exp(-2j * np.pi * f_beat * n / c.fs_hz)).astype(np.complex64)}
        return self._sub_cache[key], seg

    def frame_features(self, tr, dets, rd, fe, frame) -> dict | None:
        c = self.wf.cfg
        s = tr.kf.s
        r = float(np.hypot(s[0], s[1]))
        th = float(np.arctan2(s[1], s[0]))
        u = np.sin(th)
        k = int(np.argmin(np.abs(fe.range_m - r)))
        ks = np.arange(max(k - 2, 0), min(k + 3, len(fe.range_m)))
        a = fe.array.steering([u])[:, 0].conj()
        B = np.abs(rd.D[ks][:, :, fe.az] @ a) ** 2 / fe.array.n
        noise = float(np.median(np.abs(rd.D[::2, ::4][:, :, fe.az] @ a) ** 2) / fe.array.n / np.log(2.0))
        thr = noise * np.log(B.size / 1e-3)
        Pv = np.maximum(B - thr, 0.0)
        if Pv.sum() <= 0:
            return None
        w_v = Pv.sum(axis=0)
        v = fe.v_axis
        vm = float(w_v @ v / w_v.sum())
        spread = float(np.sqrt(w_v @ (v - vm) ** 2 / w_v.sum()))
        w_r = Pv.sum(axis=1)
        rr = fe.range_m[ks]
        r_ext = float(np.sqrt(w_r @ (rr - w_r @ rr / w_r.sum()) ** 2 / w_r.sum()))
        peak = float(B.max())
        snr_db = 10 * np.log10(max(peak / noise, 1e-12))
        rcs = snr_db - SNR_REF_DB + RCS_REF_DBSM + 40 * np.log10(max(r, 0.1) / R_REF)
        own = [d for d in dets if d.get("track") == tr.id and d.get("label") in ("core", "part", "birth")]
        xy = np.array([[d["range_m"] * np.cos(np.radians(d["theta_deg"])), d["range_m"] * np.sin(np.radians(d["theta_deg"]))]
                       for d in own]).reshape(-1, 2)
        a_ext = float(np.std(xy @ np.array([-np.sin(th), np.cos(th)]))) if len(xy) > 1 else 0.0
        out = {"rcs_dbsm": float(rcs), "doppler_spread_mps": spread, "range_extent_m": r_ext, "angle_extent_m": a_ext,
               "n_scatterers": float(max(len(own), 1)), "speed_mps": float(np.hypot(s[2], s[3])), "range_m": r}
        if frame is not None:
            f_beat = 2 * c.slope_hz_per_s * r / 299_792_458.0
            dft, seg = self._subband_dft(f_beat)
            x = frame
            d_bin = int(np.argmax(B.max(axis=0)))
            ph = np.exp(-2j * np.pi * (d_bin - fe.K // 2) * np.arange(fe.K) / fe.K).astype(np.complex64)
            cal = fe.cal[fe.az]
            hs = []
            for q in range(N_SUB):
                Y = np.tensordot(dft, x[q * seg:(q + 1) * seg], axes=(0, 0))
                Y = Y[:, fe.az] / cal * fe.tdm[d_bin][fe.az]
                hs.append(abs(complex(ph @ (Y @ a))))
            out["subband"] = hs
        return out

    def window_features(self, feats: list[dict]) -> dict | None:
        if len(feats) < 5:
            return None
        F = {k: np.array([f[k] for f in feats]) for k in ("rcs_dbsm", "doppler_spread_mps", "range_extent_m",
                                                         "angle_extent_m", "n_scatterers", "speed_mps")}
        out = {k: float(np.median(v)) for k, v in F.items()}
        sub = [f["subband"] for f in feats if "subband" in f]
        if len(sub) >= 5:
            M = np.array(sub).T
            h, g, res = rank_one_nmf(M)
            fq = self._subband_freq_ghz()
            hdb = 20 * np.log10(np.maximum(h, 1e-6))
            slope, icpt = np.polyfit(fq, hdb, 1)
            out.update(flicker=float(np.std(g) / max(np.mean(g), 1e-12)), separability=res,
                       spectral_slope_db_ghz=float(slope), spectral_ripple_db=float(np.std(hdb - (slope * fq + icpt))),
                       signature_db=hdb.round(2).tolist())
        else:
            out.update(flicker=np.nan, separability=np.nan, spectral_slope_db_ghz=np.nan, spectral_ripple_db=np.nan)
        return out

    def _subband_freq_ghz(self):
        c = self.wf.cfg
        seg = c.n_samples // N_SUB
        t_mid = c.adc_start_s + (np.arange(N_SUB) + 0.5) * seg / c.fs_hz
        return (c.f0_hz + c.slope_hz_per_s * t_mid) / 1e9

    def classify(self, wfeat: dict) -> dict:
        if self.models is None or wfeat is None:
            return {}
        out = {}
        for head, model in self.models.items():
            x = np.array([[wfeat.get(k, np.nan) for k in self.inputs[head]]], float)
            y = model._scale(x).T
            a = model.code(y)[:, 0]
            res = [np.linalg.norm(y[:, 0] - model.D[:, model.labels == c] @ a[model.labels == c]) for c in model.classes]
            share = np.array([a[model.labels == c].sum() for c in model.classes])
            share = share / max(share.sum(), 1e-12)
            k = int(np.argmin(res))
            out[head] = {"label": str(model.classes[k]), "confidence": float(share[k]),
                         "share": {c: round(float(s), 3) for c, s in zip(model.classes, share)}}
        return out

    def update(self, tracker, rd, fe, frame, t, dets=None):
        self.wf = fe.wf
        dets = dets if dets is not None else []
        if frame is not None and any(tr.confirmed and tr.last_update == t for tr in tracker.tracks):
            frame = np.asarray(frame, np.float32)
            frame = frame - frame.mean(axis=0, keepdims=True)
        for tr in tracker.tracks:
            if not tr.confirmed or tr.last_update != t:
                continue
            f = self.frame_features(tr, dets, rd, fe, frame)
            if f is None:
                continue
            tr.features.append((t, f))
            if t - getattr(tr, "t_classified", -np.inf) < self.window_s / 2:
                continue
            recent = [x for (ti, x) in tr.features if t - ti <= self.window_s]
            wfeat = self.window_features(recent)
            if wfeat is not None:
                tr.window, tr.t_classified = wfeat, t
                cls = self.classify(wfeat)
                if cls:
                    tr.classes = cls
