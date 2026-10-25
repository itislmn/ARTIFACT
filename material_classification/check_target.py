r"""
check_target.py - shows what a take detects against the empty room: the strongest change, its range and azimuth, and
the change along range at that azimuth (to tell an object from its stand).

    python check_target.py data/raw/sessionA metal-static/gold8cm_aimed_take1
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from material_classification.dataset import CFG  # noqa: E402
from material_classification.features import FeatureExtractor, background_map  # noqa: E402
from material_classification.utils.micro_doppler import FramePreprocessor, RadarParams, load_channel_cal  # noqa: E402
from material_classification.utils.recording import load_take  # noqa: E402


def middle(path, pre, n):
    st = load_take(path)
    m = len(st) // 2
    return [pre.process(c) for c in st[max(0, m - n // 2): m + n // 2]]


def main(session, takes):
    p = RadarParams.from_cfg(CFG)
    pre = FramePreprocessor(p, 2.6, channel_cal=load_channel_cal())
    fx = FeatureExtractor(p, pre)
    session = Path(session)
    files = sorted((session / "background").glob("*.npy"))
    if not files:
        sys.exit(f"no background takes in {session / 'background'}")
    bg = background_map(f for path in files for f in middle(path, pre, 60))
    rng = fx.range_axis

    print(f"{'take':44s} {'dB':>6s} {'range':>7s} {'az':>5s} | change (dB) vs range at the object azimuth")
    for t in takes:
        w = fx.window(middle(session / (t if t.endswith(".npy") else t + ".npy"), pre, 40), bg)
        k = int(np.argmin(np.abs(rng - w.range_m)))
        a = int(np.argmin(np.abs(fx.geo.uy_grid - np.sin(np.radians(w.azimuth_deg)))))
        lo, hi = max(k - 5, 0), min(k + 6, len(rng))
        print(f"{t[:44]:44s} {w.features['change_db']:6.1f} {w.range_m:6.2f}m {w.azimuth_deg:+5.0f} | "
              + " ".join(f"{r:.2f}:{v:+.0f}" for r, v in zip(rng[lo:hi], w.change_map_db[lo:hi, a])))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
