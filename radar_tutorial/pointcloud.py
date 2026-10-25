"""
pointcloud.py - the on-chip point cloud over UART (no DCA1000): sends the .cfg on the CLI port, reads the TLV frames
on the data port at 921600 baud, shows the points and the range profile live and records them to
recordings/<name>.npz; --replay shows a recording.

    python pointcloud.py --cli COM4 --data COM5 --name walk_1      (python pointcloud.py --replay recordings/pointcloud.npz)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial import tlv, uart  # noqa: E402
from radar_tutorial.cfgfile import read_lines  # noqa: E402


def save(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame_ids=np.array([f["frame"] for f in frames]),
                        num_obj=np.array([f["n_obj"] for f in frames]),
                        points=np.array([f["points"] for f in frames], dtype=object),
                        range_profiles=np.array([f["range_profile"] for f in frames if f["range_profile"] is not None]))


def load(path):
    d = np.load(path, allow_pickle=True)
    prof = d["range_profiles"]
    scale = 512.0 if prof.size and prof.max() > 64 else 1.0
    points = iter(list(d["points"]))
    out = []
    for k, (i, n) in enumerate(zip(d["frame_ids"], d["num_obj"])):
        p = next(points, None) if (n > 0 or len(d["points"]) == len(d["frame_ids"])) else None
        out.append({"frame": int(i), "n_obj": int(n), "points": np.asarray(p, float).reshape(-1, 4) if p is not None
                    else np.zeros((0, 4)), "range_profile": prof[k] / scale if k < len(prof) else None})
    return out


def show(frames_iter, title):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    pts = ax1.scatter([], [], c=[], cmap="coolwarm", vmin=-1, vmax=1, s=20)
    ax1.set(xlabel="x (m)", ylabel="y (m)", xlim=(-4, 4), ylim=(0, 8), title="Detected Points")
    ax1.set_aspect("equal")
    fig.colorbar(pts, ax=ax1, label="Velocity (m/s)")
    (line,) = ax2.plot([], [])
    ax2.set(xlabel="Range Bin", ylabel="Power (dB)", title="Range Profile")
    plt.ion()
    plt.show()
    for f in frames_iter:
        if not plt.fignum_exists(fig.number):
            break
        p = f["points"]
        pts.set_offsets(p[:, :2] if len(p) else np.empty((0, 2)))
        pts.set_array(p[:, 3] if len(p) else np.empty(0))
        if f["range_profile"] is not None:
            db = 20 * np.log10(2) * np.asarray(f["range_profile"], float)
            line.set_data(np.arange(db.size), db)
            ax2.relim()
            ax2.autoscale_view()
        fig.suptitle(f"{title}: frame {f['frame']}, {len(p)} points")
        plt.pause(0.001)


def live(cli, data_port, cfg_lines, seconds):
    import serial
    with uart.open_cli(cli) as port:
        uart.send_cfg(port, cfg_lines)
    reader = tlv.FrameReader()
    t0 = time.time()
    with serial.Serial(data_port, tlv.DATA_BAUD, timeout=0.05) as ser:
        while time.time() - t0 < seconds:
            yield from reader.feed(ser.read(4096))
    with uart.open_cli(cli) as port:
        uart.send_cfg(port, ["sensorStop"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--cfg", default=str(HERE / "config" / "awr2944p.cfg"))
    ap.add_argument("--name", default="pointcloud_live")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--replay", default=None)
    a = ap.parse_args()
    if a.replay:
        frames = load(a.replay)
        show(((time.sleep(0.05), f)[1] for f in frames), Path(a.replay).name)
        return
    got = []
    show((got.append(f) or f for f in live(a.cli, a.data, read_lines(a.cfg), a.seconds)), "live")
    out = HERE / "recordings" / f"{a.name}.npz"
    save(out, got)
    print(f"{len(got)} frames -> {out}")


if __name__ == "__main__":
    main()
