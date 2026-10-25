"""
live_raw.py - the live point cloud from raw DCA1000 data: arms the card, streams endless frames (frameCfg frames 0),
assembles every frame from the UDP packets by byte offset and shows range-azimuth power and the CFAR points; --replay
shows a recorded .bin the same way.

    python live_raw.py --cli COM4            (python live_raw.py --replay recordings/raw_capture.bin)
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial import dca1000, uart  # noqa: E402
from radar_tutorial.cfgfile import Radar, read_lines  # noqa: E402
from radar_tutorial.cube import frame_from_bytes, read_capture  # noqa: E402
from radar_tutorial.dsp import point_cloud, range_azimuth  # noqa: E402

U = np.linspace(-1, 1, 128, endpoint=False)


def endless(lines):
    return [" ".join(v[:4] + ["0"] + v[5:]) if v[0] == "frameCfg" else " ".join(v) for v in map(str.split, lines)]


def stream(cli, cfg_lines, radar, frames: queue.Queue, stop: threading.Event):
    card = dca1000.Dca1000()
    data = dca1000.data_socket()
    dca1000.arm(card)
    assemble = dca1000.FrameAssembler(radar.bytes_per_frame, lambda i, b: frames.put((i, b)) if frames.qsize() < 3 else None)
    thread = threading.Thread(target=dca1000.receive, args=(lambda: data.recvfrom(65536)[0], 1e9, assemble, stop),
                              daemon=True)
    thread.start()
    port = uart.open_cli(cli)
    uart.send_cfg(port, endless(cfg_lines))
    return card, data, port


def show(radar, next_frame, title):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    th = np.degrees(np.arcsin(U))
    img = ax1.pcolormesh(th, radar.range_axis_m, np.zeros((radar.n_range_bins, U.size)), vmin=-40, vmax=0,
                         shading="nearest", cmap="Blues")
    ax1.set(xlabel="Azimuth (deg)", ylabel="Range (m)", title="Range-Azimuth Power (dB)", ylim=(0, 6))
    pts = ax2.scatter([], [], c=[], cmap="coolwarm", vmin=-radar.v_max_mps, vmax=radar.v_max_mps, s=18)
    ax2.set(xlabel="Sideways y (m)", ylabel="Forward x (m)", xlim=(-3, 3), ylim=(0, 6), title="CFAR Points")
    ax2.set_aspect("equal")
    fig.colorbar(pts, ax=ax2, label="Velocity (m/s)")
    plt.ion()
    plt.show()
    while plt.fignum_exists(fig.number):
        got = next_frame()
        if got is None:
            break
        i, frame = got
        P = range_azimuth(frame, radar, U)
        img.set_array(10 * np.log10(P / P.max() + 1e-12).ravel())
        pc = point_cloud(frame, radar)
        pts.set_offsets(np.c_[pc[:, 4], pc[:, 3]] if len(pc) else np.empty((0, 2)))
        pts.set_array(pc[:, 1] if len(pc) else np.empty(0))
        fig.suptitle(f"{title}: frame {i}, {len(pc)} points")
        plt.pause(0.001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None)
    ap.add_argument("--cfg", default=str(HERE / "config" / "awr2944p.cfg"))
    ap.add_argument("--replay", default=None, help="a recorded .bin instead of the live stream")
    a = ap.parse_args()
    radar = Radar.from_cfg(a.cfg)
    if a.replay:
        cube = read_capture(a.replay, radar)
        it = iter(enumerate(cube))
        show(radar, lambda: (time.sleep(radar.frame_period_s), next(it, None))[1], Path(a.replay).name)
        return
    frames, stop = queue.Queue(), threading.Event()
    card, data, port = stream(a.cli, read_lines(a.cfg), radar, frames, stop)
    try:
        show(radar, lambda: (lambda i_b: (i_b[0], frame_from_bytes(i_b[1], radar)))(frames.get()), "live")
    finally:
        uart.send_cfg(port, ["sensorStop"])
        port.close()
        stop.set()
        dca1000.stop(card)
        card.close()
        data.close()


if __name__ == "__main__":
    main()
