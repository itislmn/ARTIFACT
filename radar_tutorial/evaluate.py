"""
evaluate.py - runs every step offline: decodes recorded DCA1000 replies, replays the UART/UDP timing race, recovers
the synthetic targets from raw bytes, processes the raw recording (cross-checked against estimation's loader) and the
UART point cloud; writes output/predictions.csv and output/cache/results.pkl.

    python evaluate.py
"""
import csv
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial import dataset, dca1000, tlv, uart  # noqa: E402
from radar_tutorial.capture import timing_check  # noqa: E402
from radar_tutorial.cfgfile import Radar  # noqa: E402
from radar_tutorial.cube import frame_from_bytes, read_capture, to_cube, to_samples  # noqa: E402
from radar_tutorial.dsp import point_cloud, range_azimuth, range_doppler, range_fft  # noqa: E402
from radar_tutorial.pointcloud import load  # noqa: E402
from radar_tutorial.simulate import raw_bytes  # noqa: E402

CACHE = dataset.OUTPUT / "cache" / "results.pkl"
U = np.linspace(-1, 1, 256, endpoint=False)


def protocols():
    replies = [dict(dca1000.decode_reply(bytes.fromhex(h)), hex=h) for h in dca1000.RECORDED_REPLIES]
    timing = [timing_check(uart.naive_read, 1.0, padding_s=2.0), timing_check(uart.quiet_read, 0.05, padding_s=10.0)]
    return {"replies": replies, "timing": timing}


def match(found, truth, radar):
    out, used = [], set()
    for r, v, az, amp in truth:
        near = [i for i, p in enumerate(found) if i not in used and abs(p[0] - r) <= 3 * radar.range_res_m
                and abs(p[1] - v) <= 2 * radar.v_res_mps]
        best = min(near, key=lambda i: abs(found[i][2] - az)) if near else None
        used.add(best)
        out.append((r, v, az, None if best is None else found[best]))
    return out


def synthetic(radar):
    rows, parsed_ok = [], True
    for name, targets in dataset.SCENES.items():
        frame = to_cube(to_samples(np.frombuffer(raw_bytes(radar, targets, snr_db=dataset.SCENE_SNR_DB), "<i2"),
                                   True, False), radar)[0]
        pc = point_cloud(frame, radar)
        wire = b"\x00\x11" + tlv.build_frame(0, np.c_[pc[:, 4], pc[:, 3], 0 * pc[:, 0], pc[:, 1]], np.ones(8))
        back = tlv.FrameReader().feed(wire)
        parsed_ok &= len(back) == 1 and np.allclose(back[0]["points"][:, 3], pc[:, 1], atol=1e-6)
        for r, v, az, best in match(pc, targets, radar):
            est = np.full(6, np.nan) if best is None else best
            rows.append({"source": "synthetic", "scene": name, "frame": 0, "range_m": est[0], "velocity_mps": est[1],
                         "azimuth_deg": est[2], "x_m": est[3], "y_m": est[4], "snr_db": est[5], "true_range_m": r,
                         "true_velocity_mps": v, "true_azimuth_deg": az, "detections_in_frame": len(pc)})
    return rows, parsed_ok


def recorded_raw(radar):
    cube = read_capture(dataset.RAW, radar)
    rows = []
    for f in range(cube.shape[0]):
        for r, v, az, x, y, snr in point_cloud(cube[f], radar):
            rows.append({"source": "raw", "scene": dataset.RAW.stem, "frame": f, "range_m": r, "velocity_mps": v,
                         "azimuth_deg": az, "x_m": x, "y_m": y, "snr_db": snr})
    f0 = cube[0]
    maps = {"samples": f0[0, 0], "spectrum": np.fft.fft((f0[0, 0] - f0[0, 0].mean()) * np.hanning(radar.n_samples)),
            "range_profile": np.mean(np.abs(range_fft(f0, radar)) ** 2, axis=(0, 1)),
            "range_doppler": range_doppler(f0, radar), "range_azimuth": range_azimuth(f0, radar, U),
            "frames": cube.shape[0]}
    k = int(np.argmax(maps["range_profile"][3:])) + 3
    maps["peak_bin"] = k
    maps["peak_snapshot"] = range_fft(f0, radar)[:, :, k].mean(axis=0)
    return rows, maps, verified(radar, cube), stream_check(radar, cube)


def verified(radar, cube):
    try:
        from estimation.core.radar import RadarConfig, open_recording
    except ImportError:
        return np.nan
    cfg = RadarConfig.from_cfg(dataset.CFG)
    _, get = open_recording(dataset.RAW, cfg)
    return float(max(np.abs(get(f).transpose(1, 2, 0) - cube[f]).max() for f in range(cube.shape[0])))


def stream_check(radar, cube):
    data = dataset.RAW.read_bytes()
    frames = {}
    assemble = dca1000.FrameAssembler(radar.bytes_per_frame, lambda i, b: frames.setdefault(i, b))
    lost = 5
    for seq, i in enumerate(range(0, len(data), dca1000.PACKET_BYTES), start=1):
        packet = struct.pack("<I", seq) + i.to_bytes(6, "little") + data[i:i + dca1000.PACKET_BYTES]
        if seq != lost:
            assemble(*dca1000.split_packet(packet)[1:])
    assemble.flush()
    same = [np.array_equal(frame_from_bytes(frames[f], radar), cube[f]) for f in range(1, cube.shape[0])]
    return {"frames": len(frames), "identical_after_loss": int(sum(same)), "of": len(same)}


def recorded_points():
    frames = load(dataset.POINTS)
    rows = []
    for k, fr in enumerate(frames):
        for x, y, z, v in fr["points"]:
            rows.append({"source": "pointcloud", "scene": dataset.POINTS.stem, "frame": fr["frame"],
                         "range_m": float(np.sqrt(x * x + y * y + z * z)), "velocity_mps": v,
                         "azimuth_deg": float(np.degrees(np.arctan2(x, y))), "x_m": y, "y_m": x, "snr_db": np.nan})
    return rows, frames


def run():
    radar = Radar.from_cfg(dataset.CFG)
    dataset.write()
    syn, parsed_ok = synthetic(radar)
    raw, maps, max_diff, stream = recorded_raw(radar)
    pts, frames = recorded_points()
    res = {"radar": radar, "protocols": protocols(), "synthetic": syn, "tlv_roundtrip": parsed_ok, "raw": raw,
           "maps": maps, "loader_max_diff": max_diff, "stream": stream, "points": pts, "uart_frames": frames}
    rows = syn + raw + pts
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(dataset.OUTPUT / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(res))
    return res


if __name__ == "__main__":
    r = run()
    print(f"synthetic {len(r['synthetic'])} targets, raw {len(r['raw'])} points in {r['maps']['frames']} frames, "
          f"UART {len(r['points'])} points, loader difference {r['loader_max_diff']}, TLV round trip {r['tlv_roundtrip']}, "
          f"stream {r['stream']}")
