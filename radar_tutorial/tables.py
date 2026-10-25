"""
tables.py - the tutorial's tables as booktabs LaTeX in output/tables.tex: radar parameters from the .cfg, every .cfg
command, the DCA1000 commands, the TLV types, the UART/UDP timing check, the synthetic targets and the recordings.

    python tables.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial import dataset, dca1000, tlv  # noqa: E402
from radar_tutorial.cfgfile import MEANING  # noqa: E402
from radar_tutorial.evaluate import CACHE  # noqa: E402


def tex(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(name, cols, header, rows) -> str:
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    return (f"\\newcommand{{\\{name}}}{{%\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n}}\n")


def radar_table(r) -> str:
    s = r.summary()
    rows = [("Sampled bandwidth $B = S N_s / f_s$", f"{s['bandwidth_ghz']:.2f} GHz"),
            (r"Range resolution $\Delta R = c / (2B)$", f"{s['range_res_cm']:.2f} cm"),
            (r"Maximum range (real ADC) $f_s c / (4S)$", f"{s['max_range_m']:.2f} m"),
            (r"Wavelength at the band centre $\lambda$", f"{s['wavelength_mm']:.3f} mm"),
            (r"Same-TX chirp period $T_c = N_\mathrm{TX}(T_\mathrm{idle} + T_\mathrm{ramp})$", f"{s['tx_period_us']:.1f} $\\mu$s"),
            (r"Maximum velocity $\lambda / (4 T_c)$", f"$\\pm${s['v_max_mps']:.2f} m/s"),
            (r"Velocity resolution $\lambda / (2 N_\mathrm{loops} T_c)$", f"{s['v_res_mps']:.3f} m/s"),
            ("Frame rate", f"{s['frame_rate_hz']:.0f} Hz"),
            ("Virtual channels (azimuth row)", f"{s['n_virtual']} ({s['n_azimuth']})"),
            ("Raw data per frame", f"{s['kib_per_frame']:.0f} KiB")]
    return table("tuRadar", "lr", "Quantity & Value", rows)


def cfg_table(r) -> str:
    seen, rows = set(), []
    for line in r.lines:
        cmd = line.split()[0]
        if cmd in seen:
            continue
        seen.add(cmd)
        rows.append((rf"\texttt{{{tex(cmd)}}}", tex(MEANING.get(cmd, ""))))
    return table("tuCfg", "lp{11cm}", "Command & Meaning", rows)


def dca_table() -> str:
    rows = [(f"0x{c:02X}", rf"\texttt{{{tex(n)}}}", tex(m)) for c, (n, m) in sorted(dca1000.COMMANDS.items())]
    return table("tuDca", "llp{9cm}", "Code & Command & Payload / Meaning", rows)


def tlv_table() -> str:
    return table("tuTlv", "rp{11cm}", "Type & Content", [(k, tex(v)) for k, v in tlv.TYPES.items()])


def timing_table(res) -> str:
    rows = [(rf"\texttt{{{tex(t['reader'])}}}", f"{t['port_timeout_s']:g}", t["lines"], f"{t['cfg_seconds']:.1f}",
             f"{t['window_seconds']:.0f}", t["packets"]) for t in res["protocols"]["timing"]]
    return table("tuTiming", "lrrrrr", "Reader & Port timeout (s) & Lines & .cfg sent in (s) & Window (s) & Packets", rows)


def synthetic_table(res) -> str:
    rows = [(tex(s["scene"]), f"{s['true_range_m']:.2f} / {s['range_m']:.3f}",
             f"{s['true_velocity_mps']:+.2f} / {s['velocity_mps']:+.2f}",
             f"{s['true_azimuth_deg']:+.1f} / {s['azimuth_deg']:+.1f}", f"{s['snr_db']:.0f}") for s in res["synthetic"]]
    return table("tuSynthetic", "lrrrr", r"Scene & Range (m) & Velocity (m/s) & Azimuth ($^\circ$) & SNR (dB) \\"
                 r" & true / found & true / found & true / found &", rows)


def recordings_table(res) -> str:
    raw = res["raw"]
    pts = res["points"]
    frames = res["uart_frames"]
    rows = [("Raw DCA1000 frames", res["maps"]["frames"], f"{len(raw) / res['maps']['frames']:.1f}",
             f"{np.median([p['range_m'] for p in raw]):.2f}", f"{res['loader_max_diff']:g}"),
            ("On-chip UART point cloud", len(frames), f"{len(pts) / len(frames):.1f}",
             f"{np.median([p['range_m'] for p in pts]):.2f}", "--")]
    return table("tuRecordings", "lrrrr", "Recording & Frames & Points per frame & Median range (m) & "
                 r"Difference to \texttt{estimation} loader", rows)


def write(res, path=dataset.OUTPUT / "tables.tex"):
    r = res["radar"]
    parts = [radar_table(r), cfg_table(r), dca_table(), tlv_table(), timing_table(res), synthetic_table(res),
             recordings_table(res)]
    names = r"\tuRadar, \tuCfg, \tuDca, \tuTlv, \tuTiming, \tuSynthetic, \tuRecordings"
    path.write_text(f"% \\input{{tables.tex}} once, then {names}\n" + "\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    write(pickle.loads(CACHE.read_bytes()))
    print("output/tables.tex")
