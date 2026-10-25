"""
capture.py - one raw recording: arm the DCA1000, open the data port, send the .cfg over UART (sensorStart last),
write every packet's payload at its byte offset, stop both; the receive window opens before the radar starts, so no
first frame is lost.

    from radar_tutorial.capture import record, timing_check
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from radar_tutorial import dca1000, uart


def record(cli_port: str, cfg_lines: list[str], seconds: float, out: Path, log=print) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    card = dca1000.Dca1000()
    data = dca1000.data_socket()
    stop = threading.Event()
    stats = {}
    with open(out, "wb") as f:
        def sink(offset, payload):
            f.seek(offset)
            f.write(payload)

        def listen():
            stats.update(dca1000.receive(lambda: data.recvfrom(65536)[0], seconds + 10.0, sink, stop))

        log("DCA1000:")
        dca1000.arm(card, log=log)
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
        log("Radar CLI:")
        with uart.open_cli(cli_port) as port:
            t_cfg = time.time()
            sent = uart.send_cfg(port, cfg_lines, log=log)
            t_start = time.time()
            time.sleep(seconds)
            uart.send_cfg(port, ["sensorStop"], log=log)
        stop.set()
        thread.join()
    dca1000.stop(card)
    card.close()
    data.close()
    stats.update(cfg_seconds=t_start - t_cfg, all_done=all(ok for _, ok, _ in sent), seconds=seconds)
    out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    return stats


def timing_check(reader, port_timeout_s, padding_s, seconds=1.0, adc_bytes=200_000) -> dict:
    lines = [ln for ln in Path(__file__).with_name("config").joinpath("awr2944p.cfg").read_text().splitlines()
             if ln and not ln.startswith("%")]
    port = uart.FakeCli(timeout=port_timeout_s)
    t0 = time.time()
    uart.send_cfg(port, lines, reader, log=None)
    cfg_s = time.time() - t0
    fake = dca1000.FakeDca1000(b"\x01\x00" * (adc_bytes // 2), start_delay_s=cfg_s)
    card = dca1000.Dca1000(fake)
    dca1000.arm(card, log=None)
    got = dca1000.receive(fake.recv_data, seconds + padding_s, lambda o, p: None)
    fake.close()
    return {"reader": reader.__name__, "lines": len(lines), "port_timeout_s": port_timeout_s, "cfg_seconds": cfg_s,
            "window_seconds": seconds + padding_s, "packets": got["packets"], "bytes": got["bytes"]}
