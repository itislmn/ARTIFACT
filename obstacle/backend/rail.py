"""
rail.py - the isel three-axis rail through its TMCL controller: positions read continuously and time-stamped,
software limits, a watchdog, measured speed and stopping distance, and the motor parameters the controller really uses.

    python -m backend.rail      (self-test)
"""
from __future__ import annotations

import queue
import struct
import threading
import time
from collections import deque

import numpy as np

MVP, MST, SAP, GAP = 4, 3, 5, 6
ACTUAL_POS, MAX_SPEED, MAX_ACCEL, CURRENT, MICROSTEP = 1, 4, 5, 6, 140
PARAMS = {"microstep": MICROSTEP, "current": CURRENT, "speed": MAX_SPEED, "accel": MAX_ACCEL}
STATUS = {1: "wrong checksum", 2: "invalid command", 3: "wrong type", 4: "invalid value", 5: "EEPROM locked",
          6: "command not available"}


class RailError(RuntimeError):
    pass


class TMCL:
    def __init__(self, port: str, baud: int = 9600, address: int = 1):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.addr = address
        self.lock = threading.Lock()

    def cmd(self, instr: int, typ: int, motor: int, value: int = 0, check: bool = True) -> int:
        pkt = struct.pack(">BBBBi", self.addr, instr, typ, motor, int(value))
        pkt += bytes([sum(pkt) & 0xFF])
        with self.lock:
            for _ in range(2):
                self.ser.reset_input_buffer()
                self.ser.write(pkt)
                reply = self.ser.read(9)
                if len(reply) == 9:
                    break
        if len(reply) != 9:
            raise RailError(f"no reply from the rail controller (instruction {instr}, motor {motor})")
        status = reply[2]
        if check and status not in (100, 101):
            raise RailError(f"rail controller: status {status} ({STATUS.get(status, '?')}) for instruction {instr}, "
                            f"type {typ}, motor {motor}, value {value}")
        return int.from_bytes(reply[4:8], "big", signed=True)

    def close(self):
        try:
            self.ser.close()
        except Exception:                                      # noqa: BLE001
            pass


class Rail(threading.Thread):
    def __init__(self, cfg: dict, dry: str = "", log=print):
        super().__init__(daemon=True, name="rail")
        rc = cfg["rail"]
        self.cfg, self.dry, self.say = rc, dry, log
        self.motors = list(rc["motors"])
        self.k = np.array(rc["steps_per_mm"], float) * 1000.0 * np.array(rc["axis_sign"], float)
        self.limits = np.array([rc["x_mm"], rc["y_mm"], rc["z_mm"]], float) / 1000.0
        self.poll_dt = 1.0 / float(rc["poll_hz"])
        self.watchdog_s = float(rc["watchdog_s"])
        self.pose = np.zeros(3)
        self.t_pose = time.perf_counter()
        self.target = None
        self.moving = False
        self.speed_mps = float(rc["expected_speed_mm_s"]) / 1000.0
        self.stop_m = float(rc["stop_mm"]) / 1000.0
        self.stops: list[dict] = []
        self.log = deque(maxlen=200000)
        self.commands: list[dict] = []
        self.q = queue.Queue()
        self.fed = time.perf_counter()
        self.stop_evt = threading.Event()
        self.error = ""
        self._pending_stop = None
        self.tm = None
        self.params: dict = {}
        self.listeners = []

    def start(self) -> "Rail":
        if not self.dry:
            rc = self.cfg
            self.tm = TMCL(rc["port"], rc["baud"], rc["address"])
            for a, m in enumerate(self.motors):
                self.params["xyz"[a]] = {}
                for name, typ in PARAMS.items():
                    val = rc[name][a] if isinstance(rc[name], list) else rc[name]
                    try:
                        self.tm.cmd(SAP, typ, m, val)
                    except RailError as e:
                        self.say(f"  [rail] WARNING axis {'xyz'[a]} {name} = {val} rejected: {e}")
                    self.params["xyz"[a]][name] = self.tm.cmd(GAP, typ, m, check=False)
            self.say(f"  [rail] in the controller: {self.params}")
            self.pose = self._read_all()
            if np.any(self.pose < self.limits[:, 0] - 0.005) or np.any(self.pose > self.limits[:, 1] + 0.005):
                self.tm.close()
                raise RailError(f"rail at {np.round(1e3 * self.pose, 1)} mm: outside the limits - zero the axes at the "
                                "start position (terminal controller: sethome 0/1/2) or correct [rail] x_mm/y_mm/z_mm")
        self.t_pose = time.perf_counter()
        self.say(f"  [rail] {'DRY (' + self.dry + ')' if self.dry else 'connected'}: antenna at "
                 f"x {1e3 * self.pose[0]:.1f}, y {1e3 * self.pose[1]:.1f}, z {1e3 * self.pose[2]:.1f} mm")
        super().start()
        return self

    def _read(self, axis: int) -> float:
        return self.tm.cmd(GAP, ACTUAL_POS, self.motors[axis]) / self.k[axis]

    def _read_all(self) -> np.ndarray:
        return np.array([self._read(a) for a in range(3)])

    def goto(self, target):
        self.q.put(("goto", np.clip(np.asarray(target, float), self.limits[:, 0], self.limits[:, 1])))

    def stop(self):
        self.q.put(("stop", None))

    def feed(self):
        self.fed = time.perf_counter()

    def close(self):
        self.stop_evt.set()
        self.join(timeout=2)
        if self.tm is not None:
            for m in self.motors:
                try:
                    self.tm.cmd(MST, 0, m, check=False)
                except Exception:                              # noqa: BLE001
                    pass
            self.tm.close()

    def run(self):
        k = 0
        try:
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                self._commands()
                if self.moving and t0 - self.fed > self.watchdog_s:
                    self.say(f"  [rail] WATCHDOG: no decision for {t0 - self.fed:.1f} s - stopping")
                    self._halt_motion("watchdog")
                self._poll(k)
                k += 1
                time.sleep(max(0.0, self.poll_dt - (time.perf_counter() - t0)))
        except Exception as e:                                 # noqa: BLE001 - stop everything, report
            self.error = f"{type(e).__name__}: {e}"
            self.say(f"  [rail] ERROR {self.error} - stopping all axes")
            if self.tm is not None:
                for m in self.motors:
                    try:
                        self.tm.cmd(MST, 0, m, check=False)
                    except Exception:                          # noqa: BLE001
                        pass

    def _commands(self):
        while True:
            try:
                kind, arg = self.q.get_nowait()
            except queue.Empty:
                return
            t = time.perf_counter()
            if kind == "stop":
                if self.moving or self.target is not None:
                    self._halt_motion("planner")
            elif kind == "goto":
                if self.target is not None and np.all(np.abs(arg - self.target) < 0.0005):
                    continue
                for a in range(3):
                    if self.target is None or abs(arg[a] - self.target[a]) >= 0.0005 or abs(arg[a] - self.pose[a]) >= 0.0005:
                        if self.tm is not None:
                            self.tm.cmd(MVP, 0, self.motors[a], int(round(arg[a] * self.k[a])))
                self.target = arg.copy()
                self.moving = bool(np.any(np.abs(self.target - self.pose) > 0.0005))
                self.commands.append({"t": t, "kind": "goto", "target": [round(float(v), 5) for v in arg]})

    def _halt_motion(self, why: str):
        t = time.perf_counter()
        if self.tm is not None:
            for m in self.motors:
                self.tm.cmd(MST, 0, m, check=False)
        self._pending_stop = (t, self.pose.copy(), why)
        self.target = None
        self.commands.append({"t": t, "kind": "stop", "why": why, "at": [round(float(v), 5) for v in self.pose]})

    def _poll(self, k: int):
        t = time.perf_counter()
        prev = self.pose.copy()
        if self.dry == "frozen":
            pass
        elif self.dry == "virtual":
            if self.target is not None:
                step = self.speed_mps * (t - self.t_pose)
                d = self.target - self.pose
                n = float(np.linalg.norm(d))
                self.pose = self.target.copy() if n <= step else self.pose + d / n * step
        else:
            busy = [a for a in range(3) if self.target is not None and abs(self.target[a] - self.pose[a]) > 0.0002]
            axes = busy if (busy and k % 5) else [0, 1, 2]
            stamps = []
            for a in axes:
                t0 = time.perf_counter()
                self.pose[a] = self._read(a)
                stamps.append((a, 0.5 * (t0 + time.perf_counter())))
            t = next((ts for a, ts in stamps if a == 0), float(np.mean([ts for _, ts in stamps])))
        dt = t - self.t_pose
        self.t_pose = t
        self.log.append((t, *self.pose, int(self.moving)))
        for f in self.listeners:
            f(t, self.pose.copy())
        moved = float(np.linalg.norm(self.pose - prev))
        if self.target is not None and np.all(np.abs(self.target - self.pose) <= 0.0005):
            self.moving = False
        elif self.target is not None:
            self.moving = True
        if self.moving and dt > 0 and moved > 0:
            v = moved / dt
            self._speeds = getattr(self, "_speeds", deque(maxlen=50))
            self._speeds.append(v)
            if len(self._speeds) >= 5:
                self.speed_mps = float(np.median(self._speeds))
        if self._pending_stop is not None:
            ts, p0, why = self._pending_stop
            self._still = getattr(self, "_still", 0) + (moved < 1e-5)
            if moved >= 1e-5:
                self._still = 0
            if self._still >= 3 or t - ts > 3.0:
                d = float(np.linalg.norm(self.pose - p0))
                self.stops.append({"t": ts, "why": why, "distance_m": round(d, 5), "time_s": round(t - ts, 3)})
                self.stop_m = max(self.stop_m, d)
                self.moving, self._pending_stop, self._still = False, None, 0

    def pose_at(self, t: float) -> np.ndarray:
        L = list(self.log)[-400:]
        if not L:
            return self.pose.copy()
        ts = np.array([r[0] for r in L])
        P = np.array([r[1:4] for r in L])
        return np.array([np.interp(t, ts, P[:, a]) for a in range(3)])


def selftest() -> bool:
    pkt = struct.pack(">BBBBi", 1, MVP, 0, 0, 806940)
    pkt += bytes([sum(pkt) & 0xFF])
    ref = struct.pack(">BBBBib", 1, MVP, 0, 0, 806940, 0)
    ref = ref[:8] + struct.pack("B", sum(ref[:8]) % 256)
    cfg = {"rail": {"motors": [0, 1, 2], "steps_per_mm": [806.94, 1645.244, 1677.19], "axis_sign": [1, 1, 1],
                    "x_mm": [0, 1250], "y_mm": [-50, 50], "z_mm": [0, 50], "poll_hz": 50, "watchdog_s": 5.0,
                    "expected_speed_mm_s": 200.0, "stop_mm": 5.0}}
    r = Rail(cfg, dry="virtual", log=lambda s: None).start()
    r.goto([2.0, 0.03, 0.0])
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10 and (r.moving or r.target is None):
        r.feed()
        time.sleep(0.02)
    r.close()
    ok = pkt == ref and abs(r.pose[0] - 1.25) < 1e-3 and abs(r.pose[1] - 0.03) < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}  TMCL packet {pkt.hex()} (terminal controller {ref.hex()}); goto x 2.0 m clipped "
          f"to the travel: at {np.round(r.pose, 4)} m")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
