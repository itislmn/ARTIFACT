"""
terminal_controller_3axis.py - an interactive terminal for the three isel axes (TMCL over RS-232): jog, go to, set home,
status, the parameters the controller really uses, speed tests and steps-per-millimetre measurement, recorded macros.

    python terminal_controller_3axis.py      (then help)
"""
import json
import struct
import time

import serial

STATUS = {1: "wrong checksum", 2: "invalid command", 3: "wrong type", 4: "invalid value", 5: "EEPROM locked",
          6: "command not available"}
PARAMS = {"microstep": 140, "current": 6, "speed": 4, "accel": 5}
AXES = {0: "x, the long rail ('back' drives toward the scene)", 1: "y, sideways", 2: "z, vertical"}


class MotorMaster:
    def __init__(self, port='COM3'):
        try:
            self.ser = serial.Serial(port, 9600, timeout=0.1)
        except serial.SerialException as e:
            print(f"ERROR: Connection failed ({e}).")
            raise SystemExit(1)
        self.addr = 1
        self.profiles = {
            0: {'cal': 806.94, 'microstep': 4, 'current': 1500, 'speed': 7000, 'accel': 850},
            1: {'cal': 1645.244, 'microstep': 4, 'current': 1500, 'speed': 7000, 'accel': 850},
            2: {'cal': 1677.19, 'microstep': 4, 'current': 1500, 'speed': 7000, 'accel': 850}
        }
        self.cal = {m: p['cal'] for m, p in self.profiles.items()}
        self.macros = {}
        print("--- INITIALIZING HARDWARE STATES ---")
        for m in [0, 1, 2]:
            for name, typ in PARAMS.items():
                self.send(5, typ, m, self.profiles[m][name])
        self.info()
        print("--- ISEL RADAR MASTER SYSTEM ONLINE ---")

    def send(self, cmd, type_spec, motor, value=0, record=True):
        if cmd == 99:
            if record and recording:
                current_macro.append(('wait', 0, 0, value))
            return 0
        packet = struct.pack('>BBBBi', self.addr, cmd, type_spec, motor, int(value))
        packet += bytes([sum(packet) % 256])
        self.ser.reset_input_buffer()
        self.ser.write(packet)
        reply = self.ser.read(9)
        if record and recording:
            current_macro.append((cmd, type_spec, motor, value))
        if len(reply) != 9:
            print(f"  no reply (instruction {cmd}, type {type_spec}, motor {motor})")
            return 0
        if reply[2] not in (100, 101):
            print(f"  REJECTED: instruction {cmd}, type {type_spec}, motor {motor}, value {int(value)}: "
                  f"{STATUS.get(reply[2], reply[2])}")
        return struct.unpack('>i', reply[4:8])[0]

    def get(self, m, typ):
        return self.send(6, typ, m, 0, record=False)

    def get_pos(self, m):
        return self.get(m, 1)

    def info(self):
        for m in [0, 1, 2]:
            vals = ", ".join(f"{name} {self.get(m, typ)}" for name, typ in PARAMS.items())
            print(f"Motor {m} ({AXES[m]}): {vals}, {self.cal[m]} steps/mm")

    def wait_still(self, m, timeout=120.0):
        t0, track = time.perf_counter(), []
        while time.perf_counter() - t0 < timeout:
            p = self.get_pos(m)
            track.append((time.perf_counter(), p))
            if p == self.get(m, 0) and len(track) > 2 and track[-2][1] == p:
                break
            time.sleep(0.03)
        return track

    def speed_test(self, m, mm):
        for d in (mm, -mm):
            p0 = self.get_pos(m)
            self.send(4, 1, m, d * self.cal[m], record=False)
            tr = self.wait_still(m)
            t = [a for a, _ in tr]
            v = [abs(b2 - b1) / (a2 - a1) / self.cal[m] for (a1, b1), (a2, b2) in zip(tr, tr[1:]) if a2 > a1]
            top = sorted(v)[int(0.9 * (len(v) - 1))] if v else 0.0
            print(f"  motor {m}: {(tr[-1][1] - p0) / self.cal[m]:+.2f} mm in {t[-1] - t[0]:.2f} s, "
                  f"top speed {top:.1f} mm/s")
        print("  back at the start: is the carriage exactly on your pencil mark? (no hum, no jerk = keep this speed)")


recording = False
current_macro = []
bot = MotorMaster()


def cli_help():
    print("""
======================== COMMAND REPERTOIRE ========================
[BASIC MOTION]
1. fwd <dist_mm>     | 2. back <dist_mm>   (Motor 0, x: 'back' drives toward the scene)
3. right <dist_mm>   | 4. left <dist_mm>   (Motor 1, y)
5. up <dist_mm>      | 6. down <dist_mm>   (Motor 2, z)
7. goto <m> <pos_mm> : Absolute move     8. step <m> <steps> : Raw step move
9. move <x_mm> <y_mm> <z_mm> : Move 3 motors (x = motor 0, y = 1, z = 2)

[STOP COMMANDS]
10. stop <m>          : Stop ONE motor    11. halt            : Stop ALL motors

[SYSTEM & CALIBRATION]
12. sethome <m>      : Zero one axis     13. gohome          : Move all to (0,0,0)
14. status           : Show Pos/MM       15. setcal <m> <v>  : Set Steps/MM per motor
16. micro <m> <0-6>  : Set Microstepping (4=1/16)
17. info             : Parameters the controller really uses
18. measure <m> <mm> : Drive <mm>, type the distance your ruler shows -> steps/mm
19. test <m> <mm>    : Drive <mm> out and back, measured speed in mm/s

[HIGH PERFORMANCE TUNING]
20. speed <m> <val>  : Max Velocity      21. accel <m> <val> : Max Acceleration
22. torque <m> <val> : Motor current (SAP 6)

[RADAR / ML SPECIALS]
23. scan <m> <a> <b> : Sweep motor <m> between MM points <a> and <b>
24. save <name>      : Start recording   25. play <name>     : Run macro
26. shake <m> <amt>  : High-freq vibration for ML sensor testing
27. stoprecord : Stop recording     28. export  : Export macro
29. wait <s> : Waiting time before executing command
====================================================================
""")


cli_help()

while True:
    try:
        inp = input("\nMOTOR-CMD> ").strip().lower().split()
        if not inp:
            continue
        cmd = inp[0]

        move_map = {'fwd': (0, -1), 'back': (0, 1), 'right': (1, 1), 'left': (1, -1), 'up': (2, 1), 'down': (2, -1)}
        if cmd in move_map:
            m, s = move_map[cmd]
            bot.send(4, 1, m, float(inp[1]) * bot.cal[m] * s)

        elif cmd == 'help':
            cli_help()

        elif cmd == 'goto':
            m = int(inp[1])
            bot.send(4, 0, m, float(inp[2]) * bot.cal[m])

        elif cmd == 'step':
            bot.send(4, 1, int(inp[1]), int(inp[2]))

        elif cmd == 'move':
            for m, d in enumerate(map(float, inp[1:4])):
                bot.send(4, 1, m, d * bot.cal[m])

        elif cmd == 'stop':
            bot.send(3, 0, int(inp[1]), 0)
            print(f"Motor {inp[1]} Stopped.")

        elif cmd == 'halt':
            for i in [0, 1, 2]:
                bot.send(3, 0, i, 0)
            print("SYSTEM WIDE HALT.")

        elif cmd == 'sethome':
            if len(inp) < 2:
                print("Error: You must specify a motor number (e.g., 'sethome 0')")
            else:
                m = int(inp[1])
                bot.send(3, 0, m, 0)
                bot.send(5, 1, m, 0)
                bot.send(5, 0, m, 0)
                print(f"Axis {m} zeroed.")

        elif cmd == 'gohome':
            for i in [0, 1, 2]:
                bot.send(4, 0, i, 0)

        elif cmd == 'status':
            for i in [0, 1, 2]:
                p = bot.get_pos(i)
                print(f"Motor {i}: {p} steps ({p / bot.cal[i]:.2f} mm) [Factor: {bot.cal[i]}]")

        elif cmd == 'setcal':
            bot.cal[int(inp[1])] = float(inp[2])
            print(f"Motor {inp[1]} factor set to {inp[2]}")

        elif cmd == 'info':
            bot.info()

        elif cmd == 'measure':
            m, mm = int(inp[1]), float(inp[2])
            p0 = bot.get_pos(m)
            bot.send(4, 1, m, mm * bot.cal[m], record=False)
            bot.wait_still(m)
            steps = bot.get_pos(m) - p0
            real = float(input(f"  {steps} steps commanded as {mm} mm - distance on your ruler (mm): "))
            print(f"  motor {m}: {abs(steps) / real:.3f} steps/mm (now {bot.cal[m]}); enter it with "
                  f"'setcal {m} {abs(steps) / real:.3f}' and in obstacle/config.toml [rail] steps_per_mm")

        elif cmd == 'test':
            bot.speed_test(int(inp[1]), float(inp[2]))

        elif cmd == 'speed':
            bot.send(5, 4, int(inp[1]), int(inp[2]))
            print(f"  in the controller: {bot.get(int(inp[1]), 4)}")
        elif cmd == 'accel':
            bot.send(5, 5, int(inp[1]), int(inp[2]))
            print(f"  in the controller: {bot.get(int(inp[1]), 5)}")
        elif cmd == 'torque':
            bot.send(5, 6, int(inp[1]), int(inp[2]))
            print(f"  in the controller: {bot.get(int(inp[1]), 6)}")
        elif cmd == 'micro':
            bot.send(5, 140, int(inp[1]), int(inp[2]))

        elif cmd == 'scan':
            m = int(inp[1])
            a, b = float(inp[2]) * bot.cal[m], float(inp[3]) * bot.cal[m]
            bot.send(4, 0, m, a)
            time.sleep(0.5)
            bot.send(4, 0, m, b)

        elif cmd == 'save':
            recording = True
            current_macro = []
            print("Recording macro...")

        elif cmd == 'play':
            name = inp[1]
            if name in bot.macros:
                print(f"Executing macro: {name}")
                for c, t, m, v in bot.macros[name]:
                    if c == 'wait':
                        print(f"Sleeping {v}s")
                        time.sleep(v)
                    else:
                        bot.send(c, t, m, v, record=False)
                        time.sleep(0.05)
                print("Macro complete.")

        elif cmd == 'stoprecord':
            if recording:
                recording = False
                name = input("Enter name for this macro: ").strip().lower()
                bot.macros[name] = current_macro
                print(f"Macro '{name}' saved with {len(current_macro)} steps.")
            else:
                print("Not currently recording.")

        elif cmd == 'shake':
            m, amt = int(inp[1]), int(inp[2])
            for _ in range(5):
                bot.send(4, 1, m, amt)
                time.sleep(0.1)
                bot.send(4, 1, m, -amt)
                time.sleep(0.1)

        elif cmd == 'export':
            with open('radar_macros.json', 'w') as f:
                json.dump(bot.macros, f)
            print("Macros exported to radar_macros.json")

        elif cmd == 'wait':
            seconds = float(inp[1])
            bot.send(99, 0, 0, seconds)
            time.sleep(seconds)

    except (KeyboardInterrupt, EOFError):
        for i in [0, 1, 2]:
            bot.send(3, 0, i, 0, record=False)
        print("\nAll axes stopped.")
        break
    except Exception as e:
        print(f"Input error: {e}")
