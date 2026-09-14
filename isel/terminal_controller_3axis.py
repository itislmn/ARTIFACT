import serial
import struct
import time
import json

class MotorMaster:
    def __init__(self, port='COM3'):
        try:
            self.ser = serial.Serial(port, 9600, timeout=0.1)
            self.addr = 1

            # 1. INDIVIDUAL MOTOR PROFILES
            # Format: { motor_id: [Calibration, Speed, Torque, Accel] }
            self.profiles = {
                0: {'cal': 806.94, 'speed': 7000, 'torque': 1500, 'accel': 850},  # Y (Base)
                1: {'cal': 1645.244, 'speed': 7000, 'torque': 1500, 'accel': 850},  # X (Pan)
                2: {'cal': 1677.19, 'speed': 7000, 'torque': 1500, 'accel': 850}  # Z (Lift)
            }

            # Independent calibration for each motor: {motor_id: factor}
            self.cal = {0: 806.94, 1: 1645.244, 2: 1677.19}
            self.macros = {}

            # ... existing __init__ code ...
            print("--- INITIALIZING HARDWARE STATES ---")
            for m in [0, 1, 2]:
                self.send(5, 140, m, 4)  # Force 1/16 Microstepping
                self.send(5, 6, m, self.profiles[m]['torque'])  # Force Torque
                self.send(5, 4, m, self.profiles[m]['speed'])  # Force Speed
                self.send(5, 5, m, self.profiles[m]['accel'])  # Force Accel
            print("--- ISEL RADAR MASTER SYSTEM ONLINE ---")
        except:
            print("ERROR: Connection failed.")
            exit()

    def send(self, cmd, type_spec, motor, value):
        # 1. Virtual 'Wait' (Command 99)
        if cmd == 99:
            try:
                if recording:
                    current_macro.append(('wait', 0, 0, value))
            except NameError:
                pass  # recording doesn't exist yet during init
            return 0

        # 2. YOUR ORIGINAL HARDWARE LOGIC (Preserved exactly)
        packet = struct.pack('>BBBBib', self.addr, cmd, type_spec, motor, int(value), 0)
        packet = packet[:8] + struct.pack('B', sum(packet[:8]) % 256)
        self.ser.write(packet)
        reply = self.ser.read(9)

        # 3. RECORDING LOGIC (With 'Safe' check)
        try:
            if recording:
                current_macro.append((cmd, type_spec, motor, value))
        except NameError:
            pass  # Initialization is happening, don't record yet

        return struct.unpack('>i', reply[4:8])[0] if len(reply) == 9 else 0

    def get_pos(self, m):
        return self.send(6, 1, m, 0)

    def move_xyz(dist_x, dist_y, dist_z):
        # Map distances to steps based on your calibration
        steps_x = dist_x * bot.cal[1]
        steps_y = dist_y * bot.cal[0]
        steps_z = dist_z * bot.cal[2]

        # Send the commands back-to-back with NO delays
        # Motor 0 (Y)
        bot.send(4, 1, 0, steps_y)
        # Motor 1 (X)
        bot.send(4, 1, 1, steps_x)
        # Motor 2 (Z)
        bot.send(4, 1, 2, steps_z)

        print(f"Sent simultaneous move: X:{dist_x} Y:{dist_y} Z:{dist_z}")

bot = MotorMaster()

def cli_help():
    print("""
======================== COMMAND REPERTOIRE ========================
[BASIC MOTION]
1. fwd <dist_mm>     | 2. back <dist_mm>   (Motor 0)
3. right <dist_mm>   | 4. left <dist_mm>   (Motor 1)
5. up <dist_mm>      | 6. down <dist_mm>   (Motor 2)
7. goto <m> <pos_mm> : Absolute move     8. step <m> <steps> : Raw step move
9. move <x_mm> <y_mm> <z_mm> : Move 3 motors

[STOP COMMANDS]
10. stop <m>          : Stop ONE motor    11. halt            : Stop ALL motors

[SYSTEM & CALIBRATION]
12. sethome          : Zero all axes     13. gohome          : Move all to (0,0,0)
14. status           : Show Pos/MM       15. setcal <m> <v>  : Set Steps/MM per motor
16. micro <m> <0-8>  : Set Microstepping (4=1/16, 8=1/256)

[HIGH PERFORMANCE TUNING]
17. speed <m> <val>  : Max Velocity      18. accel <m> <val> : Max Acceleration
19. torque <m> <0-2047>: Motor Power (SAP 6)

[RADAR / ML SPECIALS]
20. scan <m> <a> <b> : Sweep motor <m> between MM points <a> and <b>
21. save <name>      : Start recording   22. play <name>     : Run macro
23. shake <m> <amt>  : High-freq vibration for ML sensor testing
24. stoprecord : Stop recording     25. export  : Export macro
26. wait <s> : Waiting time before executing command
====================================================================
""")

cli_help()
recording = False
current_macro = []

while True:
    try:
        inp = input("\nMOTOR-CMD> ").strip().lower().split()
        if not inp: continue
        cmd = inp[0]

        # 1-9: MOTION (Now using independent calibration)
        move_map = {'fwd': (0, 1), 'back': (0, -1), 'right': (1, 1), 'left': (1, -1), 'up': (2, 1), 'down': (2, -1)}
        if cmd in move_map:
            m, s = move_map[cmd]
            steps = float(inp[1]) * bot.cal[m] * s
            bot.send(4, 1, m, steps)
            if recording: current_macro.append(('step', m, steps))

        elif cmd == 'goto':
            m = int(inp[1])
            bot.send(4, 0, m, float(inp[2]) * bot.cal[m])

        elif cmd == 'step':
            bot.send(4, 1, int(inp[1]), int(inp[2]))

        elif cmd == 'move':
            # Usage: move <x_mm> <y_mm> <z_mm>
            # Example: move 50 50 20
            dx, dy, dz = float(inp[1]), float(inp[2]), float(inp[3])

            # Fire all motors
            bot.send(4, 1, 1, dx * bot.cal[1])  # X
            bot.send(4, 1, 0, dy * bot.cal[0])  # Y
            bot.send(4, 1, 2, dz * bot.cal[2])  # Z

        # 10-11: STOPPING
        elif cmd == 'stop':
            bot.send(3, 0, int(inp[1]), 0)
            print(f"Motor {inp[1]} Stopped.")

        elif cmd == 'halt':
            for i in [0, 1, 2]: bot.send(3, 0, i, 0)
            print("SYSTEM WIDE HALT.")

        # 12-16: SYSTEM
        elif cmd == 'sethome':
            # Check if the list has at least 2 items (cmd and motor)
            if len(inp) < 2:
                print("Error: You must specify a motor number (e.g., 'sethome 0')")
            else:
                m = int(inp[1])
                bot.send(3, 0, m, 0)  # Stop
                bot.send(5, 1, m, 0)  # Set Actual to 0
                bot.send(5, 0, m, 0)  # Set Target to 0
                print(f"Axis {m} zeroed.")

        elif cmd == 'gohome':
            for i in [0, 1, 2]: bot.send(4, 0, i, 0)

        elif cmd == 'status':
            for i in [0, 1, 2]:
                p = bot.get_pos(i)
                print(f"Motor {i}: {p} steps ({p / bot.cal[i]:.2f} mm) [Factor: {bot.cal[i]}]")

        elif cmd == 'setcal':
            # Use this to calibrate each motor: setcal <motor_id> <value>
            bot.cal[int(inp[1])] = float(inp[2])
            print(f"Motor {inp[1]} factor set to {inp[2]}")

        # 17-19: TUNING
        elif cmd == 'speed': bot.send(5, 4, int(inp[1]), int(inp[2]))
        elif cmd == 'accel': bot.send(5, 5, int(inp[1]), int(inp[2]))
        elif cmd == 'torque': bot.send(5, 6, int(inp[1]), int(inp[2]))
        elif cmd == 'micro': bot.send(5, 140, int(inp[1]), int(inp[2]))

        # 20-25: RADAR SPECIALS
        elif cmd == 'scan':
            m = int(inp[1])
            a, b = float(inp[2]) * bot.cal[m], float(inp[3]) * bot.cal[m]
            bot.send(4, 0, m, a); time.sleep(0.5); bot.send(4, 0, m, b)

        elif cmd == 'save':
            recording = True; current_macro = []; print("Recording macro...")

        elif cmd == 'play':
            name = inp[1]
            if name in bot.macros:
                print(f"Executing macro: {name}")
                for entry in bot.macros[name]:
                    c, t, m, v = entry
                    if c == 'wait':
                        print(f"Sleeping {v}s")
                        time.sleep(v)
                    else:
                        bot.send(c, t, m, v)
                        time.sleep(0.05)
                print("Macro complete.")

        elif cmd == 'stoprecord':
            if recording:
                recording = False
                # Ask for a name to save the macro under
                name = input("Enter name for this macro: ").strip().lower()
                bot.macros[name] = current_macro
                print(f"Macro '{name}' saved with {len(current_macro)} steps.")
            else:
                print("Not currently recording.")

        elif cmd == 'shake':
            m, amt = int(inp[1]), int(inp[2])
            for _ in range(5):
                bot.send(4, 1, m, amt); time.sleep(0.1)
                bot.send(4, 1, m, -amt); time.sleep(0.1)

        elif cmd == 'export':
            with open('radar_macros.json', 'w') as f:
                json.dump(bot.macros, f)
            print("Macros exported to radar_macros.json")

        elif cmd == 'wait':
            seconds = float(inp[1])
            bot.send(99, 0, 0, seconds)
            time.sleep(seconds)


    except Exception as e:
        print(f"Input error: {e}")