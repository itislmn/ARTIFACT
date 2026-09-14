import serial
import time

ser = serial.Serial('COM3', 9600, timeout=1)


def send_tmcl(instruction, type_num, motor, value):
    packet = bytearray([0x01, instruction, type_num, motor])
    packet.extend(value.to_bytes(4, byteorder='big', signed=True))
    packet.append(sum(packet) % 256)
    ser.write(packet)
    return ser.read(9)


def release_brake_and_move():
    print("--- TMCM-310 V3.29 Hardware Release ---")

    # 1. SET ALL DIGITAL OUTPUTS TO HIGH (Potential Brake Release)
    # Instruction 14 = SIO (Set Output), Type 2 = Digital Bank, Value 255 (All pins ON)
    print("Attempting to release motor brakes via Digital I/O...")
    send_tmcl(14, 2, 0, 255)
    time.sleep(1)  # Give the relay time to click

    axis = 0  # Testing Y-Axis

    # 2. OPTIMIZE FOR TMCM-310 LIMITS (This board is 1.1A max)
    # Don't go to 255 current, the board might shut down. Use 150 (approx 0.7A)
    send_tmcl(5, 6, axis, 150)
    send_tmcl(5, 140, axis, 2)  # 4 Microsteps for balance of torque/movement

    # 3. ABSOLUTE MOVE
    print("Commanding move... Listen for a 'THUMP' (Brake releasing)")
    send_tmcl(4, 1, axis, 20000)  # Move relative 20,000 steps

    # 4. MONITOR
    for i in range(10):
        res = send_tmcl(6, 1, axis, 0)
        if res:
            pos = int.from_bytes(res[4:8], byteorder='big', signed=True)
            print(f"Current Position: {pos}")
        time.sleep(0.5)


try:
    release_brake_and_move()
finally:
    ser.close()