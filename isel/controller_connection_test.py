"""
controller_connection_test.py - checks the TMCM-310 controller: releases the brakes, sets a safe current, moves the
Y axis 20,000 steps and prints its position.

    python controller_connection_test.py
"""
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

    print("Attempting to release motor brakes via Digital I/O...")
    send_tmcl(14, 2, 0, 255)
    time.sleep(1)

    axis = 0

    send_tmcl(5, 6, axis, 150)
    send_tmcl(5, 140, axis, 2)

    print("Commanding move... Listen for a 'THUMP' (Brake releasing)")
    send_tmcl(4, 1, axis, 20000)

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