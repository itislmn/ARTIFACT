"""
loopback_test.py - checks the serial cable alone: with RX and TX bridged, what is sent must come back.

    python loopback_test.py
"""
import serial
import time

ser = serial.Serial('COM3', 9600, timeout=1)


def run_loopback_test():
    test_message = "@0v\r"
    print(f"Sending: {test_message.strip()}")

    ser.write(test_message.encode('ascii'))

    time.sleep(0.1)

    response = ser.read_all().decode('ascii')

    if response == test_message:
        print(f"SUCCESS! Echo received: {response.strip()}")
        print("Your PC, USB adapter, and Cable are WORKING.")
    elif len(response) > 0:
        print(f"PARTIAL SUCCESS: Received something else: {response}")
    else:
        print("FAILURE: Received nothing. The signal is not leaving the computer.")


run_loopback_test()
ser.close()