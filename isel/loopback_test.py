import serial
import time

# Use the COM port you saw in Device Manager
ser = serial.Serial('COM3', 9600, timeout=1)


def run_loopback_test():
    test_message = "@0v\r"
    print(f"Sending: {test_message.strip()}")

    # Send the data
    ser.write(test_message.encode('ascii'))

    # Wait a tiny bit for the electricity to travel
    time.sleep(0.1)

    # Read what came back
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