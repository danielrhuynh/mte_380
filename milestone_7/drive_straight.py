#!/usr/bin/env python3
import serial
import time
import sys

# -----------------------------------
# Default DRIVE Parameters (modifiable)
# -----------------------------------
DEFAULT_TARGET_DISTANCE = 200    # desired distance in mm to drive
DEFAULT_MAX_SPEED         = 170     # maximum PWM speed
DEFAULT_MIN_SPEED         = 60     # minimum PWM speed
DEFAULT_KP_DIFF           = 15     # proportional gain for difference error (wheel imbalance)
DEFAULT_KI_DIFF           = 0.0     # integral gain for difference error
DEFAULT_KD_DIFF           = 3     # derivative gain for difference error
DEFAULT_KP_BASE           = 1.4     # gain for scaling base speed based on distance error


TURN_DEFAULT_TARGET_DEGREES = 180.0    # degrees to turn (positive = clockwise, negative = counter-clockwise)
TURN_DEFAULT_MAX_SPEED      = 180     # maximum PWM speed
TURN_DEFAULT_MIN_SPEED      = 0  # minimum PWM speed
TURN_DEFAULT_KP_DIFF        = 25   # proportional gain for difference error
TURN_DEFAULT_KI_DIFF        = 0     # integral gain for difference error
TURN_DEFAULT_KD_DIFF        = 15    # derivative gain for difference error
TURN_DEFAULT_KP_BASE        = 500    # gain for scaling base speed based on distance error

def open_serial_connection(port, baudrate):
    """Open and return a serial connection."""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Allow time for the connection to establish
        print(f"Serial connection established on {port} at {baudrate} baud.")
        return ser
    except Exception as e:
        print("Failed to open serial connection:", e)
        sys.exit(1)
def send_turn_command(ser, target_degrees, max_speed, min_speed, kp_diff, ki_diff, kd_diff, kp_base):
    """Format and send the TURN command over serial."""
    command = f"TURN,{target_degrees},{max_speed},{min_speed},{kp_diff},{ki_diff},{kd_diff},{kp_base}\n"
    print("Sending command:", command.strip())
    ser.write(command.encode())
    time.sleep(1)

def send_drive_command(ser, target_distance, max_speed, min_speed, kp_diff, ki_diff, kd_diff, kp_base):
    """Format and send the DRIVE command over serial."""
    command = f"DRIVE,{target_distance},{max_speed},{min_speed},{kp_diff},{ki_diff},{kd_diff},{kp_base}\n"
    print("Sending command:", command.strip())
    ser.write(command.encode())
    time.sleep(1)
    # Optionally, you can read and display any response from the ESP32:
    # while ser.in_waiting:
    #    response = ser.readline().decode().strip()
    #    print("Response:", response)

def main():
    print("=== ESP32 DRIVE Command Tuner ===")
    # Change the port as needed (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows)
    ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=1)
    
    # Initialize parameters with default values
    target_distance = DEFAULT_TARGET_DISTANCE
    max_speed = DEFAULT_MAX_SPEED
    min_speed = DEFAULT_MIN_SPEED
    kp_diff = DEFAULT_KP_DIFF
    ki_diff = DEFAULT_KI_DIFF
    kd_diff = DEFAULT_KD_DIFF
    kp_base = DEFAULT_KP_BASE

    while True:
        print("\n--- Current DRIVE Parameters ---")
        print(f"Target Distance (mm) : {target_distance}")
        print(f"Max Speed            : {max_speed}")
        print(f"Min Speed            : {min_speed}")
        print(f"Kp_diff              : {kp_diff}")
        print(f"Ki_diff              : {ki_diff}")
        print(f"Kd_diff              : {kd_diff}")
        print(f"Kp_base              : {kp_base}")
        print("-------------------------------")
        print("Options:")
        print("1. Change parameters")
        print("2. Send DRIVE command")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            # Allow user to change each parameter (press Enter to keep current value)
            target_input = input(f"Enter target distance (mm) [current: {target_distance}]: ").strip()
            if target_input:
                try:
                    target_distance = float(target_input)
                except ValueError:
                    print("Invalid number. Keeping current value.")
            max_input = input(f"Enter max speed [current: {max_speed}]: ").strip()
            if max_input:
                try:
                    max_speed = int(max_input)
                except ValueError:
                    print("Invalid number. Keeping current value.")
            min_input = input(f"Enter min speed [current: {min_speed}]: ").strip()
            if min_input:
                try:
                    min_speed = int(min_input)
                except ValueError:
                    print("Invalid number. Keeping current value.")
            kp_diff_input = input(f"Enter Kp_diff [current: {kp_diff}]: ").strip()
            if kp_diff_input:
                try:
                    kp_diff = float(kp_diff_input)
                except ValueError:
                    print("Invalid number. Keeping current value.")
            ki_diff_input = input(f"Enter Ki_diff [current: {ki_diff}]: ").strip()
            if ki_diff_input:
                try:
                    ki_diff = float(ki_diff_input)
                except ValueError:
                    print("Invalid number. Keeping current value.")
            kd_diff_input = input(f"Enter Kd_diff [current: {kd_diff}]: ").strip()
            if kd_diff_input:
                try:
                    kd_diff = float(kd_diff_input)
                except ValueError:
                    print("Invalid number. Keeping current value.")
            kp_base_input = input(f"Enter Kp_base [current: {kp_base}]: ").strip()
            if kp_base_input:
                try:
                    kp_base = float(kp_base_input)
                except ValueError:
                    print("Invalid number. Keeping current value.")
        elif choice == '2':
            send_drive_command(ser, 500, max_speed, min_speed, kp_diff, ki_diff, kd_diff, kp_base)
            # time.sleep(5)  # Wait for the command to complete
            # send_turn_command(ser, 90, TURN_DEFAULT_MAX_SPEED, TURN_DEFAULT_MIN_SPEED, TURN_DEFAULT_KP_DIFF, TURN_DEFAULT_KI_DIFF, TURN_DEFAULT_KD_DIFF, TURN_DEFAULT_KP_BASE)
            # time.sleep(3)
            # send_drive_command(ser, 500, max_speed, min_speed, kp_diff, ki_diff, kd_diff, kp_base)
            # time.sleep(5)
            # send_turn_command(ser, 90, TURN_DEFAULT_MAX_SPEED, TURN_DEFAULT_MIN_SPEED, TURN_DEFAULT_KP_DIFF, TURN_DEFAULT_KI_DIFF, TURN_DEFAULT_KD_DIFF, TURN_DEFAULT_KP_BASE)
            # time.sleep(3)
            # send_drive_command(ser, 500, max_speed, min_speed, kp_diff, ki_diff, kd_diff, kp_base)
            # time.sleep(5)
            # send_turn_command(ser, 90, TURN_DEFAULT_MAX_SPEED, TURN_DEFAULT_MIN_SPEED, TURN_DEFAULT_KP_DIFF, TURN_DEFAULT_KI_DIFF, TURN_DEFAULT_KD_DIFF, TURN_DEFAULT_KP_BASE)
            # time.sleep(3)
            # send_drive_command(ser, 500, max_speed, min_speed, kp_diff, ki_diff, kd_diff, kp_base)
            # send_drive_command(ser, -target_distance, max_speed, min_speed, kp_diff, ki_diff, kd_diff, kp_base)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            
    ser.close()
    print("Serial connection closed.")

if __name__ == "__main__":
    main()
