import cv2 as cv
import numpy as np
import math
import time
import serial

# ------------------------------------------------------------
# Initialize Serial Port to ESP32
# ------------------------------------------------------------
# Update '/dev/ttyUSB0' to your serial port as needed.
ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=1)
# Clear any existing data in the serial buffers.
ser.reset_input_buffer()
ser.reset_output_buffer()

DRIVE_DEFAULT_TARGET_DISTANCE = -570.0    # desired distance in mm to drive
DRIVE_DEFAULT_MAX_SPEED         = 170     # maximum PWM speed
DRIVE_DEFAULT_MIN_SPEED         = 90     # minimum PWM speed
DRIVE_DEFAULT_KP_DIFF           = 15     # proportional gain for difference error (wheel imbalance)
DRIVE_DEFAULT_KI_DIFF           = 0.0     # integral gain for difference error
DRIVE_DEFAULT_KD_DIFF           = 3     # derivative gain for difference error
DRIVE_DEFAULT_KP_BASE           = 1.4     # gain for scaling base speed based on distance error

# ------------------------------------------------------------
# Function to Clear Serial Buffers
# ------------------------------------------------------------
def clear_serial_buffer():
    ser.reset_input_buffer()
    ser.reset_output_buffer()

def send_open_command(sleep_time):
    """Format and send the DRIVE command over serial."""
    command = f"OPEN\n"
    print("Sending command:", command.strip())
    ser.write(command.encode())
    time.sleep(sleep_time)

def send_drive_command(target_distance, sleep_time):
    """Format and send the DRIVE command over serial."""
    command = f"DRIVE,{target_distance},{DRIVE_DEFAULT_MAX_SPEED},{DRIVE_DEFAULT_MIN_SPEED},{DRIVE_DEFAULT_KP_DIFF},{DRIVE_DEFAULT_KI_DIFF},{DRIVE_DEFAULT_KD_DIFF},{DRIVE_DEFAULT_KP_BASE}\n"
    # command =f"DRIVE,500,170,60,15,0.0,3,1.4\n"
    print("Sending command:", command.strip())
    ser.write(command.encode())
    time.sleep(sleep_time)

send_open_command(1)

send_drive_command(-150, 0.5)