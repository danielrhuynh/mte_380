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

# ------------------------------------------------------------
# Function to Clear Serial Buffers
# ------------------------------------------------------------
def clear_serial_buffer():
    ser.reset_input_buffer()
    ser.reset_output_buffer()

# ------------------------------------------------------------
# Default Motor/PID Parameters
# ------------------------------------------------------------
DEFAULT_KP         = 0.5
DEFAULT_KI         = 0.5
DEFAULT_KD         = 0.0
SLIPPING_FACTOR = 2.8
# ------------------------------------------------------------
# Function to Perform a Turn Using ESP32's Built-in Turn Function
# ------------------------------------------------------------
def turn_degrees(degrees=180):
    clear_serial_buffer()
    
    print(f"Starting {degrees}-degree turn using ESP32's built-in turn function")
    
    command = f"TURN_DEG_{degrees}_{DEFAULT_KP}_{DEFAULT_KI}_{DEFAULT_KD}_{SLIPPING_FACTOR}\n"
    ser.write(command.encode())
    print(f"Sent command: {command.strip()}")
    
    turn_completed = False
    timeout = time.time() + 10  # 10-second timeout
    
    while not turn_completed and time.time() < timeout:
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"ESP32 response: {response}")
            
            if "Turn completed" in response:
                turn_completed = True
                break
        
        time.sleep(0.1)  # Small delay to prevent CPU hogging
    
    if turn_completed:
        print(f"{degrees}-degree turn completed successfully")
    else:
        print(f"{degrees}-degree turn timed out or failed")
        # Emergency stop
        clear_serial_buffer()
    
    return turn_completed

# ------------------------------------------------------------
# Turn 180 wrapper lol
# ------------------------------------------------------------
def turn_180_degrees():
    return turn_degrees(180)

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
def main():
    time.sleep(1)
    
    print("Starting 180-degree turn sequence")
    
    # Perform the 180-degree turn
    success = turn_180_degrees()
    
    print(f"Turn sequence {'successful' if success else 'failed'}")

if __name__ == "__main__":
    main()