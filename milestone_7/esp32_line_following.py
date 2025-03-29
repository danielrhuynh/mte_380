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
# Default Motor/PID Parameters (values on a 0-1 scale)
# ------------------------------------------------------------
# OFFSET = -140 this makes the track 1 way
OFFSET = -100
ANGLE_GAIN = 19
DEFAULT_BASE_SPEED = 200
# DEFAULT_BASE_SPEED = 175

DEFAULT_MIN_SPEED  = -205
DEFAULT_MAX_SPEED  = 205
DEFAULT_KP         = 1 #0.6 then was 1
# DEFAULT_KI         = 0.05
DEFAULT_KI         = 0
# DEFAULT_KD         = 0.1
# DEFAULT_KD         = 0.35
# DEFAULT_KD         = 0.2
DEFAULT_KD         = 0.15

turn_counter = 0

has_seen_blue = False

# -----------------------------------
# Default DRIVE Parameters (modifiable)
# -----------------------------------
DRIVE_DEFAULT_TARGET_DISTANCE = -570.0    # desired distance in mm to drive
DRIVE_DEFAULT_MAX_SPEED         = 170     # maximum PWM speed
DRIVE_DEFAULT_MIN_SPEED         = 90     # minimum PWM speed
DRIVE_DEFAULT_KP_DIFF           = 15     # proportional gain for difference error (wheel imbalance)
DRIVE_DEFAULT_KI_DIFF           = 0.0     # integral gain for difference error
DRIVE_DEFAULT_KD_DIFF           = 3     # derivative gain for difference error
DRIVE_DEFAULT_KP_BASE           = 1.4     # gain for scaling base speed based on distance error


TURN_DEFAULT_TARGET_DEGREES = 180.0    # degrees to turn (positive = clockwise, negative = counter-clockwise)
TURN_DEFAULT_MAX_SPEED      = 200     # maximum PWM speed
TURN_DEFAULT_MIN_SPEED      = 30  # minimum PWM speed
TURN_DEFAULT_KP_DIFF        = 25   # proportional gain for difference error
TURN_DEFAULT_KI_DIFF        = 0     # integral gain for difference error
TURN_DEFAULT_KD_DIFF        = 15    # derivative gain for difference error
TURN_DEFAULT_KP_BASE        = 750    # gain for scaling base speed based on distance error
# TURN_DEFAULT_KP_BASE        = 750    # gain for scaling base speed based on distance error

def send_turn_command(target_degrees, sleep_time):
    """Format and send the TURN command over serial."""
    command = f"TURN,{target_degrees},{TURN_DEFAULT_MAX_SPEED},{TURN_DEFAULT_MIN_SPEED},{TURN_DEFAULT_KP_DIFF},{TURN_DEFAULT_KI_DIFF},{TURN_DEFAULT_KD_DIFF},{TURN_DEFAULT_KP_BASE}\n"
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

def send_open_command(sleep_time):
    """Format and send the DRIVE command over serial."""
    command = f"OPEN\n"
    print("Sending command:", command.strip())
    ser.write(command.encode())
    time.sleep(sleep_time)

def send_close_command(sleep_time):
    """Format and send the DRIVE command over serial."""
    command = f"CLOSE\no"
    print("Sending command:", command.strip())
    ser.write(command.encode())
    time.sleep(sleep_time)

def send_stop_command(sleep_time):
    """Format and send the DRIVE command over serial."""
    command = f"STOP\n"
    print("Sending command:", command.strip())
    ser.write(command.encode())
    time.sleep(1)

# ------------------------------------------------------------
# (Removed) GPIO MotorDriver and Encoder Setup
# ------------------------------------------------------------
# Previously, motor control was handled via gpiozero.
# That code has been removed so that motor control is now
# implemented on the ESP32 via serial commands.

def detect_blue_line(frame):
    threshold_percent = 0.20 # This is the percentage of blue pixels that are detected in the frame
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    
    blue_mask = cv.inRange(hsv, blue_lower, blue_upper)
    
    # kernel = np.ones((3, 3), np.uint8)
    # blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
    # blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_CLOSE, kernel)
    
    total_pixels = frame.shape[0] * frame.shape[1]
    blue_pixels = cv.countNonZero(blue_mask)
    blue_percentage = (blue_pixels / total_pixels)

    # print(f"Blue pixels: {blue_pixels}")
    # print(f"total_pixels: {total_pixels}")
    blue_detected = blue_percentage > threshold_percent
    
    return blue_detected

# ------------------------------------------------------------
# Red Line Detection using Bottom Crop (unchanged)
# ------------------------------------------------------------
def process_frame(frame, bottom_crop=200):
    global has_seen_blue  # Ad
    is_blue_line_detected = False
    # Resize the frame.
    frame = cv.resize(frame, (640, 480))
    # Crop Image to remove bottom portion
    frame = frame[:-200, :]
    height, width = frame.shape[:2]
    center_x = width // 2

    if not has_seen_blue and detect_blue_line(frame):
        print("Blue line detected")
        is_blue_line_detected = True
        has_seen_blue = True
        return frame, None, 0, 0, False, is_blue_line_detected
    # Convert to HSV and threshold for red.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    mask1 = cv.inRange(hsv, red_lower, red_upper)
    mask2 = cv.inRange(hsv, red_lower2, red_upper2)
    mask = cv.bitwise_or(mask1, mask2)

    # Apply morphological operations to remove noise.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)  # Removes small noise.
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) # Fills small holes.

    red_regions = cv.bitwise_and(frame, frame, mask=mask)
    gray = cv.cvtColor(red_regions, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    
    edges = cv.Canny(blurred, 200, 255)
    # Create an overlay for visualization.
    overlay = frame.copy()

    # Use only the bottom portion of the frame.
    bottom_region_top = height - bottom_crop
    rows = bottom_crop  # number of rows used

    # Allocate arrays for left and right edge positions in the cropped region.
    left_edges = np.full((rows,), np.nan, dtype=np.float32)
    right_edges = np.full((rows,), np.nan, dtype=np.float32)

    # Keep track of the last known left/right values.
    last_left = None
    last_right = None

    # Process each row in the bottom cropped region.
    for idx, y in enumerate(range(bottom_region_top, height)):
        xs = np.where(edges[y, :] > 0)[0]
        if len(xs) >= 2:
            left_edges[idx] = np.min(xs)
            right_edges[idx] = np.max(xs)
            last_left = left_edges[idx]
            last_right = right_edges[idx]
            cv.circle(overlay, (int(np.min(xs)), y), 1, (0, 255, 0), -1)
            cv.circle(overlay, (int(np.max(xs)), y), 1, (0, 255, 0), -1)
        elif len(xs) == 1:
            point = xs[0]
            # Decide if this single point is left or right based on previous known values.
            if last_left is not None and last_right is not None:
                prev_mid = (last_left + last_right) / 2.0
                if point < prev_mid:
                    left_edges[idx] = point
                else:
                    right_edges[idx] = point
            else:
                # Without previous data, use a heuristic based on the frame center.
                if point < width / 2:
                    left_edges[idx] = point
                    last_left = point
                else:
                    right_edges[idx] = point
                    last_right = point
            cv.circle(overlay, (int(point), y), 1, (0, 255, 0), -1)
        # If no edge is detected, the row remains NaN.

    # Interpolate missing left and right edge values.
    valid_left_idx = np.where(~np.isnan(left_edges))[0]
    if len(valid_left_idx) > 0:
        all_idx = np.arange(rows)
        left_edges = np.interp(all_idx, valid_left_idx, left_edges[valid_left_idx])
    valid_right_idx = np.where(~np.isnan(right_edges))[0]
    if len(valid_right_idx) > 0:
        all_idx = np.arange(rows)
        right_edges = np.interp(all_idx, valid_right_idx, right_edges[valid_right_idx])

    # Compute middle points row-by-row.
    middle_points = []
    for idx, y in enumerate(range(bottom_region_top, height)):
        if not np.isnan(left_edges[idx]) and not np.isnan(right_edges[idx]):
            mid_x = (left_edges[idx] + right_edges[idx]) / 2.0
            middle_points.append([mid_x, y])
            cv.circle(overlay, (int(mid_x), y), 1, (255, 0, 255), -1)

    # Initialize error, angle, and detection flag.
    error = 0
    angle = 0
    detected = False
    if len(middle_points) > 0:
        detected = True
        middle_points = np.array(middle_points, dtype=np.float32)
        # Fit a robust line through the middle points.
        [vx, vy, x0, y0] = cv.fitLine(middle_points, cv.DIST_L2, 0, 0.01, 0.01)
        # Extrapolate the line to cover the full height.
        t_top = (0 - y0) / vy
        t_bottom = ((height - 1) - y0) / vy
        x_top = int(x0 + t_top * vx)
        x_bottom = int(x0 + t_bottom * vx)
        cv.line(overlay, (x_top, 0), (x_bottom, height - 1), (255, 0, 0), 2)
        # Compute error using the line's intersection at the bottom (with a camera offset, if needed).
        error = (x_bottom - center_x) + OFFSET  # Adjust "75" as a camera offset if necessary.
        # Compute the angle relative to vertical.
        angle = math.degrees(math.atan2(vx, vy))
        # Adjust angle so that right tilt is positive and left tilt negative.
        if angle < 90:
            angle = -angle
        else:
            angle = 180 - angle

        # Draw an arrow at the bottom of the frame to illustrate the angle.
        pivot = (x_bottom, height - 1)
        arrow_length = 50
        angle_rad = math.radians(angle)
        end_point = (int(pivot[0] + arrow_length * math.sin(angle_rad)),
                     int(pivot[1] - arrow_length * math.cos(angle_rad)))
        cv.arrowedLine(overlay, pivot, end_point, (0, 255, 255), 2)
    else:
        cv.putText(overlay, "No robust red line detected", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv.putText(overlay, f"Error: {error}", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.putText(overlay, f"Angle: {angle:.2f} deg", (10, 90),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return overlay, edges, error, angle, detected, is_blue_line_detected

# ------------------------------------------------------------
# New Function: Send Serial Command to ESP32
# ------------------------------------------------------------
def send_serial_command(error, base_speed=DEFAULT_BASE_SPEED, min_speed=DEFAULT_MIN_SPEED,
                        max_speed=DEFAULT_MAX_SPEED, kp=DEFAULT_KP, ki=DEFAULT_KI, kd=DEFAULT_KD):
    # Format the command string to be parsed by the ESP32.
    # Example command: "MC_243,0.15,0.05,0.3,0.0005,0,0\n"
    command = f"MC_{error},{base_speed},{min_speed},{max_speed},{kp:.3f},{ki:.3f},{kd:.3f}\n"
    ser.write(command.encode())
    print("Sent command:", command)

# ------------------------------------------------------------
# Main Loop: Capture Video and Send Serial Commands to ESP32
# ------------------------------------------------------------
def main():
    global OFFSET
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Set bottom_crop (in pixels) to use from the bottom of the frame.
    bottom_crop = 300 # was 200
    # Variables to store last valid error and angle when a line was detected.
    last_valid_error = None
    last_valid_angle = None

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        overlay, edges, error, angle, detected, is_blue_line_detected = process_frame(frame, bottom_crop=bottom_crop)

        if is_blue_line_detected:
            clear_serial_buffer()
            send_stop_command(1)
            # time.sleep(2)
            send_open_command(2)
            send_drive_command(-50, 3)
            # send_drive_command(-15, 3)
            send_turn_command(7, 1)
            # send_turn_command(15, 1)
            # send_drive_command(150, 3)
            # send_turn_command(-90, 1)
            # time.sleep(3)
            # send_drive_command(100, 3)
            send_drive_command(150, 3)
            # time.sleep(3)
            send_close_command(2)
            # time.sleep(3)
            send_drive_command(-125, 3)
            # time.sleep(3)
            send_turn_command(270, 3)
            # time.sleep(3)
            clear_serial_buffer()
            # OFFSET = 140
            OFFSET *= -1
            continue
        
        # If a robust line is not detected, use the last valid error and angle.
        if not detected and last_valid_error is not None:
            error = last_valid_error
            angle = last_valid_angle
        elif detected:
            last_valid_error = error
            last_valid_angle = angle

        # Optionally, modify error based on the angle (if needed).
        if last_valid_angle and abs(last_valid_angle - angle) < 60:
            # angle was working at 20
            if abs(angle) > 40 and abs(angle) < 70:
                error = error + angle * ANGLE_GAIN

        print(f"Error: {error:.2f}, Angle: {angle:.2f} deg")

        # Clear serial buffers to avoid overfilling before sending new command.
        clear_serial_buffer()
        
        # Send serial command with the current error and motor/PID parameters.
        send_serial_command(error, base_speed=DEFAULT_BASE_SPEED, min_speed=DEFAULT_MIN_SPEED,
                            max_speed=DEFAULT_MAX_SPEED, kp=DEFAULT_KP, ki=DEFAULT_KI, kd=DEFAULT_KD)
        # t_end = time.perf_counter()
        # processing_time = t_end - t_start
        # Uncomment for performance info:
        # print(f"Frame processed in {processing_time * 1000:.2f} ms (FPS: {1/processing_time:.2f})")
        
        # # Uncomment these lines to display the frames if needed.
        # cv.imshow("Edges", edges)
        # cv.imshow("Overlay", overlay)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
    
