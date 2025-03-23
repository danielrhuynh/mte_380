import cv2 as cv
import numpy as np
import math
import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Button

# -----------------------------------
# Motor/Servo and Encoder Configurations
# -----------------------------------

# Motor 1 pins (Left)
M1_IN1 = 17
M1_IN2 = 27
M1_EN  = 18  # PWM for speed control

# Motor 2 pins (Right)
M2_IN1 = 22
M2_IN2 = 23
M2_EN  = 19  # PWM for speed control

ENC_A_LEFT  = 5
ENC_A_RIGHT = 13

# MotorDriver class remains unchanged.
class MotorDriver:
    def __init__(self, in1_pin, in2_pin, en_pin):
        self.in1 = DigitalOutputDevice(in1_pin)
        self.in2 = DigitalOutputDevice(in2_pin)
        self.en = PWMOutputDevice(en_pin)
    
    def run(self, speed, forward=True):
        if forward:
            self.in1.off()
            self.in2.on()
        else:
            self.in1.on()
            self.in2.off()
        self.en.value = speed
    
    def stop(self):
        self.in1.off()
        self.in2.off()
        self.en.value = 0

# Initialize motor drivers.
motor_left = MotorDriver(M1_IN1, M1_IN2, M1_EN)
motor_right = MotorDriver(M2_IN1, M2_IN2, M2_EN)

# Encoder callback functions to count ticks.
left_ticks = 0
right_ticks = 0

def left_encoder_pressed():
    global left_ticks
    left_ticks += 1

def right_encoder_pressed():
    global right_ticks
    right_ticks += 1

encoder_left = Button(ENC_A_LEFT, pull_up=True, bounce_time=0.001)
encoder_left.when_pressed = left_encoder_pressed
encoder_right = Button(ENC_A_RIGHT, pull_up=True, bounce_time=0.001)
encoder_right.when_pressed = right_encoder_pressed

# -----------------------------------
# Motor Control Callback (PID Controller)
# -----------------------------------
prev_error = 0
integral_error = 0
last_pid_time = time.time()

def motor_control_callback(error, angle):
    global prev_error, integral_error, last_pid_time
    
    # If error is None, stop the motors.
    if error is None:
        motor_left.stop()
        motor_right.stop()
        integral_error = 0
        return
    
    current_time = time.time()
    dt = current_time - last_pid_time
    last_pid_time = current_time
    if dt < 0.001:
        dt = 0.001
    
    # PID constants – adjust these to suit your robot.
    base_speed = 0.15  # Base motor speed (0 to 1)
    max_speed = 0.3
    min_speed = 0.05
    Kp = 0.0005     # Proportional gain
    Ki = 0           # Integral gain
    Kd = 0      # Derivative gain
    
    p_term = Kp * error
    integral_error += error * dt
    max_integral = 100
    integral_error = max(-max_integral, min(integral_error, max_integral))
    i_term = Ki * integral_error
    derivative = (error - prev_error) / dt
    d_term = Kd * derivative
    prev_error = error
    
    correction = p_term + i_term + d_term
    
    # Apply correction: positive error means the detected line is to the right,
    # so increase left motor speed and decrease right motor speed.
    left_speed = base_speed + correction
    right_speed = base_speed - correction
    left_speed = max(min_speed, min(max_speed, left_speed))
    right_speed = max(min_speed, min(max_speed, right_speed))
    
    print(f"Error: {error:.2f}")
    print(f"dt: {dt:.2f}")
    print(f"P: {p_term:.3f}, I: {i_term:.3f}, D: {d_term:.3f}, Correction: {correction:.3f}")
    print(f"Motor speeds - Left: {left_speed:.2f}, Right: {right_speed:.2f}")
    
    motor_left.run(left_speed, forward=True)
    motor_right.run(right_speed, forward=True)

# -----------------------------------
# Global variables for fallback.
# -----------------------------------
last_error = None
last_angle = None

# -----------------------------------
# Red Line Detection using Bottom Crop
# -----------------------------------
def process_frame(frame, bottom_crop=100):
    # Resize the frame.
    frame = cv.resize(frame, (640, 480))
    height, width = frame.shape[:2]
    center_x = width // 2

   # Convert to HSV and threshold for red.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    mask1 = cv.inRange(hsv, red_lower, red_upper)
    mask2 = cv.inRange(hsv, red_lower2, red_upper2)
    mask = cv.bitwise_or(mask1, mask2)

    red_regions = cv.bitwise_and(frame, frame, mask=mask)
    gray = cv.cvtColor(red_regions, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
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

    # Initialize error and angle.
    error = 0
    angle = 0
    if len(middle_points) > 0:
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
        error = (x_bottom - center_x) + 75  # Adjust "75" as a camera offset if necessary.
        # Compute the angle relative to vertical.
        # Use the direction vector (vx, vy) and compute angle = atan2(vx, vy)
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
    
    return overlay, edges, error, angle
# -----------------------------------
# Main Loop: Capture Video and Control the Robot
# -----------------------------------
def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Set bottom_crop (in pixels) to use from the bottom of the frame.
    bottom_crop = 100

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay, edges, error, angle = process_frame(frame, bottom_crop=bottom_crop)

        # Use the computed error and angle to control the robot.
        # motor_control_callback(error, angle)
        
        cv.imshow("Edges", edges)
        cv.imshow("Overlay", overlay)

        t_end = time.perf_counter()
        processing_time = t_end - t_start
        # Uncomment for performance info:
        print(f"Frame processed in {processing_time * 1000:.2f} ms (FPS: {1/processing_time:.2f})")
        
        if cv.waitKey(1) and 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    motor_left.stop()
    motor_right.stop()

if __name__ == '__main__':
    main()
