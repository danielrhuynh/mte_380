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

motor_left = MotorDriver(M1_IN1, M1_IN2, M1_EN)
motor_right = MotorDriver(M2_IN1, M2_IN2, M2_EN)

# Encoder callback functions to count ticks
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

WHEEL_DIAMETER = 0.065 #[m]
TICKS_PER_REVOLUTION = 170
WHEEL_CIRCUMFERENCE = WHEEL_DIAMETER * math.pi

def ticks_to_distance(ticks):
    revolutions = ticks / TICKS_PER_REVOLUTION
    distance = revolutions * WHEEL_CIRCUMFERENCE
    return distance

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
# Acceleration Test with Line Following
# -----------------------------------
def run_acceleration_test(isVeloTest=False):
    print("Starting acceleration test with line following...")
    if isVeloTest:
        print("Velocity test mode: will run for 3 extra seconds after reaching max velocity")
    
    # Initialize camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return None
    
    # Initialize video writer
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output_video = cv.VideoWriter('line_accel_test.avi', fourcc, 30.0, (640, 480))
    
    # Reset encoder ticks
    global left_ticks, right_ticks
    left_ticks = 0
    right_ticks = 0
    
    # Set up variables for PID controller
    prev_error = 0
    integral_error = 0
    last_pid_time = time.time()
    
    # Set up variables for acceleration measurement
    data_points = []
    velocities = []
    stable_count = 0
    stable_threshold = 3
    velocity_variance_threshold = 0.02  # [m/s] threshold for stability
    max_duration = 5.0  # Max test duration
    velocity_stabilized = False
    stabilization_time = None
    extra_runtime = 3.0  # Additional runtime for velocity test
    
    # Record start time
    start_time = time.time()
    last_time = start_time
    last_distance = 0
    
    # Define bottom crop for line detection
    bottom_crop = 100
    
    try:
        print("Running motors at full speed while following the line...")
        while time.time() - start_time < max_duration:
            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame to detect line
            overlay, edges, error, angle = process_frame(frame, bottom_crop=bottom_crop)
            
            # Calculate current time and distances
            current_time = time.time()
            elapsed = current_time - start_time
            
            left_distance = ticks_to_distance(left_ticks)
            right_distance = ticks_to_distance(right_ticks)
            avg_distance = (left_distance + right_distance) / 2
            
            # Calculate instantaneous velocity
            time_delta = current_time - last_time
            if time_delta >= 0.1:  # Poll every 0.1s
                distance_delta = avg_distance - last_distance
                velocity = distance_delta / time_delta if time_delta > 0 else 0
                
                # Add telemetry to the frame
                cv.putText(overlay, f"Time: {elapsed:.2f}s", (10, 120), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(overlay, f"Velocity: {velocity:.3f}m/s", (10, 150), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"Time: {elapsed:.2f}s | Left: {left_ticks} | Right: {right_ticks} | Distance: {avg_distance:.3f}m | Velocity: {velocity:.3f}m/s")
                
                data_points.append({
                    'time': elapsed,
                    'left_ticks': left_ticks,
                    'right_ticks': right_ticks,
                    'avg_distance': avg_distance,
                    'velocity': velocity,
                    'error': error,
                    'angle': angle
                })
                
                velocities.append(velocity)
                last_time = current_time
                last_distance = avg_distance
                
                # Check for velocity stabilization
                if len(velocities) >= 5:
                    recent_velocities = velocities[-5:]
                    avg_velocity = sum(recent_velocities) / len(recent_velocities)
                    variance = max(recent_velocities) - min(recent_velocities)
                    
                    # Replace the current stabilization detection with this version
                    if variance < velocity_variance_threshold:
                        stable_count += 1
                        if stable_count >= stable_threshold and not velocity_stabilized:
                            velocity_stabilized = True
                            stabilization_time = elapsed
                            
                            # Clear debug info
                            print("\n" + "="*40)
                            print(f"Velocity has stabilized at {avg_velocity:.3f}m/s at time {stabilization_time:.2f}s")
                            
                            # If not a velocity test, break the loop
                            if not isVeloTest:
                                print(f"Acceleration test complete, stopping motors.")
                                break
                            else:
                                # For velocity test, continue running
                                print(f"Continuing to run for {extra_runtime} more seconds...")
                                print(f"Will stop at elapsed time: {stabilization_time + extra_runtime:.2f}s")
                                # Do NOT modify max_duration here - it can cause issues
                    else:
                        stable_count = 0
            
            # Apply PID control for line following while maintaining max speed
            # PID constants
            Kp = 0.0005 # Proportional gain (reduced from line following to allow higher speed)
            Ki = 0      # Integral gain
            Kd = 0 # Derivative gain
            
            # Calculate PID terms
            dt = current_time - last_pid_time
            last_pid_time = current_time
            if dt < 0.001:
                dt = 0.001
                
            p_term = Kp * error
            integral_error += error * dt
            max_integral = 100
            integral_error = max(-max_integral, min(integral_error, max_integral))
            i_term = Ki * integral_error
            derivative = (error - prev_error) / dt
            d_term = Kd * derivative
            prev_error = error
            
            correction = p_term + i_term + d_term
            
            # Apply correction while maintaining maximum speed
            base_speed = 0.4 # Full speed for acceleration test
            min_speed = 0.15   # Minimum speed to maintain forward motion
            
            left_speed = base_speed - correction
            right_speed = base_speed + correction
            
            # Ensure speeds stay within bounds
            left_speed = max(min_speed, min(1.0, left_speed))
            right_speed = max(min_speed, min(1.0, right_speed))
            
            motor_left.run(left_speed, forward=True)
            motor_right.run(right_speed, forward=True)
            
            # Add this AFTER the PID control and motor commands, near the end of the loop
            # Dedicated check for extended runtime completion
            if isVeloTest and velocity_stabilized:
                extra_time_elapsed = elapsed - stabilization_time
                # Add visual feedback about remaining time
                cv.putText(overlay, f"Extra time: {extra_time_elapsed:.1f}/{extra_runtime:.1f}s", 
                          (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check if we've completed the extra runtime
                if extra_time_elapsed >= extra_runtime:
                    print(f"Completed {extra_time_elapsed:.2f}s of extra runtime after velocity stabilization")
                    break
            
            # Display and record video
            cv.imshow("Line Following Acceleration Test", overlay)
            output_video.write(overlay)
            
            # Check for key press to exit
            # if cv.waitKey(1) & 0xFF == ord('q'):
            # if 0xFF == ord('q'):
            #     break
            
            # For velocity test, check if we've run the extra time
            if isVeloTest and velocity_stabilized and (elapsed - stabilization_time) >= extra_runtime:
                print(f"Completed extra runtime after velocity stabilization")
                break
                
    finally:
        # Stop motors
        motor_left.stop()
        motor_right.stop()
        
        # Release resources
        cap.release()
        output_video.release()
        cv.destroyAllWindows()
    
    # Calculate results
    max_velocity = max(velocities) if velocities else 0
    
    # Find time to reach 95% of max velocity
    accel_threshold = 0.95 * max_velocity
    accel_time = None
    for point in data_points:
        if point['velocity'] >= accel_threshold:
            accel_time = point['time']
            break
    
    # Calculate acceleration
    if accel_time:
        acceleration = max_velocity / accel_time
    else:
        acceleration = 0
    
    return max_velocity, accel_time, acceleration, data_points

def main():
    print("===== Line Following Acceleration Test =====")
    print("Press Enter to start the test (make sure a red line is visible to the camera)")
    try:
        input()
    except KeyboardInterrupt:
        print("Test cancelled")
        return
    
    try:
        results = run_acceleration_test(True)
        if results:
            max_velocity, accel_time, acceleration, data_points = results
            
            print("\n===== Test Results =====")
            print(f"Maximum Velocity: {max_velocity:.3f} m/s")
            print(f"Acceleration Time: {accel_time:.3f} s")
            print(f"Average Acceleration: {acceleration:.3f} m/s²")
            
            # Save data to CSV
            with open('line_accel_data.csv', 'w') as f:
                f.write('Time,Left_Ticks,Right_Ticks,Distance,Velocity,Error,Angle\n')
                for point in data_points:
                    f.write(f"{point['time']:.3f},{point['left_ticks']},{point['right_ticks']}," +
                            f"{point['avg_distance']:.3f},{point['velocity']:.3f}," +
                            f"{point['error']:.3f},{point['angle']:.3f}\n")
            print("Data saved to line_accel_data.csv")
            
        else:
            print("Test failed or no data collected.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        motor_left.stop()
        motor_right.stop()
    except Exception as e:
        print(f"\nError during test: {e}")
        motor_left.stop()
        motor_right.stop()

if __name__ == "__main__":
    main()