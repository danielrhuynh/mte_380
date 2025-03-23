import cv2 as cv
import numpy as np
import os
import time
from threading import Thread
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from gpiozero import Button

# -----------------------------------
# GPIO and Motor/Servo Configurations
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

# Button pin (active low)
BUTTON1 = 4
# Motor driver remains unchanged.
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

motor1 = MotorDriver(M1_IN1, M1_IN2, M1_EN)
motor2 = MotorDriver(M2_IN1, M2_IN2, M2_EN)

left_ticks = 0
right_ticks = 0

# Encoder callback functions – these are called on each rising edge.
def left_encoder_pressed():
    global left_ticks
    left_ticks += 1

def right_encoder_pressed():
    global right_ticks
    right_ticks += 1

# Create encoder Button objects (bounce_time set low to capture fast pulses).
encoder_left = Button(ENC_A_LEFT, pull_up=True, bounce_time=0.001)
encoder_left.when_pressed = left_encoder_pressed

encoder_right = Button(ENC_A_RIGHT, pull_up=True, bounce_time=0.001)
encoder_right.when_pressed = right_encoder_pressed

# -----------------------------------
# Red Line Detection API (Single Error Version)
# -----------------------------------
# In this version, we detect the leftmost and rightmost edges in the bottom 50 pixels,
# then compute the midline as the average and define the error as:
#     error = midline_x - (cropped_width/2)
# Positive error indicates a shift to the right; negative to the left.
class RedLineDetectionAPI:
    def __init__(self, frame_width=600, frame_height=480, cropped_width=600):
        self.frame_width = frame_width      # Original image width
        self.frame_height = frame_height    # Image height
        self.cropped_width = cropped_width  # Cropped image width for processing
        self.error_history = []
        self.history_size = 10
        self.max_jump = 10
        self.last_valid_error = None
        
    def process_frame(self, frame):
        if frame is None:
            return None, None, None

        # Resize full image and crop horizontally (if needed).
        frame = cv.resize(frame, (self.frame_width, self.frame_height))
        cropped_frame = frame[:, :self.cropped_width]

        # Convert to HSV and threshold for red.
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_lower_2 = np.array([170, 50, 50])
        red_upper_2 = np.array([180, 255, 255])
        hsv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(hsv, red_lower, red_upper)
        mask2 = cv.inRange(hsv, red_lower_2, red_upper_2)
        mask = cv.bitwise_or(mask1, mask2)
        
        red_regions = cv.bitwise_and(cropped_frame, cropped_frame, mask=mask)
        gray = cv.cvtColor(red_regions, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 200, 255)
        
        # Detect lines using HoughLinesP.
        lines = cv.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=5,
            maxLineGap=10
        )
        
        # Define the region near the bottom (last 50 pixels).
        bottom_region_top = self.frame_height - 25
        # Initialize boundaries:
        # Start with leftmost_x at the far right and rightmost_x at the far left.
        leftmost_x = self.cropped_width  
        rightmost_x = 0
        
        debug_frame = cropped_frame.copy()

        # Draw the boundary line that shows our region of interest
        cv.line(debug_frame, (0, bottom_region_top), (self.cropped_width, bottom_region_top), 
                (0, 255, 255), 2)  # Yellow line showing boundary
        cv.putText(debug_frame, "Detection Zone", (self.cropped_width//2 - 80, bottom_region_top - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        lines_detected = False
        
        # Increase detection region for better stability
        bottom_region_top = self.frame_height - 50  # Increased from 25 to 50 pixels
        
        if lines is not None:
            valid_bottom_points = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Consider all lines, but prioritize those in bottom region
                if x2 != x1:  # Avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    # Calculate intersection with bottom of frame
                    try:
                        if abs(slope) < 0.001:  # Nearly horizontal line
                            x_at_bottom = x1  # Use endpoint x
                        else:
                            x_at_bottom = int((self.frame_height - intercept) / slope)
                        
                        # Store point if within reasonable bounds (with some margin)
                        if -50 <= x_at_bottom <= self.cropped_width + 50:
                            # Clamp to valid range for display and calculations
                            x_display = max(0, min(self.cropped_width, x_at_bottom))
                            valid_bottom_points.append((x_at_bottom, x_display))
                            
                            # Lines in bottom region get priority for detection status
                            if (bottom_region_top <= y1 <= self.frame_height) or (bottom_region_top <= y2 <= self.frame_height):
                                lines_detected = True
                    except:
                        continue  # Skip problematic calculations
            
            # If we have points but none in bottom region, still use them
            if valid_bottom_points and not lines_detected:
                lines_detected = True
                
            # Find leftmost and rightmost points from valid points
            if valid_bottom_points:
                # Sort by x coordinate
                valid_bottom_points.sort()
                
                # Get leftmost and rightmost (considering outlier rejection)
                if len(valid_bottom_points) >= 3:
                    # Skip extreme outliers if we have enough points
                    leftmost_x = valid_bottom_points[0][1]  # Use display-safe x value
                    rightmost_x = valid_bottom_points[-1][1]
                else:
                    # Use all points if we don't have many
                    leftmost_x = valid_bottom_points[0][1]
                    rightmost_x = valid_bottom_points[-1][1]
        
        # If no lines were detected, use the last valid error if available.
        if not lines_detected:
            cv.putText(debug_frame, "No lines detected", 
                       (self.cropped_width//2 - 80, self.frame_height//2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.last_valid_error is not None:
                error = self.last_valid_error
            else:
                return None, debug_frame, edges

        else:
            # Compute the midline between the leftmost and rightmost edges.
            mid_line = (leftmost_x + rightmost_x) / 2.0
            # The error is the deviation of this midline from the ideal center.
            error = mid_line - (self.cropped_width / 2)
            # Smooth the error value.
            error = self.smooth_value(error, self.error_history, self.last_valid_error)
            self.last_valid_error = error

        # Draw debugging markers:
        mid_line_int = int((leftmost_x + rightmost_x) / 2.0)
        cv.line(debug_frame, (mid_line_int, 0), (mid_line_int, self.frame_height), (255, 0, 0), 2)
        cv.circle(debug_frame, (leftmost_x, self.frame_height - 10), 5, (255, 0, 0), -1)
        cv.circle(debug_frame, (rightmost_x, self.frame_height - 10), 5, (255, 0, 0), -1)
        cv.putText(debug_frame, f"Error: {error:.2f}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return error, debug_frame, edges

    def smooth_value(self, new_value, history, last_valid):
        if new_value is None:
            return last_valid if last_valid is not None else new_value
        
        if history and abs(new_value - history[-1]) > self.max_jump:
            new_value = history[-1] * 0.7 + new_value * 0.3
        
        history.append(new_value)
        if len(history) > self.history_size:
            history.pop(0)
        
        if history:
            alpha = 0.6
            smoothed = new_value * alpha
            weight = (1 - alpha) / len(history)
            for hist_val in history[:-1]:
                smoothed += hist_val * weight
            return smoothed
        
        return new_value

    def process_video(self, video_source, callback=None, show_debug=True):
        cap = cv.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        while True:
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            error, debug_frame, edges = self.process_frame(frame)
            if callback is not None:
                callback(error)
            
            if show_debug:
                cv.imshow('Edges', edges)
                cv.imshow('Red Line Detection', debug_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
            t_end = time.perf_counter()
            processing_time = t_end - t_start
            # Uncomment for performance info:
            # print(f"Frame processed in {processing_time * 1000:.2f} ms (FPS: {1/processing_time:.2f})")
        
        cap.release()
        if show_debug:
            cv.destroyAllWindows()
    
    def get_error(self):
        return self.last_valid_error

# -----------------------------------
# Motor Control Callback (PID Control)
# -----------------------------------
# PID state variables
prev_error = 0
integral_error = 0
last_pid_time = time.time()

def motor_control_callback(error):
    global prev_error, integral_error, last_pid_time
    
    # If error is None (should not happen due to our modification), stop motors.
    if error is None:
        motor1.stop()
        motor2.stop()
        integral_error = 0
        return
    
    current_time = time.time()
    dt = current_time - last_pid_time
    last_pid_time = current_time
    if dt < 0.001:
        dt = 0.001
    
    # PID constants – adjust these to suit your robot.
    base_speed = 0.2  # Base motor speed (0 to 1)
    max_speed = 0.4
    Kp = 0.0001      # Proportional gain
    Ki = 0           # Integral gain
    Kd = 0           # Derivative gain
    
    p_term = Kp * error
    integral_error += error * dt
    max_integral = 100
    integral_error = max(-max_integral, min(integral_error, max_integral))
    i_term = Ki * integral_error
    derivative = (error - prev_error) / dt
    d_term = Kd * derivative
    prev_error = error
    
    correction = p_term + i_term + d_term
    
    # Apply correction: positive error means the midline is to the right,
    # so reduce the left motor speed and increase the right motor speed.
    left_speed = base_speed - correction
    right_speed = base_speed + correction
    left_speed = max(0, min(max_speed, left_speed))
    right_speed = max(0, min(max_speed, right_speed))
    
    print(f"Error: {error:.2f}")
    print(f"P: {p_term:.3f}, I: {i_term:.3f}, D: {d_term:.3f}, Correction: {correction:.3f}")
    print(f"Motor speeds - Left: {left_speed:.2f}, Right: {right_speed:.2f}")
    
    motor1.run(left_speed, forward=True)
    motor2.run(right_speed, forward=True)

# -----------------------------------
# Distance Function (WIP)
# -----------------------------------
ENCODER_TICKS_PER_REV = 285  # Suspected value
WHEEL_DIAMETER_MM = 65
WHEEL_CIRCUMFERENCE_MM = np.pi * WHEEL_DIAMETER_MM

def drive_distance(distance_mm, speed=0.5):
    """
    Drives the robot forward for a specified distance in mm.
    
    Parameters:
      distance_mm (float): The target distance to drive in millimeters.
      speed (float): The motor speed (0 to 1) to run while driving.
    """
    motor1.run(speed, forward=False)
    motor2.run(speed, forward=False)
    
    target_encoder = ((left_ticks + right_ticks) / 2.0) + (distance_mm * ENCODER_TICKS_PER_REV) / WHEEL_CIRCUMFERENCE_MM
    
    print(f"Driving for {distance_mm} mm...")
    while True:
        avg_ticks = (left_ticks + right_ticks) / 2.0
        if avg_ticks >= target_encoder:
            break
    motor1.stop()
    motor2.stop()
    print("Target reached.")

# -----------------------------------
# Main function to run script
# -----------------------------------
if __name__ == "__main__":
    line_detector = RedLineDetectionAPI()
    video_source = 0  # Change to your video source if needed
    
    try:
        line_detector.process_video(video_source, callback=motor_control_callback, show_debug=True)
        # drive_distance(50, 0.5)  # Uncomment to test driving a specified distance
    finally:
        motor1.stop()
        motor2.stop()
