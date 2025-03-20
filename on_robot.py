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

# Motor 1 pins
M1_IN1 = 17
M1_IN2 = 27
M1_EN  = 18  # PWM for speed control

# Motor 2 pins
M2_IN1 = 22
M2_IN2 = 23
M2_EN  = 19  # PWM for speed control

ENC_A_LEFT  = 5
ENC_A_RIGHT = 13

# Button pin (active low)
BUTTON1 = 4

# Servo configuration
class HardwarePWMServo:
    def __init__(self, pwmchip=0, pwmchannel=0, period=20000000, min_duty=1500000, max_duty=2000000):
        self.pwmchip = pwmchip
        self.pwmchannel = pwmchannel
        self.period = period
        self.min_duty = min_duty
        self.max_duty = max_duty
        self.base_path = f"/sys/class/pwm/pwmchip{self.pwmchip}"
        self.channel_path = f"{self.base_path}/pwm{self.pwmchannel}"
        self.export_pwm()
        self.setup_pwm()

    def export_pwm(self):
        if not os.path.exists(self.channel_path):
            with open(f"{self.base_path}/export", "w") as f:
                f.write(str(self.pwmchannel))
            time.sleep(0.5)

    def setup_pwm(self):
        self.write_pwm_file("period", self.period)
        center = int((self.min_duty + self.max_duty) / 2)
        self.write_pwm_file("duty_cycle", center)
        self.write_pwm_file("enable", 1)

    def write_pwm_file(self, filename, value):
        with open(f"{self.channel_path}/{filename}", "w") as f:
            f.write(str(value))

    def set_angle(self, angle):
        if angle < 0:
            angle = 0
        elif angle > 180:
            angle = 180
        duty = int(self.min_duty + (angle / 180.0) * (self.max_duty - self.min_duty))
        self.write_pwm_file("duty_cycle", duty)
        print(f"Servo set to {angle}° (duty_cycle: {duty} ns)")

class MotorDriver:
    def __init__(self, in1_pin, in2_pin, en_pin):
        self.in1 = DigitalOutputDevice(in1_pin)
        self.in2 = DigitalOutputDevice(in2_pin)
        self.en = PWMOutputDevice(en_pin)
    
    def run(self, speed, forward=True):
        if forward:
            self.in1.on()
            self.in2.off()
        else:
            self.in1.off()
            self.in2.on()
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

# Create encoder Button objects.
# Setting bounce_time low to capture fast pulses.
encoder_left = Button(ENC_A_LEFT, pull_up=True, bounce_time=0.001)
encoder_left.when_pressed = left_encoder_pressed

encoder_right = Button(ENC_A_RIGHT, pull_up=True, bounce_time=0.001)
encoder_right.when_pressed = right_encoder_pressed

servo = HardwarePWMServo(pwmchip=0, pwmchannel=0, period=20000000, min_duty=1500000, max_duty=2000000)

# -----------------------------------
# Red Line Detection API (Refined Version)
# -----------------------------------
class RedLineDetectionAPI:
    def __init__(self, frame_width=600, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.left_error = None
        self.right_error = None
        
        self.left_history = []
        self.right_history = []
        self.history_size = 10
        self.max_jump = 10
        
        self.last_valid_left = None
        self.last_valid_right = None
        
    def process_frame(self, frame):
        if frame is None:
            return None, None, None, None

        frame = cv.resize(frame, (self.frame_width, self.frame_height))
        frame = frame[:, :480]
        
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_lower_2 = np.array([170, 50, 50])
        red_upper_2 = np.array([180, 255, 255])
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(hsv, red_lower, red_upper)
        mask2 = cv.inRange(hsv, red_lower_2, red_upper_2)
        mask = cv.bitwise_or(mask1, mask2)
        
        red_regions = cv.bitwise_and(frame, frame, mask=mask)
        gray = cv.cvtColor(red_regions, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 200, 255)
        
        lines = cv.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=5,
            maxLineGap=10
        )
        
        bottom_row = self.frame_height
        bottom_region_height = 50
        bottom_region_top = bottom_row - bottom_region_height
        
        leftmost_x = self.frame_width
        rightmost_x = 0
        
        debug_frame = frame.copy()
        lines_detected = False
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Only consider lines crossing near the bottom of the image
                if (bottom_region_top <= y1 <= bottom_row) or (bottom_region_top <= y2 <= bottom_row):
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        intercept = y1 - slope * x1
                        x_at_bottom = int((bottom_row - intercept) / slope) if slope != 0 else x1
                        if 0 <= x_at_bottom <= self.frame_width:
                            lines_detected = True
                            leftmost_x = min(leftmost_x, x_at_bottom)
                            rightmost_x = max(rightmost_x, x_at_bottom)
        
        if not lines_detected:
            cv.putText(debug_frame, "No lines detected", (self.frame_width//2 - 80, self.frame_height//2), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, None, debug_frame, edges
        
        raw_left_error = leftmost_x 
        raw_right_error = self.frame_width - rightmost_x
        
        left_error = self.smooth_value(raw_left_error, self.left_history, self.last_valid_left)
        right_error = self.smooth_value(raw_right_error, self.right_history, self.last_valid_right)
        
        if leftmost_x < self.frame_width:
            self.last_valid_left = left_error
        if rightmost_x > 0:
            self.last_valid_right = right_error
        
        self.left_error = left_error
        self.right_error = right_error
        
        return left_error, right_error, debug_frame, edges
        
    def smooth_value(self, new_value, history, last_valid):
        if new_value is None or new_value <= 0 or new_value >= self.frame_width:
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
            # Tracking runtime
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            left_error, right_error, debug_frame, edges = self.process_frame(frame)
            if callback is not None:
                callback(left_error, right_error)
            
            if show_debug:
                cv.imshow('Edges', edges)
                cv.imshow('Red Line Detection', debug_frame)
            t_end = time.perf_counter()

            # Tracking runtime
            processing_time = t_end - t_start
            print(f"Frame processed in {processing_time * 1000:.2f} ms (FPS: {1/processing_time:.2f})")
            
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
        
        cap.release()
        if show_debug:
            cv.destroyAllWindows()
    
    def get_errors(self):
        return self.left_error, self.right_error

# -----------------------------------
# Motor Control Callback (PID Control)
# TUNE ME!!!
# -----------------------------------
# PID state variables
prev_position_error = 0
integral_error = 0
last_pid_time = time.time()

def motor_control_callback(left_error, right_error):
    global prev_position_error, integral_error, last_pid_time
    
    if left_error is None or right_error is None:
        print("No lines detected - stopping motors")
        motor1.stop()
        motor2.stop()
        integral_error = 0
        return
    
    position_error = left_error - right_error
    
    current_time = time.time()
    dt = current_time - last_pid_time
    last_pid_time = current_time
    
    # Include epsilon term
    if dt < 0.001:
        dt = 0.001
    
    # PID constants - tune these values!
    base_speed = 0.5  # Base speed value (0 to 1)
    Kp = 0.005 # Start low and increase until the robot follows the line but starts to oscillate, back off kp to about 0.6kp at this point
    Kd = 0.002 # Start with Kd = 0.2Kp, increase until oscillations are reduced
    Ki = 0.001 # We really only need this if the robot fails to center on the line,, start with Ki = 0.1Kp and increase until it looks good?
    
    # Calculate P term
    p_term = Kp * position_error
    
    # Calculate I term
    integral_error += position_error * dt
    max_integral = 100
    integral_error = max(-max_integral, min(integral_error, max_integral))
    i_term = Ki * integral_error
    
    # Calculate D term
    derivative = (position_error - prev_position_error) / dt
    d_term = Kd * derivative
    prev_position_error = position_error
    
    # Calculate total correction
    correction = p_term + i_term + d_term
    
    left_speed = base_speed - correction
    right_speed = base_speed + correction
    
    left_speed = max(0, min(1, left_speed))
    right_speed = max(0, min(1, right_speed))
    
    # Debug information
    print(f"Lane errors - Left: {left_error:.1f}px, Right: {right_error:.1f}px")
    print(f"Position error: {position_error:.1f}px")
    print(f"P: {p_term:.3f}, I: {i_term:.3f}, D: {d_term:.3f}, Total: {correction:.3f}")
    print(f"Motor speeds - Left: {left_speed:.2f}, Right: {right_speed:.2f}")
    
    motor1.run(left_speed, forward=False)
    motor2.run(right_speed, forward=False)
    
    # Visualization - for debugging in CLI
    max_bars = 20
    center = max_bars // 2
    position = center + int(correction * 10)
    position = max(0, min(max_bars, position))
    bar = ['|'] * (max_bars + 1)
    bar[position] = 'X'
    print(''.join(bar))
    print(f"{'LEFT':<10} {'CENTER':^10} {'RIGHT':>10}\n")

# -----------------------------------
# Distance Function (WIP
# -----------------------------------
ENCODER_TICKS_PER_REV = 285 # Sus value?
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
        print(left_ticks)
        avg_ticks = (left_ticks + right_ticks) / 2.0
        
        if avg_ticks >= target_encoder:
            break
        time.sleep(0.01)  # Small delay to avoid busy waiting
    
    motor1.stop()
    motor2.stop()
    print(f"Target reached: {distance_traveled:.1f} mm traveled.")

# -----------------------------------
# Main function to run script
# -----------------------------------
if __name__ == "__main__":
    line_detector = RedLineDetectionAPI()
    
    video_source = 0
    
    try:
       
        line_detector.process_video(video_source, callback=motor_control_callback, show_debug=True)
        # drive_distance(50, 0.5) # Drive testing
    finally:
        motor1.stop()
        motor2.stop()
