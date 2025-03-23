import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Button
import math

# -----------------------------------
# Motor and Encoder Configurations
# -----------------------------------

# Left Motor pins
M1_IN1 = 17
M1_IN2 = 27
M1_EN  = 18  # PWM for speed control

# Right Motor pins
M2_IN1 = 22
M2_IN2 = 23
M2_EN  = 19  # PWM for speed control

# Encoder pins (active low)
ENC_A_LEFT  = 13   # Left motor encoder
ENC_A_RIGHT = 5    # Right motor encoder

# -----------------------------------
# Motor Driver Class
# -----------------------------------
class MotorDriver:
    def __init__(self, in1_pin, in2_pin, en_pin):
        self.in1 = DigitalOutputDevice(in1_pin)
        self.in2 = DigitalOutputDevice(in2_pin)
        self.en = PWMOutputDevice(en_pin)
    
    def run(self, speed, forward=True):
        """
        :param speed: Float speed in the range [0, 1].
        :param forward: Boolean indicating direction (True=forward, False=reverse).
        """
        # Clamp speed between 0 and 1
        speed = max(min(speed, 1.0), 0.0)
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

# Instantiate motor drivers
motor_left = MotorDriver(M1_IN1, M1_IN2, M1_EN)
motor_right = MotorDriver(M2_IN1, M2_IN2, M2_EN)

# -----------------------------------
# Encoder Handling
# -----------------------------------
left_ticks = 0
right_ticks = 0

def left_encoder_pressed():
    global left_ticks
    left_ticks += 1

def right_encoder_pressed():
    global right_ticks
    right_ticks += 1

# Create encoder Button objects to trigger on rising edge.
encoder_left = Button(ENC_A_LEFT, pull_up=True, bounce_time=0.001)
encoder_left.when_pressed = left_encoder_pressed

encoder_right = Button(ENC_A_RIGHT, pull_up=True, bounce_time=0.001)
encoder_right.when_pressed = right_encoder_pressed

# -----------------------------------
# Drive Straight with PID
# -----------------------------------
def drive_straight(distance_mm):
    """
    Drive straight for 'distance_mm' millimeters using a PID loop
    on the left and right encoder ticks, always going forward.
    Speed will remain between 0 and 1.
    """
    global left_ticks, right_ticks
    
    # Reset encoder counts at the start
    left_ticks = 0
    right_ticks = 0
    
    # Hardware / mechanical constants
    TICKS_PER_REV = 170              # for your encoders
    WHEEL_DIAMETER_MM = 65           # for your wheels
    WHEEL_CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM
    
    # Distance covered per encoder tick
    distance_per_tick = WHEEL_CIRCUMFERENCE_MM / TICKS_PER_REV
    
    # Calculate the total tick count needed
    target_ticks = distance_mm / distance_per_tick
    
    # -----------------------------------
    # PID Gains
    # -----------------------------------
    #
    # A good starting point for "heading control" (keeping left/right in sync)
    # might be something like:
    #
    #   Kp = 0.01   (Proportional gain)
    #   Ki = 0.0    (Integral gain, typically small or zero to start)
    #   Kd = 0.0    (Derivative gain, can help reduce overshoot if needed)
    #
    # These values assume your "error" is in tick counts (left_ticks - right_ticks)
    # and your max speed is 1. If you see the robot drifting left/right or oscillating,
    # tweak these values accordingly.
    #
    # Increase Kp if the robot reacts too weakly to heading error
    # Decrease Kp if the robot oscillates or jitters around center
    #
    Kp = 0.0005     # Proportional gain
    Ki = 0           # Integral gain
    Kd = 0
    
    base_speed = 0.5  # your nominal "straight line" speed: half of max
    
    # For integral/derivative terms (if you choose to use them)
    integral = 0.0
    last_error = 0.0
    
    while True:
        # Average ticks to see how far we've traveled
        avg_ticks = (left_ticks + right_ticks) / 2.0
        
        # Stop if we've reached or exceeded the distance
        if avg_ticks >= target_ticks:
            break
        
        # PID error: difference in encoder counts (left minus right)
        error = left_ticks - right_ticks
        
        # Accumulate integral (if you enable Ki > 0)
        integral += error
        
        # Derivative (if you enable Kd > 0)
        derivative = error - last_error
        last_error = error
        
        # Calculate the correction using the PID terms
        control = (Kp * error) + (Ki * integral) + (Kd * derivative)
        
        # Adjust each motor's speed
        left_speed = base_speed - control
        right_speed = base_speed + control
        
        # Clamp speeds to [0.0, 1.0]
        left_speed = max(min(left_speed, 1.0), 0.0)
        right_speed = max(min(right_speed, 1.0), 0.0)
        
        # Always run motors forward
        motor_left.run(left_speed, forward=False)
        motor_right.run(right_speed, forward=False)
    
    # Stop motors when done
    motor_left.stop()
    motor_right.stop()

# -----------------------------------
# Main Execution
# -----------------------------------
if __name__ == "__main__":
    # Example usage:
    # Drive 300 mm (0.3 m)
    drive_straight(10000)
    time.sleep(1)
