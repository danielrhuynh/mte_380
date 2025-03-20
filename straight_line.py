import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Button

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
motor1 = MotorDriver(M1_IN1, M1_IN2, M1_EN)
motor2 = MotorDriver(M2_IN1, M2_IN2, M2_EN)

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
# RPM Measurement Function
# -----------------------------------
# Set this constant to your encoder's ticks per revolution.
ENCODER_TICKS_PER_REV = 270

def measure_rpm(duration_sec):
    """
    Runs both motors at full speed for a specified duration and prints the RPM of each motor.
    
    Parameters:
      duration_sec (float): Measurement duration in seconds.
    """
    global left_ticks, right_ticks
    
    # Reset encoder tick counters
    left_ticks = 0
    right_ticks = 0

    # Run both motors at full speed (speed value 1.0)
    motor1.run(0.80, forward=True)
    motor2.run(0.80, forward=True)
    
    print(f"Running motors at full speed for {duration_sec} seconds...")
    start_time = time.time()
    time.sleep(duration_sec)  # Let the motors run for the measurement duration
    end_time = time.time()

    # Stop the motors
    motor1.stop()
    motor2.stop()

    # Calculate RPM for each motor.
    elapsed_time = end_time - start_time
    left_rpm = (left_ticks / ENCODER_TICKS_PER_REV) * (60 / elapsed_time)
    right_rpm = (right_ticks / ENCODER_TICKS_PER_REV) * (60 / elapsed_time)

    print(f"Left Motor RPM: {left_rpm:.2f}")
    print(f"Right Motor RPM: {right_rpm:.2f}")

# -----------------------------------
# Main Execution
# -----------------------------------
if __name__ == "__main__":
    # Measure RPM over a period of, for example, 5 seconds.
    measure_rpm(5)
