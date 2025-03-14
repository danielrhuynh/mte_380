import cv2 as cv
import numpy as np

class RedLineDetectionAPI:
    def __init__(self, frame_width=480, frame_height=480):
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
        
        # Circle detection parameters
        self.circle_detected = False
        self.circle_distance = None
        self.circle_within_threshold = False
        self.distance_threshold = 10.0  # [cm]
        
        # Circle calibration parameters - for known red circle
        self.known_circle_diameter_cm = 20.0
        self.focal_length = None
        
        # Distance hysteresis parameters
        self.distance_history = []
        self.distance_history_size = 10  # Number of frames to average over
        self.max_distance_jump = 5.0    # Maximum allowed distance change between frames
        self.last_valid_distance = None
        
        # Hysteresis thresholds to prevent flickering
        self.near_threshold = 2.0 # Below = near
        self.far_threshold = 2.5 # Above = far
        
        self.circle_debug_mode = True
    
    # Tool
    # https://math.stackexchange.com/questions/3739230/calculate-the-distance-to-an-object-through-a-pinhole-camera-2-approaches
    def calibrate_focal_length(self, known_distance_cm, pixel_diameter):
        """Calibrate the focal length using a reference image
        known_distance_cm: how far the circle was from camera when reference image was taken
        pixel_diameter: diameter of the circle in pixels in the reference image
        """
        self.focal_length = (pixel_diameter * known_distance_cm) / self.known_circle_diameter_cm
        print(f"Camera calibrated with focal length: {self.focal_length:.2f}")
    
    def process_frame(self, frame):
        """Process a single frame and return left and right errors"""
        if frame is None:
            return None, None, None, None

        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.resize(frame, (self.frame_width, self.frame_height))
        
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        mask1 = cv.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv.inRange(hsv, red_lower2, red_upper2)
        mask = cv.bitwise_or(mask1, mask2)
        
        # Supposed to clean up mask? - lowkey does
        kernel = np.ones((5,5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        red_regions = cv.bitwise_and(frame, frame, mask=mask)
        
        debug_frame = frame.copy()
        self.detect_circles(mask, debug_frame)
        
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
        
        # cv.line(debug_frame, (0, bottom_region_top), (self.frame_width, bottom_region_top), (0, 255, 255), 1)
        # cv.line(debug_frame, (0, bottom_row), (self.frame_width, bottom_row), (0, 255, 255), 1)
        
        lines_detected = False
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                cv.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if (y1 >= bottom_region_top and y1 <= bottom_row) or (y2 >= bottom_region_top and y2 <= bottom_row):
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        intercept = y1 - slope * x1
                        
                        x_at_bottom = int((bottom_row - intercept) / slope) if slope != 0 else x1
                        
                        if 0 <= x_at_bottom <= self.frame_width:
                            lines_detected = True
                            if x_at_bottom < leftmost_x:
                                leftmost_x = x_at_bottom
                            if x_at_bottom > rightmost_x:
                                rightmost_x = x_at_bottom
                            
                            # cv.circle(debug_frame, (x_at_bottom, bottom_row), 5, (0, 0, 255), -1)
        
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
        
        if leftmost_x < self.frame_width:
            smooth_left_x = int(leftmost_x)
            # cv.line(debug_frame, (0, bottom_row), (smooth_left_x, bottom_row), (255, 0, 0), 2)
            # cv.putText(debug_frame, f"L:{left_error:.1f}", (10, bottom_row-10), 
            #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if rightmost_x > 0:
            smooth_right_x = int(self.frame_width - right_error)
            # cv.line(debug_frame, (smooth_right_x, bottom_row), (self.frame_width, bottom_row), (0, 0, 255), 2)
            # cv.putText(debug_frame, f"R:{right_error:.1f}", (rightmost_x+10, bottom_row-10), 
            #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        self.left_error = left_error
        self.right_error = right_error
        
        # Debug text for circle detection
        if self.circle_detected:
            text = f"Circle: {self.circle_distance:.1f}cm"
            color = (0, 0, 255) if self.circle_within_threshold else (0, 255, 0)
            cv.putText(debug_frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if self.circle_within_threshold:
                cv.putText(debug_frame, "ACTUATE GRABBER!", (10, 60), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return left_error, right_error, debug_frame, edges
    
    def detect_circles(self, mask, debug_frame):
        """Detect circles in the red regions and estimate their distance"""
        self.circle_detected = False
        raw_distance = None
        
        blurred = cv.GaussianBlur(mask, (9, 9), 2)
        
        if self.circle_debug_mode:
            cv.imshow('Circle Detection - Mask', mask)
            cv.imshow('Circle Detection - Blurred', blurred)
        
        circles = cv.HoughCircles(
            blurred,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=self.frame_height/4,  
            param1=50,                   
            param2=20,
            minRadius=10,
            maxRadius=int(self.frame_width/2)
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            circle = circles[0][0]
            center_x, center_y, radius = circle[0], circle[1], circle[2]
            
            if self.circle_debug_mode:
                print(f"Found circle via HoughCircles: radius={radius}")
            
            if self.focal_length is None:
                self.focal_length = 300
                
            pixel_diameter = radius * 2
            raw_distance = (self.focal_length * self.known_circle_diameter_cm) / pixel_diameter
            
            self.circle_detected = True
            
            color = (0, 0, 255) if self.circle_within_threshold else (0, 255, 0)
            cv.circle(debug_frame, (center_x, center_y), radius, color, 2)
            cv.circle(debug_frame, (center_x, center_y), 2, color, 3)
            
            distance_text = f"{raw_distance:.1f}cm"
            cv.putText(debug_frame, distance_text, 
                    (center_x - 40, center_y - radius - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.circle_detected and raw_distance is not None:
            self.circle_distance = self.smooth_distance(raw_distance)
            
            if self.circle_distance <= self.near_threshold:
                self.circle_within_threshold = True
            elif self.circle_distance >= self.far_threshold:
                self.circle_within_threshold = False
            
            color = (0, 0, 255) if self.circle_within_threshold else (0, 255, 0)
            cv.putText(debug_frame, f"Smooth: {self.circle_distance:.1f}cm", 
                      (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if self.circle_debug_mode and raw_distance is not None:
                cv.putText(debug_frame, f"Raw: {raw_distance:.1f}cm", 
                          (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def smooth_distance(self, new_distance):
        """Apply smoothing and outlier rejection to distance measurements"""
        # Sanity check - reject obviously incorrect values
        if new_distance is None or new_distance <= 0 or new_distance > 200:  # Max reasonable distance
            return self.last_valid_distance if self.last_valid_distance is not None else 100.0
        
        # Outlier rejection - if the jump is too large, dampen it
        if self.last_valid_distance is not None:
            if abs(new_distance - self.last_valid_distance) > self.max_distance_jump:
                # Instead of rejecting completely, blend with previous value
                weight = self.max_distance_jump / abs(new_distance - self.last_valid_distance)
                new_distance = (new_distance * weight) + (self.last_valid_distance * (1 - weight))
                if self.circle_debug_mode:
                    print(f"Large distance jump detected! Smoothing from {self.last_valid_distance:.1f} to {new_distance:.1f}")
        
        self.distance_history.append(new_distance)
        if len(self.distance_history) > self.distance_history_size:
            self.distance_history.pop(0)
        
        if len(self.distance_history) > 0:
            total_weight = 0
            weighted_sum = 0
            
            for i, dist in enumerate(self.distance_history):
                weight = 1.0 + i / len(self.distance_history)
                weighted_sum += dist * weight
                total_weight += weight
                
            smoothed_distance = weighted_sum / total_weight
            self.last_valid_distance = smoothed_distance
            return smoothed_distance
        
        self.last_valid_distance = new_distance
        return new_distance
    
    def smooth_value(self, new_value, history, last_valid):
        """Apply smoothing and outlier rejection to a value"""
        if new_value is None or new_value <= 0 or new_value >= self.frame_width:
            return last_valid if last_valid is not None else new_value
            
        if history and abs(new_value - history[-1]) > self.max_jump:
            new_value = history[-1] * 0.7 + new_value * 0.3
        
        history.append(new_value)
        
        if len(history) > self.history_size:
            history.pop(0)
            
        if len(history) > 0:
            alpha = 0.6 
            smoothed = new_value * alpha
            
            weight = (1 - alpha) / len(history)
            for hist_val in history[:-1]:
                smoothed += hist_val * weight
                
            return smoothed
        
        return new_value
    
    def process_video(self, video_path, callback=None, show_debug=True, line_detector=None):
        """Process video and report errors via callback"""
        cap = cv.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            left_error, right_error, debug_frame, edges = self.process_frame(frame)
            
            if callback is not None and left_error is not None and right_error is not None:
                callback(left_error, right_error)
            
            if show_debug:
                cv.imshow('Edges', edges)
                cv.imshow('Red Line Detection', debug_frame)
            
            if cv.waitKey(30) & 0xFF == ord('q'):
                break

            print(line_detector.get_circle_info())
        
        cap.release()
        if show_debug:
            cv.destroyAllWindows()
    
    def get_errors(self):
        """Get the most recent error values"""
        return self.left_error, self.right_error
    
    # Serve distance info to servo
    def get_circle_info(self):
        """Get information about the detected circle"""
        return {
            "detected": self.circle_detected,
            "distance": self.circle_distance,
            "within_threshold": self.circle_within_threshold,
            # "raw_history": self.distance_history if self.distance_history else None
        }

def motor_control_callback(left_error, right_error):
    """Proportional control for steering based on lane detection errors"""
    if left_error is None or right_error is None:
        print("No lines detected - stopping motors")
        left_voltage = 0
        right_voltage = 0
        print(f"Motor voltages - Left: {left_voltage:.2f}V, Right: {right_voltage:.2f}V")
        return
    
    position_error = left_error - right_error
    
    base_voltage = 7.4
    
    # Proportional gain (adjust on testing)
    Kp = 0.01
    
    correction = Kp * position_error
    
    left_voltage = base_voltage - correction
    right_voltage = base_voltage + correction
    
    left_voltage = max(0, min(base_voltage, left_voltage))
    right_voltage = max(0, min(base_voltage, right_voltage))
    
    print(f"Lane errors - Left: {left_error:.1f}px, Right: {right_error:.1f}px")
    print(f"Position error: {position_error:.1f}px (Correction: {correction:.2f}V)")
    print(f"Motor voltages - Left: {left_voltage:.2f}V, Right: {right_voltage:.2f}V")

if __name__ == "__main__":
    line_detector = RedLineDetectionAPI()
    
    line_detector.focal_length = 300
    
    line_detector.circle_debug_mode = True
    
    video_path = './IMG_5564 2.MOV'
    line_detector.process_video(video_path, callback=motor_control_callback, line_detector=line_detector)
    
    # For webcam:
    # line_detector.process_video(0, callback=motor_control_callback)