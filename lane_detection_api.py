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
        
    def process_frame(self, frame):
        """Process a single frame and return left and right errors"""
        if frame is None:
            return None, None, None, None

        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.resize(frame, (self.frame_width, self.frame_height))
        
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
        
        # Define bottom region for error calculation
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
        
        return left_error, right_error, debug_frame, edges
        
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
            # Smoothing factor for current value (0.7 means 70% current, 30% history)
            alpha = 0.6 
            smoothed = new_value * alpha
            
            weight = (1 - alpha) / len(history)
            for hist_val in history[:-1]:
                smoothed += hist_val * weight
                
            return smoothed
        
        return new_value
    
    def process_video(self, video_path, callback=None, show_debug=True):
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
        
        cap.release()
        if show_debug:
            cv.destroyAllWindows()
    
    def get_errors(self):
        """Get the most recent error values"""
        return self.left_error, self.right_error

def motor_control_callback(left_error, right_error):
    """Example callback for motor control"""
    if left_error is None or right_error is None:
        print("No lines detected - no control action")
        return
    
    print(f"Left error: {left_error}, Right error: {right_error}")

if __name__ == "__main__":
    line_detector = RedLineDetectionAPI()
    
    video_path = './IMG_5564 2.MOV'
    line_detector.process_video(video_path, callback=motor_control_callback)
    
    # For webcam:
    # line_detector.process_video(0, callback=motor_control_callback)