import cv2 as cv
import numpy as np

def detect_red_line(image_path):
    frame = cv.imread(image_path)
    if frame is None:
        print("Error: Could not read the image")
        return
    
    frame = cv.resize(frame, (480, 480))

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

    if lines is not None:
            print("Red line detected")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('Edges', edges)
    cv.imshow('Red Line Detection', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detect_red_line_video(video_path):
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
        
        frame = cv.resize(frame, (480, 480))
        
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
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv.imshow('Edges', edges)
        cv.imshow('Red Line Detection', frame)
        
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

# For image testing
# image_path = './red_line_contrast.png'
# detect_red_line(image_path)

# For video testing
video_path = './IMG_5564 2.MOV'
detect_red_line_video(video_path)

# For webcam
# detect_red_line_video(0)