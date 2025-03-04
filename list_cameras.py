import cv2 as cv

def list_cameras():
    """List all available camera devices"""
    camera_idx = 0
    while True:
        cap = cv.VideoCapture(camera_idx)
        if not cap.read()[0]:
            break
        print(f"Camera index {camera_idx} is available")
        cap.release()
        camera_idx += 1

# Test available cameras
list_cameras()