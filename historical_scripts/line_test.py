import cv2 as cv
import numpy as np
import math
import time

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


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # You can change bottom_crop here.
    bottom_crop = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay, edges, error, angle = process_frame(frame, bottom_crop=bottom_crop)
        
        cv.imshow("Edges", edges)
        cv.imshow("Overlay", overlay)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
