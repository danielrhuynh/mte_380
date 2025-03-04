import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("./red_line_contrast.png")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Canny edge detection with wide thresholds to observe full gradient intensity
edges = cv2.Canny(blurred_image, 0, 255)

# Calculate the histogram of gradient intensities
histogram = np.histogram(edges.ravel(), bins=256, range=(0, 256))

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.title("Gradient Intensity Histogram")
plt.xlabel("Gradient Intensity")
plt.ylabel("Frequency")
plt.plot(histogram[1][:-1], histogram[0], color='blue')
plt.grid(True)
plt.show()

# Calculate the 10th and 30th percentiles of non-zero gradient intensities
non_zero_edges = edges[edges > 0]
lower_threshold = np.percentile(non_zero_edges, 10)
upper_threshold = np.percentile(non_zero_edges, 30)

print(f"Suggested Lower Threshold: {lower_threshold:.2f}")
print(f"Suggested Upper Threshold: {upper_threshold:.2f}")
