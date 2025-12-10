import cv2
import numpy as np
import math


# Load and prepare the image

path = r"input.jpg"                 # Path of the input image
img = cv2.imread(path)              # Read the image using OpenCV (BGR format)

img = img.astype(np.float32)        # Convert image to float32 for accurate filtering operations



# BOX FILTERS 

# boxFilter() with normalize=True  → output is the average of pixels
# boxFilter() with normalize=False → output is the sum of pixels

box5_avg = cv2.boxFilter(img, -1, (5,5), normalize=True)     
box5_sum = cv2.boxFilter(img, -1, (5,5), normalize=False)    

box20_avg = cv2.boxFilter(img, -1, (20,20), normalize=True)  
box20_sum = cv2.boxFilter(img, -1, (20,20), normalize=False) 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
sigma = np.std(gray)                          # Compute standard deviation
print("Sigma =", sigma)

size = int(round(2 * math.pi * sigma))        # Formula-based initial kernel size

if size < 3:
    size = 3                                   # Minimum size for a valid Gaussian kernel

if size % 2 == 0:
    size += 1                                  # Make sure size is odd

print("Gaussian kernel size =", size)

radius = size // 2
x = np.arange(-radius, radius + 1, 1, dtype=np.float32)  # 1D sample positions

# Compute unnormalized Gaussian (raw values)
gauss_unnorm = np.exp(-(x**2) / (2 * sigma * sigma + 1e-6))

# Convert to normalized version (sum = 1)
gauss_norm = gauss_unnorm / np.sum(gauss_unnorm)

print("Gaussian kernel (normalized) sum =", np.sum(gauss_norm))

# APPLY SEPARABLE GAUSSIAN FILTER

gauss_sep_unnorm = cv2.sepFilter2D(img, -1, gauss_unnorm, gauss_unnorm)
gauss_sep_norm  = cv2.sepFilter2D(img, -1, gauss_norm, gauss_norm)

# SAVE ALL OUTPUT IMAGES

cv2.imwrite("box5_avg.jpg", box5_avg)
cv2.imwrite("box5_sum.jpg", box5_sum)
cv2.imwrite("box20_avg.jpg", box20_avg)
cv2.imwrite("box20_sum.jpg", box20_sum)

cv2.imwrite("gaussian_unnormalized.jpg", gauss_sep_unnorm)
cv2.imwrite("gaussian_normalized.jpg", gauss_sep_norm)

print("All images saved!")
