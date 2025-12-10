import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image and convert it to grayscale
img = Image.open("bright_input.jpg").convert("L")
original = np.array(img)

# Take only the last 3 bits of each pixel (0–7 range)
reconstructed = original & 7

# Stretch the 0–7 range up to 0–255 so it looks like a normal image
reconstructed_scaled = (reconstructed * 255) // 7

# Check how different the LSB reconstruction is from the original
difference = np.abs(original - reconstructed_scaled)

# Plot everything side by side
plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(original, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("LSB Reconstruction")
plt.imshow(reconstructed_scaled, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Difference Image")
plt.imshow(difference, cmap="gray")
plt.axis("off")

plt.show()

# Save the outputs for reference
Image.fromarray(reconstructed_scaled).save("bright_lsb_reconstructed.png")
Image.fromarray(difference).save("bright_difference.png")
