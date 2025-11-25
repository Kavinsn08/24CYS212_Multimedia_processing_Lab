# -------------------------------------------------------------
# Importing required libraries
# -------------------------------------------------------------

# PIL (Pillow) is used for opening, converting, and saving images.
from PIL import Image

# NumPy is used here for efficient array-based pixel operations.
import numpy as np


# -------------------------------------------------------------
# Function: greyscale_desaturation
# Purpose: Convert an image to greyscale using the "desaturation" method
# -------------------------------------------------------------
def greyscale_desaturation(path, save_path):

    # -------------------------------------------------------------
    # Step 1: Load image and convert to RGB
    # -------------------------------------------------------------

    # Image.open(path) loads the image from the given file location.
    # .convert("RGB") ensures the image has Red, Green, and Blue channels.
    img = Image.open(path).convert("RGB")

    # -------------------------------------------------------------
    # Step 2: Convert the image into a NumPy array
    # -------------------------------------------------------------

    # np.array(img) changes the PIL image into a 3-D array:
    #   arr.shape = (height, width, 3)
    #   arr[y, x] = [R, G, B]
    arr = np.array(img)


    # -------------------------------------------------------------
    # Step 3: Extract min and max color per pixel (Desaturation method)
    # -------------------------------------------------------------

    # arr.max(axis=2)
    #   - Computes maximum value among R, G, B for each pixel.
    #   - max_c becomes a 2-D array containing the brightest channel.
    max_c = arr.max(axis=2)

    # arr.min(axis=2)
    #   - Computes minimum value among R, G, B for each pixel.
    #   - min_c becomes a 2-D array containing the darkest channel.
    min_c = arr.min(axis=2)

    # -------------------------------------------------------------
    # Step 4: Compute greyscale using Desaturation formula
    # -------------------------------------------------------------

    # Desaturation formula:
    #     grey = (max(R,G,B) + min(R,G,B)) / 2
    #
    # Meaning:
    #   - Takes the average between the brightest and darkest channel.
    #   - Produces a neutral, balanced grey tone.
    #
    # Example:
    #   Pixel RGB = (120, 200, 60)
    #   max = 200, min = 60
    #   grey = (200 + 60) / 2 = 130
    #
    grey = ((max_c + min_c) / 2).astype(np.uint8)

    # -------------------------------------------------------------
    # Step 5: Convert NumPy array back to a PIL image
    # -------------------------------------------------------------

    # Image.fromarray(grey)
    #   - Automatically detects that this is a single-channel array
    #     and assigns mode "L" (8-bit greyscale).
    grey_img = Image.fromarray(grey)

    # -------------------------------------------------------------
    # Step 6: Save the final greyscale image
    # -------------------------------------------------------------
    grey_img.save(save_path)



# -------------------------------------------------------------
# Function call
# -------------------------------------------------------------

greyscale_desaturation("input.jpg","output.png")
