# -------------------------------------------------------------
#  Importing required library
# -------------------------------------------------------------

# Pillow (PIL) is a Python imaging library used to open, modify,
# and save many different image file formats.
# Here we import only the Image class, which provides functions
# for loading and processing images.
from PIL import Image


# -------------------------------------------------------------
#  Loading and preparing the image
# -------------------------------------------------------------

# Image.open("input.jpg"):
#   - Opens the file named "input.jpg" from the current directory.
#   - The returned object is an image instance that can be edited.
#
# .convert("RGB"):
#   - Ensures the image uses 3 color channels (Red, Green, Blue).
#   - This is necessary because quantization algorithms typically
#     work with RGB images.
#
# The result is stored in the variable 'img'.
img = Image.open("input.jpg").convert("RGB")


# -------------------------------------------------------------
#  Performing Octree Color Quantization
# -------------------------------------------------------------

# .quantize(colors=256, method=Image.Quantize.FASTOCTREE):
#
# Quantization means reducing the number of colors in an image.
# The goal is to compress the image or reduce complexity while
# retaining visual quality.
#
# colors=256:
#   - Limits the output image to only 256 unique colors.
#   - 256 colors is typical for GIFs or index color images.
#
# method=Image.Quantize.FASTOCTREE:
#   - Uses the **Fast Octree Quantization algorithm**.
#   - Octree builds a tree data structure where each node represents
#     a subdivision of the RGB color cube.
#   - FASTOCTREE gives a good balance of speed and quality.
#
# The result is a new image with reduced colors.
img = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)


# -------------------------------------------------------------
#  Saving the quantized output image
# -------------------------------------------------------------

# .save("octree_true.png"):
#   - Saves the processed (quantized) image under the name
#     "octree_true.png" in PNG format.
#
# PNG supports 256-color indexed images, so it preserves the
# quantized palette without losing quality.
img.save("output.png")
