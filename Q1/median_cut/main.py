from PIL import Image
import numpy as np

def median_cut(path, save_path, colors=16):
    img = Image.open(path).convert("RGB")

    # Use Pillow's built-in Median Cut quantizer
    quantized = img.quantize(colors=colors, method=Image.MEDIANCUT)

    # Convert quantized result back to RGB for saving as common formats
    quantized = quantized.convert("RGB")
    quantized.save(save_path)


median_cut(input.jpg,output.png,colors=16)
