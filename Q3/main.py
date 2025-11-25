import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def load_gray(path):
    img = Image.open(path).convert("L")
    return np.array(img).astype(np.float32)
def frequency_sampling(gray_img, factor):
    H, W = gray_img.shape
    F = np.fft.fft2(gray_img)
    F_shift = np.fft.fftshift(F)
    h_new = int(H * factor)
    w_new = int(W * factor)
    F_low = np.zeros_like(F_shift)
    cx, cy = H // 2, W // 2
    F_low[cx - h_new//2 : cx + h_new//2,
        cy - w_new//2 : cy + w_new//2] = \
        F_shift[cx - h_new//2 : cx + h_new//2,
                cy - w_new//2 : cy + w_new//2]
    F_ishift = np.fft.ifftshift(F_low)
    img_rec = np.abs(np.fft.ifft2(F_ishift))
    return img_rec.astype(np.uint8)
def spatial_sampling(gray_img, factor):
    H, W = gray_img.shape
    new_H = max(1, int(H * factor))
    new_W = max(1, int(W * factor))
    img_small = Image.fromarray(gray_img.astype(np.uint8)).resize(
        (new_W, new_H), Image.NEAREST)
    img_back = img_small.resize((W, H), Image.NEAREST)
    return np.array(img_back)
def run_all(path):
    gray = load_gray(path)
    factors = [0.5, 0.25, 0.125, 0.0625]
    for f in factors:
        out_freq = frequency_sampling(gray, f)
        out_spatial = spatial_sampling(gray, f)
        Image.fromarray(out_freq).save(
            rf"D:\College\Programming\Python\Multimedia\Frequency & Spatial Sampling\freq_{f}.png"
        )
        Image.fromarray(out_spatial).save(
            rf"D:\College\Programming\Python\Multimedia\Frequency & Spatial Sampling\spatial_{f}.png"
        )
        print(f"Saved: freq_{f}.png  | spatial_{f}.png")
run_all(
    r"D:\College\Programming\Python\Multimedia\Frequency & Spatial Sampling\input.jpg"
)
