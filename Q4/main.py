import numpy as np
import imageio.v2 as imageio
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # Allow large images

# Resize all input images to the smallest width & height among them
def resize_all(imgs):
    h = min(im.shape[0] for im in imgs)
    w = min(im.shape[1] for im in imgs)
    return [np.array(Image.fromarray(im).resize((w, h))) for im in imgs]

# Debevec & Malik response curve solver
def gsolve(Z, B, lam=100):
    n = 256                        # pixel intensity range
    P, N = Z.shape
    w = np.array([min(z, 255 - z) for z in range(256)])  # simple weighting

    A = np.zeros((P*N + n + 1, n + P))
    b = np.zeros(P*N + n + 1)

    k = 0
    # Data-fitting constraints
    for i in range(P):
        for j in range(N):
            z = Z[i, j]
            A[k, z] = w[z]
            A[k, n+i] = -w[z]
            b[k] = w[z] * B[j]
            k += 1

    # Fix the curve at middle value to remove scale ambiguity
    A[k, 128] = 1
    k += 1

    # Smoothness term on the response curve
    for z in range(1, n-1):
        A[k, z-1:z+2] = lam * w[z] * np.array([1, -2, 1])
        k += 1

    # Solve least-squares system
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:n]

# Build HDR radiance map
def build_hdr(imgs, log_t, lam=100, samples=200):
    H, W = imgs[0].shape[:2]
    N = len(imgs)

    # Random sample points for solving the CRF
    ys = np.random.randint(0, H, samples)
    xs = np.random.randint(0, W, samples)

    Z = lambda c: np.stack([im[ys, xs, c] for im in imgs], axis=1)

    # Solve camera response for each channel
    gR = gsolve(Z(0), log_t, lam)
    gG = gsolve(Z(1), log_t, lam)
    gB = gsolve(Z(2), log_t, lam)

    w = np.array([min(z, 255 - z) for z in range(256)])
    hdr = np.zeros((H, W, 3), float)

    # Merge exposure images into HDR
    for c, g in enumerate([gR, gG, gB]):
        num = np.zeros((H, W))
        den = np.zeros((H, W))

        for j in range(N):
            Zc = imgs[j][:, :, c]
            ww = w[Zc]
            num += ww * (g[Zc] - log_t[j])
            den += ww

        hdr[:, :, c] = np.exp(num / np.maximum(den, 1e-8))

    return hdr

# Reinhard global tone mapping
def reinhard(hdr, key=0.18, gamma=1/2.2):
    eps = 1e-6

    # Compute luminance
    L = 0.2126*hdr[:,:,0] + 0.7152*hdr[:,:,1] + 0.0722*hdr[:,:,2]
    L = np.maximum(L, eps)

    # Log average luminance
    Lavg = np.exp(np.mean(np.log(L)))

    # Scale scene brightness
    Lm = (key / Lavg) * L

    # Reinhard compression
    Ld = Lm / (1 + Lm)

    # Apply scale back to RGB
    out = hdr * (Ld / L)[:, :, None]

    # Normalization and gamma correction
    out /= out.max() + eps
    out **= gamma

    return (out * 255).clip(0, 255).astype(np.uint8)

# Main execution
if __name__ == "__main__":
    files = ["bright.jpg", "medium.jpg", "dark.jpg"]
    imgs = resize_all([imageio.imread(f) for f in files])

    t = np.array([1/30, 1/60, 1/125])   # exposure times
    log_t = np.log(t)

    hdr = build_hdr(imgs, log_t)
    ldr = reinhard(hdr)

    imageio.imwrite("hdr_out.jpg", ldr)
    print("Saved hdr_out.jpg")
