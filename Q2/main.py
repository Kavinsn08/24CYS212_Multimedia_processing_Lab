import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Compute cluster rate term  R = -log2(p_k)
# ---------------------------------------------------------
def compute_rate(labels, K):
    # Calculate probability of each cluster
    prob = np.array([(labels == i).mean() for i in range(K)])
    prob = np.maximum(prob, 1e-12)      # avoid log(0)
    R = -np.log2(prob)
    return R


# ---------------------------------------------------------
# Rate–Distortion K-Means
# ---------------------------------------------------------
def rd_kmeans(pixels, K=16, lam=2.0, iters=15):
    """
    pixels : (N, 3) array of RGB values
    K      : number of clusters
    lam    : lambda (rate–distortion trade-off)
    iters  : number of K-means iterations
    """

    N = len(pixels)

    # ---- 1. K-means initialization ----
    idx = np.random.choice(N, K, replace=False)
    centroids = pixels[idx]
    labels = np.zeros(N, dtype=int)

    for t in range(iters):

        # ---- Compute rate term for each cluster ----
        R = compute_rate(labels, K)      # shape (K,)

        # ---- Compute distortion term D(x, c_k) ----
        D = np.sum((pixels[:, None] - centroids[None, :])**2, axis=2)

        # ---- RD cost → D + λR ----
        RD_cost = D + lam * R

        # ---- Assign pixel to lowest RD cost cluster ----
        labels = np.argmin(RD_cost, axis=1)

        # ---- Update centroids with assigned pixels ----
        new_centroids = []
        for k in range(K):
            pts = pixels[labels == k]
            if len(pts) == 0:
                new_centroids.append(centroids[k])  # keep old centroid
            else:
                new_centroids.append(pts.mean(axis=0))

        centroids = np.array(new_centroids)

        print(f"Iteration {t+1}/{iters} complete")

    return centroids, labels



# ---------------------------------------------------------
# Run RD-KMeans Quantization for an Image
# ---------------------------------------------------------
def rd_kmeans_quantize(path, save_path, K=16, lam=2.0, iters=15):

    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]

    pixels = arr.reshape(-1, 3).astype(float)

    print(f"[+] Running RD-KMeans with K={K}, λ={lam} ...")

    centroids, labels = rd_kmeans(pixels, K, lam, iters)

    # Reconstruct image
    out_pixels = centroids[labels].reshape(H, W, 3).astype(np.uint8)
    Image.fromarray(out_pixels).save(save_path)

    print(f"[+] Saved quantized RD-KMeans image → {save_path}")


# ---------------------------------------------------------
# Example Call
# ---------------------------------------------------------
rd_kmeans_quantize(
    path="input.jpg",
    save_path="output.jpg",
    K=16,
    lam=2.0,
    iters=10
)
