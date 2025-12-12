import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Load and preprocess the image
# ===============================
img = cv2.imread(r"C:\Users\aarus\Downloads\Picture3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Flatten image into (num_pixels, 3)

pixels = img.reshape(-1, 3).astype(np.float32)

# ===============================
# Choose number of clusters (K)
# ===============================
K = 8
iterations = 10

# ===============================
# Initialize centroids randomly
# ===============================
np.random.seed(42)
centroids = pixels[np.random.choice(pixels.shape[0], K, replace=False)]

# ===============================
# K-Means Algorithm
# ===============================
for _ in range(iterations):
    # Step 1: Assign each pixel to nearest centroid
    distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(distances, axis=1)

    # Step 2: Recompute centroids
    new_centroids = np.array([pixels[labels == k].mean(axis=0) for k in range(K)])

    # If centroids don't change → convergence
    if np.allclose(centroids, new_centroids, atol=1e-3):
        break

    centroids = new_centroids

# ===============================
# Reconstruct compressed image
# ===============================
compressed_pixels = centroids[labels].astype(np.uint8)
compressed_img = compressed_pixels.reshape(img.shape)

# ===============================
# Display original vs compressed
# ===============================
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Compressed (K={K})")
plt.imshow(compressed_img)
plt.axis("off")

plt.show()
