import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal
import os

#part 1.1: Convolutions From Scratch!
HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")

def padding(im, kh, kw):
    top = kh // 2
    bottom = (kh - 1) // 2
    left = kw // 2
    right = (kw - 1) // 2
    H, W = im.shape
    padded = np.zeros((H + top + bottom, W + left + right), dtype=np.float64)
    padded[top:top + H, left:left + W] = im
    return padded

def convolution_four(im, kernel):
    kh, kw = kernel.shape
    k = kernel[::-1, ::-1]
    padded = padding(im, kh, kw)
    H, W = im.shape
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            total = 0.0
            for m in range(kh):
                for n in range(kw):
                    total += padded[i + m, j + n] * k[m, n]
            out[i, j] = total
    return out

def convolution_two(im, kernel):
    kh, kw = kernel.shape
    k = kernel[::-1, ::-1]
    padded = padding(im, kh, kw)
    H, W = im.shape
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * k)
    return out

def to_display(x):
    return (x - x.min()) / (x.max() - x.min())

box = np.ones((9, 9)) / 81.0
Dx = np.array([[1, -1]], dtype=np.float64)
Dy = np.array([[1], [-1]], dtype=np.float64)

im = cv.imread(os.path.join(MEDIA, "part1-1.JPG"), cv.IMREAD_GRAYSCALE)
assert im is not None, f"couldn't find {os.path.join(MEDIA, 'part1-1.JPG')}"
im = im.astype(np.float32) / 255.0

# shrink big photos so the four-loop version finishes in reasonable time
scale = 500 / max(im.shape)
if scale < 1:
    im = cv.resize(im, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
print("image size:", im.shape)

t0 = time.perf_counter(); out4 = convolution_four(im, box); t4 = time.perf_counter() - t0
t0 = time.perf_counter(); out2 = convolution_two(im, box); t2 = time.perf_counter() - t0
t0 = time.perf_counter()
scipy_box = scipy.signal.convolve2d(im, box, mode="same", boundary="fill", fillvalue=0)
ts = time.perf_counter() - t0
print(f"four loops: {t4:.3f} s   two loops: {t2:.3f} s   scipy: {ts:.4f} s")

# correctness: reuse the box results from the timing run instead of recomputing them
print("box  four vs scipy:", np.abs(out4 - scipy_box).max(), "  two vs scipy:", np.abs(out2 - scipy_box).max())

dx = convolution_two(im, Dx)
dy = convolution_two(im, Dy)
for name, k, mine in [("Dx", Dx, dx), ("Dy", Dy, dy)]:
    scipy_result = scipy.signal.convolve2d(im, k, mode="same", boundary="fill", fillvalue=0)
    print(name, " two vs scipy:", np.abs(mine - scipy_result).max())

blurred = out2

plt.imsave(os.path.join(MEDIA, "part1-1-gray.jpg"), im, cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-1-box.jpg"), blurred, cmap="gray", vmin=0, vmax=1)
plt.imsave(os.path.join(MEDIA, "part1-1-dx.jpg"), to_display(dx), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-1-dy.jpg"), to_display(dy), cmap="gray")
print("saved results to", MEDIA)

#part 1.2: Finite Difference Operator

cam = cv.imread(os.path.join(MEDIA, "cameraman.png"), cv.IMREAD_GRAYSCALE)
assert cam is not None, f"couldn't find {os.path.join(MEDIA, 'cameraman.png')}"
cam = cam.astype(np.float64) / 255.0

# partial derivatives
cam_dx = scipy.signal.convolve2d(cam, Dx, mode="same", boundary="symm")
cam_dy = scipy.signal.convolve2d(cam, Dy, mode="same", boundary="symm")

# gradient magnitude: how strong the edge is, in any direction
grad_mag = np.sqrt(cam_dx ** 2 + cam_dy ** 2)
print("gradient magnitude range:", grad_mag.min(), grad_mag.max())

plt.imsave(os.path.join(MEDIA, "part1-2-dx.jpg"), to_display(cam_dx), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-2-dy.jpg"), to_display(cam_dy), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-2-gradient-magnitude.jpg"), grad_mag, cmap="gray", vmin=0, vmax=grad_mag.max())

# try several thresholds and look at them to choose one
for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
    edges = grad_mag > t
    print(f"threshold {t}: {edges.mean() * 100:.1f}% of pixels are edges")
    plt.imsave(os.path.join(MEDIA, f"part1-2-edges-{t}.jpg"), edges, cmap="gray")
