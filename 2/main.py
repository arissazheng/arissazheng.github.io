import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal
import os

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

t0 = time.perf_counter(); out4 = convolution_four(im, box); t4 = time.perf_counter() - t0
t0 = time.perf_counter(); out2 = convolution_two(im, box); t2 = time.perf_counter() - t0
t0 = time.perf_counter()
ref = scipy.signal.convolve2d(im, box, mode="same", boundary="fill", fillvalue=0)
ts = time.perf_counter() - t0
print(f"four loops: {t4:.3f} s   two loops: {t2:.3f} s   scipy: {ts:.4f} s")

for name, k in [("box", box), ("Dx", Dx), ("Dy", Dy)]:
    ref = scipy.signal.convolve2d(im, k, mode="same", boundary="fill", fillvalue=0)
    d4 = np.abs(convolution_four(im, k) - ref).max()
    d2 = np.abs(convolution_two(im, k) - ref).max()
    print(name, "four vs scipy:", d4, "  two vs scipy:", d2)

blurred = out2
dx = convolution_two(im, Dx)
dy = convolution_two(im, Dy)

plt.imsave(os.path.join(MEDIA, "part1-1-gray.jpg"), im, cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-1-box.jpg"), blurred, cmap="gray", vmin=0, vmax=1)
plt.imsave(os.path.join(MEDIA, "part1-1-dx.jpg"), to_display(dx), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-1-dy.jpg"), to_display(dy), cmap="gray")
print("saved results to", MEDIA)