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

cameraman = cv.imread(os.path.join(MEDIA, "cameraman.png"), cv.IMREAD_GRAYSCALE)
cameraman = cameraman.astype(np.float64) / 255.0

cameraman_dx = scipy.signal.convolve2d(cameraman, Dx, mode="same", boundary="symm")
cameraman_dy = scipy.signal.convolve2d(cameraman, Dy, mode="same", boundary="symm")

gradient_magnitude = np.sqrt(cameraman_dx ** 2 + cameraman_dy ** 2)
print("gradient magnitude range:", gradient_magnitude.min(), gradient_magnitude.max())

plt.imsave(os.path.join(MEDIA, "part1-2-dx.jpg"), to_display(cameraman_dx), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-2-dy.jpg"), to_display(cameraman_dy), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-2-gradient-magnitude.jpg"), gradient_magnitude, cmap="gray", vmin=0, vmax=gradient_magnitude.max())

for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
    edges = gradient_magnitude   > t
    print(f"threshold {t}: {edges.mean() * 100:.1f}% of pixels are edges")
    plt.imsave(os.path.join(MEDIA, f"part1-2-edges-{t}.jpg"), edges, cmap="gray")

#Part 1.3: Derivative of Gaussian (DoG) Filter

def gaussian_2d(sigma):
    kernel_size = int(6 * sigma) + 1               
    gaussian_1d = cv.getGaussianKernel(kernel_size, sigma)   
    return gaussian_1d @ gaussian_1d.T                      

sigma = 2
G = gaussian_2d(sigma)

cameraman_blur = scipy.signal.convolve2d(cameraman, G, mode="same", boundary="symm")
blur_dx = scipy.signal.convolve2d(cameraman_blur, Dx, mode="same", boundary="symm")
blur_dy = scipy.signal.convolve2d(cameraman_blur, Dy, mode="same", boundary="symm")
blur_magnitude = np.sqrt(blur_dx ** 2 + blur_dy ** 2)
print("blurred gradient magnitude range:", blur_magnitude.min(), blur_magnitude.max())

DoG_x = scipy.signal.convolve2d(G, Dx, mode="full")
DoG_y = scipy.signal.convolve2d(G, Dy, mode="full")
dog_dx = scipy.signal.convolve2d(cameraman, DoG_x, mode="same", boundary="symm")
dog_dy = scipy.signal.convolve2d(cameraman, DoG_y, mode="same", boundary="symm")
dog_magnitude = np.sqrt(dog_dx ** 2 + dog_dy ** 2)

b = G.shape[0]
print("DoG filter shapes:", DoG_x.shape, DoG_y.shape)
print("max diff (whole image):", np.abs(blur_magnitude - dog_magnitude).max())
print("max diff (ignoring border):", np.abs(blur_magnitude - dog_magnitude)[b:-b, b:-b].max())

plt.imsave(os.path.join(MEDIA, "part1-3-blurred.jpg"), cameraman_blur, cmap="gray", vmin=0, vmax=1)
plt.imsave(os.path.join(MEDIA, "part1-3-blur-dx.jpg"), to_display(blur_dx), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-3-blur-dy.jpg"), to_display(blur_dy), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-3-blur-gradient-magnitude.jpg"), blur_magnitude, cmap="gray", vmin=0, vmax=blur_magnitude.max())
plt.imsave(os.path.join(MEDIA, "part1-3-dog-dx.jpg"), to_display(dog_dx), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-3-dog-dy.jpg"), to_display(dog_dy), cmap="gray")
plt.imsave(os.path.join(MEDIA, "part1-3-dog-gradient-magnitude.jpg"), dog_magnitude, cmap="gray", vmin=0, vmax=dog_magnitude.max())

for name, f in [("x", DoG_x), ("y", DoG_y)]:
    big = np.kron(to_display(f), np.ones((20, 20)))
    plt.imsave(os.path.join(MEDIA, f"part1-3-dog-filter-{name}.png"), big, cmap="gray")

for t in [0.02, 0.03, 0.05, 0.08]:
    plt.imsave(os.path.join(MEDIA, f"part1-3-blur-edges-{t}.jpg"), blur_magnitude > t, cmap="gray")
    plt.imsave(os.path.join(MEDIA, f"part1-3-dog-edges-{t}.jpg"), dog_magnitude > t, cmap="gray")
    print(f"threshold {t}: {(blur_magnitude > t).mean() * 100:.1f}% edges, "
          f"{np.sum((blur_magnitude > t) != (dog_magnitude > t))} pixels differ between methods")
