import os, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import glob 


def load(name, width=1000):
    path = glob.glob(f"4/media/{name}.*")[0]      # finds name.jpeg, name.JPG, name.jpg...
    im = ImageOps.exif_transpose(Image.open(path))
    im = im.resize((width, round(im.height * width / im.width)))
    return np.asarray(im) / 255.0

def computeH(im1_pts, im2_pts):
    x, y = im1_pts.T
    x_prime, y_prime = im2_pts.T
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.vstack([np.stack([x, y, o, z, z, z, -x*x_prime, -y*x_prime], 1),
                   np.stack([z, z, z, x, y, o, -x*y_prime, -y*y_prime], 1)])
    b = np.concatenate([x_prime, y_prime])
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)

def apply_H(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]

def warp(im, H, bilinear):
    h, w = im.shape[:2]
    c = apply_H(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    (xmin, ymin), (xmax, ymax) = np.floor(c.min(0)).astype(int), np.ceil(c.max(0)).astype(int)
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    sx, sy = apply_H(np.linalg.inv(H), np.c_[xs.ravel(), ys.ravel()]).T
    out = np.zeros((xs.size, im.shape[2] + 1))
    if bilinear:
        x0 = np.floor(sx).astype(int)
        y0 = np.floor(sy).astype(int)
        ok = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
        ax = (sx - x0)[ok, None]
        ay = (sy - y0)[ok, None]
        x0 = x0[ok]
        y0 = y0[ok]
        out[ok, :-1] = ((1-ax)*(1-ay)*im[y0, x0] + ax*(1-ay)*im[y0, x0+1] +
                       (1-ax)*ay*im[y0+1, x0] + ax*ay*im[y0+1, x0+1])
    else:
        rx = np.round(sx).astype(int)
        ry = np.round(sy).astype(int)
        ok = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
        out[ok, :-1] = im[ry[ok], rx[ok]]
    out[ok, -1] = 1
    return out.reshape(xs.shape + (out.shape[1],)), (xmin, ymin)

def warpImageNearestNeighbor(im, H): 
    return warp(im, H, bilinear=False)
def warpImageBilinear(im, H): 
    return warp(im, H, bilinear=True)

def rectify(name, W, Hh):
    im = load(name)
    pts_file = f"4/{name}_rect.npy"
    if not os.path.exists(pts_file):
        plt.imshow(im); plt.title(f"{name}: click top-left, top-right, bottom-right, bottom-left")
        np.save(pts_file, np.array(plt.ginput(4, timeout=0))); plt.close()
    corners = np.load(pts_file)
    x0, y0 = corners[0]
    H = computeH(corners, np.array([[x0, y0], [x0 + W, y0], [x0 + W, y0 + Hh], [x0, y0 + Hh]]))

    t = time.time(); nn, (ox, oy) = warpImageNearestNeighbor(im, H); t_nn = time.time() - t
    t = time.time(); bil, _ = warpImageBilinear(im, H); t_bil = time.time() - t
    zoom = np.s_[int(y0 - oy):int(y0 - oy) + 120, int(x0 - ox):int(x0 - ox) + 120]  

    fig, ax = plt.subplots(1, 5, figsize=(26, 7))
    for a, img, title in zip(ax, [im, nn, bil, nn[zoom], bil[zoom]],
                             ["original", f"nearest ({t_nn:.2f}s)", f"bilinear ({t_bil:.2f}s)", "nearest (zoom)", "bilinear (zoom)"]):
        a.imshow(img); a.set_title(title); a.axis("off")
    ax[0].plot(*np.vstack([corners, corners[:1]]).T, "r-")
    plt.savefig(f"4/{name}_rectified.png", dpi=150, bbox_inches="tight")
    plt.show()

rectify("madamex", W=200, Hh=428)
rectify("monet", W=300, Hh=300)