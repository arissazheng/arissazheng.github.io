import numpy as np
import matplotlib.pyplot as plt
from a3 import load, computeH, warp

def mosaic(center, others):
    names = [center] + others
    imgs = [load(n) for n in names]
    Hs = [np.eye(3)] + [computeH(*np.load(f"4/{n}_{center}.npy")) for n in others]  

    warped = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape[:2]
        x = np.meshgrid(np.arange(w), np.arange(h))
        weight = np.minimum.reduce([x, w - 1 - x, y, h - 1 - y]).astype(float)   
        weight /= weight.max()
        warped.append(warp(np.dstack([im, weight]), H, bilinear=True))       

    o0, (x0, y0) = warped[0]
    xmin = x0,
    ymin =y0
    xmax=x0 + o0.shape[1], 
    ymax=y0 + o0.shape[0]
    for o, (ox, oy) in warped:
        for x, y in ((ox, oy), (ox + o.shape[1], oy + o.shape[0])):
            if x < xmin: 
                xmin = x
            if y < ymin: 
                ymin = y
            if x > xmax: 
                xmax = x
            if y > ymax: 
                ymax = y
    color = np.zeros((ymax - ymin, xmax - xmin, 3))
    total = np.zeros((ymax - ymin, xmax - xmin))
    overwrite = np.zeros((ymax - ymin, xmax - xmin, 4))
    for out, (ox, oy) in warped:
        sl = np.s_[oy - ymin:oy - ymin + out.shape[0], ox - xmin:ox - xmin + out.shape[1]]
        wgt = out[..., 3] * out[..., 4]               
        color[sl] += out[..., :3] * wgt[..., None]
        total[sl] += wgt
        inside = out[..., 4] > 0
        overwrite[sl][inside] = np.c_[out[..., :3][inside], np.ones(inside.sum())]  
    result = np.dstack([color / np.maximum(total, 1e-8)[..., None], total > 0])

    name = "_".join(names)
    plt.imsave(f"4/media/mosaic_{name}.png", np.clip(result, 0, 1))
    fig, ax = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 6))
    for a, im, n in zip(ax, imgs, names):
        a.imshow(im); a.set_title(n); a.axis("off")
    plt.savefig(f"4/media/sources_{name}.png", dpi=120, bbox_inches="tight")
    fig, ax = plt.subplots(2, 1, figsize=(14, 14))
    ax[0].imshow(np.clip(overwrite, 0, 1)); ax[0].set_title("no blending (later image overwrites)")
    ax[1].imshow(np.clip(result, 0, 1)); ax[1].set_title("weighted averaging (feathered)")
    for a in ax: a.axis("off")
    plt.savefig(f"4/media/compare_{name}.png", dpi=120, bbox_inches="tight")
    plt.show()

mosaic("doe8", ["doe6", "doe10"])
mosaic("gateway7", ["gateway5"])
mosaic("doe13", ["doe11"])