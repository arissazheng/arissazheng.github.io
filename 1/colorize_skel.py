# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# name of the input file
imname = './1/CS180_fa2026_proj1_data/cathedral.jpg'

def crop(im, frac=0.1):
    dh = int(im.shape[0] * frac)
    dw = int(im.shape[1] * frac)
    return im[dh:im.shape[0] - dh, dw:im.shape[1] - dw]

def squared_diff(a, b):
    return np.sum((a - b) ** 2)

def ncc(a, b):
    av = a.ravel() - a.mean()
    bv = b.ravel() - b.mean()
    return np.dot(av / np.linalg.norm(av), bv / np.linalg.norm(bv))

def align(moving, base, radius=15, metric='ncc'):
    if metric == 'ncc':
        score = ncc
        better = (lambda s, b: s > b)
    else: 
        score = squared_diff
        better = (lambda s, b: s < b)
    base_crop = crop(base)
    best_score = None
    best_offset = (0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            moving_crop = crop(np.roll(moving, (dy, dx), axis=(0, 1)))
            s = score(moving_crop, base_crop)
            if best_score is None or better(s, best_score):
                best_score = s
                best_offset = (dy, dx)  
    return best_offset

def colorize(name, radius=15, metric='ncc', show=False):
  
    # read in the image as grayscale (the glass plate scan is stacked grayscale)
    im = cv.imread(name, cv.IMREAD_GRAYSCALE)
    
    if im is None:
        raise FileNotFoundError(f"Image file '{name}' not found.")
 
    # convert to float in [0,1] (might want to do this later on to save memory)
    im = im.astype(np.float32) / 255.0
    
    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    #ag = align(g, b, radius, metric)
    #ar = align(r, b, radius, metric)
    g_off = align(g, b, radius, metric)
    r_off = align(r, b, radius, metric)
    print(f'{name.split("/")[-1]:16s} [{metric}] G={g_off} R={r_off}', flush=True)
    ag = np.roll(g, g_off, axis=(0, 1))
    ar = np.roll(r, r_off, axis=(0, 1))

    # placeholders so the script runs before implementing align()
    #ag = g
   # ar = r
    
    # create a color image
    im_out = np.dstack([ar, ag, b])

    # display the image using matplotlib (expects RGB)
    if show: 
        plt.figure(figsize=(8, 8))
        plt.imshow(im_out)
        plt.title('Colorized')
        plt.axis('off')
        plt.show()

    # prepare for OpenCV saving/display (expects BGR uint8)
    out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

    # save the image
    #fname = './out_fname.jpg'
    fname = 'media/' + name.split('/')[-1].split('.')[0] + '_colorized.jpg'
    cv.imwrite(fname, out_bgr)
    
if __name__ == '__main__':
    for imname in ['./CS180_fa2026_proj1_data/cathedral.jpg', './CS180_fa2026_proj1_data/monastery.jpg', './CS180_fa2026_proj1_data/tobolsk.jpg']:
        colorize(imname)
        


