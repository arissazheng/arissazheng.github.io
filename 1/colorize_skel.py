# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# name of the input file
#imname = './1/CS180_fa2026_proj1_data/cathedral.jpg'

#cropping to remove image edges from scoring and returns middle 80% of rols/cols
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

def align(moving, base, radius=15, metric='ncc', center=(0, 0)):
    if metric == 'ncc':
        score = ncc
        better = (lambda s, b: s > b)
    else: 
        score = squared_diff
        better = (lambda s, b: s < b)
    center_y, center_x = center
    base_crop = crop(base)
    best_score = None
    best_offset = center
    for dy in range(center_y -radius, center_y +radius + 1):
        for dx in range(center_x - radius, center_x + radius + 1):
            moving_crop = crop(np.roll(moving, (dy, dx), axis=(0, 1)))
            s = score(moving_crop, base_crop)
            if best_score is None or better(s, best_score):
                best_score = s
                best_offset = (dy, dx)  
    return best_offset

def downsample(im):
    return cv.resize(im, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

def pyramid(moving, base, radius=15, metric='ncc', min_size=400):
    if min(base.shape) < min_size:
        return align(moving, base, radius, metric)
    coarse = pyramid(downsample(moving), downsample(base), radius, metric, min_size)
    guess = (coarse[0] * 2, coarse[1] * 2)
    return align(moving, base, 2, metric, guess)

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
    #g_off = align=(g, b, radius, metric)
    #r_off = align(r, b, radius, metric)
    g_off = pyramid(g, b, radius, metric)
    r_off = pyramid(r, b, radius, metric)
    
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
    fname = 'media/' + name.split('/')[-1].split('.')[0] + '_pyramid.jpg'
    cv.imwrite(fname, out_bgr)
    
if __name__ == '__main__':
    for imname in ['./CS180_fa2026_proj1_data/cathedral.jpg', 
                   './CS180_fa2026_proj1_data/monastery.jpg', 
                   './CS180_fa2026_proj1_data/tobolsk.jpg',
                   './CS180_fa2026_proj1_data/church.tif',
                   './CS180_fa2026_proj1_data/emir.tif',
                   './CS180_fa2026_proj1_data/harvesters.tif',
                   './CS180_fa2026_proj1_data/icon.tif',
                   './CS180_fa2026_proj1_data/ilemselga.tif',
                   './CS180_fa2026_proj1_data/melons.tif',
                   './CS180_fa2026_proj1_data/religous_painting.tif',
                   './CS180_fa2026_proj1_data/self_portrait.tif',
                   './CS180_fa2026_proj1_data/siren.tif',
                   './CS180_fa2026_proj1_data/three_generations.tif',
                   './CS180_fa2026_proj1_data/wharf.tif',
                   './CS180_fa2026_proj1_data/camel.jpg',
                   './CS180_fa2026_proj1_data/interior.jpg',
                   './CS180_fa2026_proj1_data/statue.jpg']:
        colorize(imname, metric='l2')
        colorize(imname, metric='ncc')
        
MAXDIM = 900
if max(out_bgr.shape[:2]) > MAXDIM:
    s = MAXDIM / max(out_bgr.shape[:2])
    out_bgr = cv.resize(out_bgr, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
cv.imwrite(fname, out_bgr, [cv.IMWRITE_JPEG_QUALITY, 85])
