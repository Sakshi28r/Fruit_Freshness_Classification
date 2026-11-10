"""
feature_extraction.py
---------------------
Turns one RGB image (128x128x3 array) into one numeric feature vector.

A PNN does not look at raw pixels; it compares feature vectors using Euclidean
distance, so we need features that capture what makes fruit look fresh/rotten:

  Block 1  HSV colour histogram   - overall colour distribution
  Block 2  Lab colour moments     - average brightness/colour + spread
  Block 3  HOG                    - edge and shape structure
  Block 4  LBP texture histogram  - fine surface texture (spots, wrinkles, mould)

Run a self-test with:   python -m src.feature_extraction
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from src.preprocessing import load_image

# Sizes of each block (used later to balance the blocks - see pnn / train).
HSV_BINS = (8, 4, 4)             # 8 hue x 4 saturation x 4 value = 128 values
LBP_POINTS, LBP_RADIUS = 8, 1    # 'uniform' LBP with 8 points -> 10 bins
HOG_SIZE = (64, 64)              # HOG is computed on a 64x64 grayscale copy


def hsv_histogram(img_rgb):
    """
    3-D colour histogram in HSV space, normalised to sum to 1.

    Why HSV instead of RGB? Hue separates 'what colour' from 'how bright',
    so a yellow banana in shadow still has the same hue as one in sunlight.
    Why normalise? So image size/brightness of the whole picture does not
    change the total - only the *proportion* of each colour matters.
    Rotten fruit shifts towards brown/black (low value, low saturation).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # OpenCV ranges: H 0-179, S 0-255, V 0-255
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS,
                        [0, 180, 0, 256, 0, 256]).flatten()
    return hist / (hist.sum() + 1e-9)


def lab_moments(img_rgb):
    """
    Mean and standard deviation of each Lab channel (6 numbers).

    Lab is designed so that distances match human colour perception.
    L = lightness, a = green-red, b = blue-yellow. Rotten fruit is usually
    darker (lower mean L) and browner (shifted a/b), so these 6 numbers are
    compact, strong cues.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])


def hog_features(img_rgb):
    """
    Histogram of Oriented Gradients on a 64x64 grayscale image.

    HOG counts edge directions in small cells. It captures shape and the
    pattern of wrinkles/bruises while ignoring colour. Settings:
    9 orientations, 16x16-pixel cells, 2x2-cell blocks with L2-Hys
    normalisation -> 324 values.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_AREA)
    return hog(gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


def lbp_histogram(img_rgb):
    """
    Local Binary Pattern histogram (10 values, sums to 1).

    For every pixel LBP compares it with its 8 neighbours and records a
    binary pattern. The histogram of patterns describes micro-texture:
    smooth fresh skin vs. blotchy, mouldy or wrinkled rotten skin.
    This is the "extra feature justified" beyond colour and HOG.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    return hist / (hist.sum() + 1e-9)


# Name -> function. The ORDER here is the order of the blocks in the vector.
_BLOCKS = [
    ("hsv_hist", hsv_histogram),
    ("lab_moments", lab_moments),
    ("hog", hog_features),
    ("lbp", lbp_histogram),
]


def extract_features(img_rgb):
    """Concatenate all feature blocks into one 1-D float32 vector."""
    return np.concatenate([fn(img_rgb) for _, fn in _BLOCKS]).astype(np.float32)


def feature_block_slices():
    """
    Returns {block_name: slice} describing where each block sits inside the
    vector. We use it to give every block equal influence (a 324-value HOG
    block would otherwise drown out the 6 Lab numbers).
    """
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    slices, start = {}, 0
    for name, fn in _BLOCKS:
        n = len(fn(dummy))
        slices[name] = slice(start, start + n)
        start += n
    return slices


def extract_from_path(path):
    """Convenience: load an image file, then extract its features."""
    return extract_features(load_image(path))


if __name__ == "__main__":
    # Quick self-test on a synthetic image so you can see the block sizes.
    test_img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    vec = extract_features(test_img)
    print("Total feature length:", vec.shape[0])
    for name, sl in feature_block_slices().items():
        print(f"  {name:12s} -> {sl.stop - sl.start:4d} values")
    print("Any NaN/Inf?", bool(~np.isfinite(vec).all()))