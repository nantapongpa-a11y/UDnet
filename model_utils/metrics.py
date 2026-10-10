"""Image quality metrics for underwater image enhancement.

All functions take uint8 (or float 0-255) RGB images of shape (H, W, 3).

Full-reference : PSNR, SSIM, GMSD
No-reference   : UIQM, UCIQE

NOTE: published UIQM / UCIQE numbers come from many slightly different
implementations, so absolute values are only comparable when every method is
scored with the SAME script. Re-score baselines with this file instead of
copying numbers from papers.
"""
import math
import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ----------------------------------------------------------------------------
# Full-reference
# ----------------------------------------------------------------------------
def psnr(pred, gt):
    return peak_signal_noise_ratio(gt, pred, data_range=255)


def ssim(pred, gt):
    # SSIM on RGB, averaged over channels (common practice in UIE papers)
    return structural_similarity(gt, pred, channel_axis=2, data_range=255)


def _rgb2y(img):
    img = img.astype(np.float64)
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def gmsd(pred, gt):
    # Xue et al. 2014, port of the authors' MATLAB code. Lower is better.
    T = 170
    y1, y2 = _rgb2y(gt), _rgb2y(pred)
    ave = np.ones((2, 2)) / 4.0
    y1 = convolve2d(y1, ave, mode='same')[::2, ::2]
    y2 = convolve2d(y2, ave, mode='same')[::2, ::2]
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
    dy = dx.T
    g1 = np.sqrt(convolve2d(y1, dx, mode='same') ** 2 + convolve2d(y1, dy, mode='same') ** 2)
    g2 = np.sqrt(convolve2d(y2, dx, mode='same') ** 2 + convolve2d(y2, dy, mode='same') ** 2)
    quality_map = (2 * g1 * g2 + T) / (g1 ** 2 + g2 ** 2 + T)
    return float(np.std(quality_map, ddof=1))


# ----------------------------------------------------------------------------
# UIQM (Panetta et al. 2016) -- follows the widely used Python port from
# FUnIE-GAN (uqim_utils.py), vectorised but numerically the same.
# ----------------------------------------------------------------------------
def _mu_a(x, alpha_L=0.1, alpha_R=0.1):
    # asymmetric alpha-trimmed mean
    x = np.sort(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    weight = 1.0 / (K - T_a_L - T_a_R)
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    return weight * np.sum(x[s:e])


def _s_a(x, mu):
    return np.mean(np.power(x - mu, 2))


def _uicm(x):
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = _mu_a(RG)
    mu_a_YB = _mu_a(YB)
    s_a_RG = _s_a(RG, mu_a_RG)
    s_a_YB = _s_a(YB, mu_a_YB)
    l = math.sqrt(mu_a_RG ** 2 + mu_a_YB ** 2)
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


def _sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    m = np.max(mag)
    if m > 0:
        mag *= 255.0 / m
    return mag


def _block_max_min(x, window_size):
    # max / min of each non-overlapping window_size x window_size block
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    x = x[:window_size * k2, :window_size * k1]
    if x.ndim == 2:
        b = x.reshape(k2, window_size, k1, window_size)
        axes = (1, 3)
    else:
        b = x.reshape(k2, window_size, k1, window_size, x.shape[2])
        axes = (1, 3, 4)
    return b.max(axis=axes), b.min(axis=axes), k1 * k2


def _eme(x, window_size):
    max_, min_, n = _block_max_min(x, window_size)
    valid = (min_ != 0) & (max_ != 0)
    ratio = np.where(valid, max_ / np.where(valid, min_, 1), 1)
    return (2.0 / n) * np.sum(np.log(ratio))


def _uism(x):
    R, G, B = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    r_eme = _eme(_sobel(R) * R, 10)
    g_eme = _eme(_sobel(G) * G, 10)
    b_eme = _eme(_sobel(B) * B, 10)
    return 0.299 * r_eme + 0.587 * g_eme + 0.114 * b_eme


def _uiconm(x, window_size):
    max_, min_, n = _block_max_min(x, window_size)
    top = max_ - min_
    bot = max_ + min_
    valid = (top != 0) & (bot != 0)
    ratio = np.where(valid, top / np.where(valid, bot, 1), 1)
    return (-1.0 / n) * np.sum(ratio * np.log(ratio))


def uiqm(img):
    x = img.astype(np.float32)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * _uicm(x) + c2 * _uism(x) + c3 * _uiconm(x, 10)


# ----------------------------------------------------------------------------
# UCIQE (Yang & Sowmya 2015) -- follows the common Python port that works on
# OpenCV 8-bit Lab. That port does NOT centre a/b at 128, which inflates the
# chroma terms; pass centered_ab=True for the corrected version. Pick one and
# use it for every method.
# ----------------------------------------------------------------------------
def uciqe(img, centered_ab=False):
    lab = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    lum = lab[:, :, 0] / 255.0
    if centered_ab:
        a = (lab[:, :, 1] - 128.0) / 255.0
        b = (lab[:, :, 2] - 128.0) / 255.0
    else:
        a = lab[:, :, 1] / 255.0
        b = lab[:, :, 2] / 255.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)

    lum = lum.flatten()
    order = np.argsort(lum)
    top = order[int(len(lum) * 0.99)]
    bottom = order[int(len(lum) * 0.01)]
    con_lum = lum[top] - lum[bottom]

    chroma = chroma.flatten()
    sat = np.divide(chroma, lum, out=np.zeros_like(chroma), where=lum != 0)
    avg_sat = np.mean(sat)
    return c1 * sigma_c + c2 * con_lum + c3 * avg_sat
