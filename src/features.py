"""
AgriSense - M2: Leaf disease feature extractor (Python reference implementation).

CONTRACT (see docs/01_Project_Outline.md section 7)
--------------------------------------------------
This module MUST stay bit-for-bit equivalent to web/features.js.
Every operation here is written in plain NumPy rather than calling scikit-image,
precisely so that the JavaScript mirror can reproduce it exactly. Do not
"simplify" this by swapping in a library call - it will silently break browser
predictions. Any change must be made in both files and re-verified with
src/parity_test.py.

Feature vector: 97 dimensions
  B1 colour moments        18
  B2 HSV histogram         32
  B3 lesion chromaticity    8
  B4 GLCM texture          24
  B5 uniform LBP           10
  B6 shape                  5
"""

from __future__ import annotations

import numpy as np

IMG_SIZE = 160
GLCM_LEVELS = 16
GLCM_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))   # (dy, dx)
EPS = 1e-12


# ----------------------------------------------------------------------------
# 0. Pre-processing: centre crop + manual bilinear resize
# ----------------------------------------------------------------------------

def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    return img[top:top + s, left:left + s]


def resize_bilinear(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Half-pixel-centre bilinear resize. Mirrored exactly in features.js."""
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    src = img.reshape(h, w, c).astype(np.float64)

    scale_y = h / size
    scale_x = w / size
    yy = (np.arange(size) + 0.5) * scale_y - 0.5
    xx = (np.arange(size) + 0.5) * scale_x - 0.5
    yy = np.clip(yy, 0, h - 1)
    xx = np.clip(xx, 0, w - 1)

    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0)[:, None, None]
    wx = (xx - x0)[None, :, None]

    a = src[np.ix_(y0, x0)]
    b = src[np.ix_(y0, x1)]
    cc = src[np.ix_(y1, x0)]
    d = src[np.ix_(y1, x1)]
    out = (a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx +
           cc * wy * (1 - wx) + d * wy * wx)
    return out


def preprocess(img_uint8: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float64 (size,size,3) in [0,1]."""
    img = center_crop_square(np.asarray(img_uint8))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[:, :, :3]
    out = resize_bilinear(img, size)
    return np.clip(out / 255.0, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 1. Colour space
# ----------------------------------------------------------------------------

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] -> h in [0,1), s in [0,1], v in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn

    h = np.zeros_like(mx)
    mask = d > EPS
    rm = mask & (mx == r)
    gm = mask & (mx == g) & ~rm
    bm = mask & (mx == b) & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4.0
    h = (h / 6.0) % 1.0

    s = np.where(mx > EPS, d / np.maximum(mx, EPS), 0.0)
    return np.stack([h, s, mx], axis=-1)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


# ----------------------------------------------------------------------------
# 2. Leaf segmentation: ExGR index -> Otsu -> largest component -> fill holes
# ----------------------------------------------------------------------------

def otsu_threshold(vals_u8: np.ndarray) -> int:
    hist = np.bincount(vals_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    idx = np.arange(256, dtype=np.float64)
    sum_all = float((hist * idx).sum())
    w_b = 0.0
    sum_b = 0.0
    best_var = -1.0
    best_t = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += idx[t] * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


def largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    best_label, best_size = 0, 0
    cur = 0
    flat = mask.ravel()
    lab = labels.ravel()
    stack = np.empty(h * w, dtype=np.int64)
    for start in range(h * w):
        if not flat[start] or lab[start]:
            continue
        cur += 1
        sp = 0
        stack[sp] = start; sp += 1
        lab[start] = cur
        size = 0
        while sp:
            sp -= 1
            p = stack[sp]
            size += 1
            y, x = divmod(p, w)
            if y > 0:
                q = p - w
                if flat[q] and not lab[q]:
                    lab[q] = cur; stack[sp] = q; sp += 1
            if y < h - 1:
                q = p + w
                if flat[q] and not lab[q]:
                    lab[q] = cur; stack[sp] = q; sp += 1
            if x > 0:
                q = p - 1
                if flat[q] and not lab[q]:
                    lab[q] = cur; stack[sp] = q; sp += 1
            if x < w - 1:
                q = p + 1
                if flat[q] and not lab[q]:
                    lab[q] = cur; stack[sp] = q; sp += 1
        if size > best_size:
            best_size, best_label = size, cur
    return (labels == best_label) if best_label else mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return mask | ~outside


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    exgr = exg - exr
    lo, hi = float(exgr.min()), float(exgr.max())
    if hi - lo < 1e-9:
        return np.ones(rgb.shape[:2], dtype=bool)
    q = np.clip(np.floor((exgr - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
    t = otsu_threshold(q)
    mask = q > t
    if mask.mean() < 0.05 or mask.mean() > 0.98:
        return np.ones(rgb.shape[:2], dtype=bool)
    mask = largest_component(mask)
    mask = fill_holes(mask)
    if mask.mean() < 0.05:
        return np.ones(rgb.shape[:2], dtype=bool)
    return mask


# ----------------------------------------------------------------------------
# 3. Feature blocks
# ----------------------------------------------------------------------------

def _moments(x: np.ndarray) -> tuple:
    if x.size == 0:
        return 0.0, 0.0, 0.0
    m = float(x.mean())
    s = float(x.std())
    sk = float((((x - m) / (s + EPS)) ** 3).mean()) if s > EPS else 0.0
    return m, s, sk


def block1_colour_moments(rgb, hsv, mask):
    out = []
    for ch in (rgb[..., 0], rgb[..., 1], rgb[..., 2],
               hsv[..., 0], hsv[..., 1], hsv[..., 2]):
        out.extend(_moments(ch[mask]))
    return out


def block2_hsv_histogram(hsv, mask):
    h = hsv[..., 0][mask]
    s = hsv[..., 1][mask]
    v = hsv[..., 2][mask]
    n = max(h.size, 1)
    hh = np.bincount(np.clip((h * 16).astype(int), 0, 15), minlength=16) / n
    hs = np.bincount(np.clip((s * 8).astype(int), 0, 7), minlength=8) / n
    hv = np.bincount(np.clip((v * 8).astype(int), 0, 7), minlength=8) / n
    return list(hh) + list(hs) + list(hv)


def block3_lesion_chromaticity(hsv, mask):
    h = hsv[..., 0][mask] * 360.0
    s = hsv[..., 1][mask]
    v = hsv[..., 2][mask]
    n = max(h.size, 1)
    green = ((h >= 75) & (h < 160) & (s > 0.25) & (v > 0.15)).sum() / n
    yellow = ((h >= 40) & (h < 75) & (s > 0.30) & (v > 0.30)).sum() / n
    brown = ((h >= 15) & (h < 40) & (s > 0.20) & (v >= 0.15) & (v < 0.60)).sum() / n
    dark = (v < 0.22).sum() / n
    grey = ((s < 0.18) & (v >= 0.22)).sum() / n
    purple = (((h >= 260) | (h < 15)) & (s > 0.15)).sum() / n
    coverage = float(mask.mean())
    vitality = green / (green + brown + yellow + EPS)
    return [green, yellow, brown, dark, grey, purple, coverage, vitality]


def block4_glcm(gray, mask):
    q = np.clip((gray * GLCM_LEVELS).astype(int), 0, GLCM_LEVELS - 1)
    feats = []
    L = GLCM_LEVELS
    i_idx = np.arange(L)[:, None]
    j_idx = np.arange(L)[None, :]
    for dy, dx in GLCM_OFFSETS:
        h, w = q.shape
        ys0 = slice(max(0, -dy), h - max(0, dy))
        xs0 = slice(max(0, -dx), w - max(0, dx))
        ys1 = slice(max(0, dy), h - max(0, -dy))
        xs1 = slice(max(0, dx), w - max(0, -dx))
        a = q[ys0, xs0].ravel()
        b = q[ys1, xs1].ravel()
        m = (mask[ys0, xs0] & mask[ys1, xs1]).ravel()
        a, b = a[m], b[m]
        P = np.zeros((L, L), dtype=np.float64)
        if a.size:
            np.add.at(P, (a, b), 1.0)
            P = P + P.T
            P /= P.sum()
        contrast = float((P * (i_idx - j_idx) ** 2).sum())
        dissim = float((P * np.abs(i_idx - j_idx)).sum())
        homog = float((P / (1.0 + (i_idx - j_idx) ** 2)).sum())
        asm = float((P ** 2).sum())
        energy = float(np.sqrt(asm))
        mu_i = float((P * i_idx).sum())
        mu_j = float((P * j_idx).sum())
        sd_i = float(np.sqrt((P * (i_idx - mu_i) ** 2).sum()))
        sd_j = float(np.sqrt((P * (j_idx - mu_j) ** 2).sum()))
        corr = float((P * (i_idx - mu_i) * (j_idx - mu_j)).sum() / (sd_i * sd_j + EPS))
        ent = float(-(P * np.log(P + EPS)).sum())
        feats.extend([contrast, dissim, homog, energy, corr, ent])
    return feats


_LBP_NEIGHBOURS = ((0, 1), (-1, 1), (-1, 0), (-1, -1),
                   (0, -1), (1, -1), (1, 0), (1, 1))


def block5_lbp(gray, mask):
    h, w = gray.shape
    hist = np.zeros(10, dtype=np.float64)
    core = np.zeros((h, w), dtype=bool)
    core[1:-1, 1:-1] = mask[1:-1, 1:-1]
    code = np.zeros((h, w), dtype=np.int32)
    for bit, (dy, dx) in enumerate(_LBP_NEIGHBOURS):
        shifted = np.roll(np.roll(gray, -dy, axis=0), -dx, axis=1)
        code |= ((shifted >= gray).astype(np.int32) << bit)
    codes = code[core]
    if codes.size == 0:
        return list(hist)
    bits = ((codes[:, None] >> np.arange(8)[None, :]) & 1)
    transitions = (bits != np.roll(bits, -1, axis=1)).sum(axis=1)
    popcount = bits.sum(axis=1)
    binidx = np.where(transitions <= 2, popcount, 9)
    hist = np.bincount(binidx, minlength=10).astype(np.float64)
    hist /= max(codes.size, 1)
    return list(hist)


def block6_shape(mask):
    h, w = mask.shape
    area = float(mask.sum())
    if area < 1:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    ys, xs = np.nonzero(mask)
    bh = ys.max() - ys.min() + 1
    bw = xs.max() - xs.min() + 1
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                         & mask[1:-1, :-2] & mask[1:-1, 2:])
    perimeter = float((mask & ~inner).sum())
    area_frac = area / (h * w)
    circularity = 4.0 * np.pi * area / (perimeter ** 2 + EPS)
    extent = area / float(bh * bw)
    cy, cx = ys.mean(), xs.mean()
    myy = float(((ys - cy) ** 2).mean())
    mxx = float(((xs - cx) ** 2).mean())
    mxy = float(((ys - cy) * (xs - cx)).mean())
    tmp = np.sqrt(max((mxx - myy) ** 2 + 4 * mxy ** 2, 0.0))
    l1 = (mxx + myy + tmp) / 2.0
    l2 = (mxx + myy - tmp) / 2.0
    ecc = float(np.sqrt(max(1.0 - l2 / (l1 + EPS), 0.0)))
    aspect = float(bw) / float(bh)
    return [area_frac, float(min(circularity, 4.0)), extent, ecc, aspect]


# ----------------------------------------------------------------------------
# 4. Public API
# ----------------------------------------------------------------------------

def feature_names() -> list:
    names = []
    for ch in ("R", "G", "B", "H", "S", "V"):
        names += [f"B1_{ch}_mean", f"B1_{ch}_std", f"B1_{ch}_skew"]
    names += [f"B2_H_bin{i}" for i in range(16)]
    names += [f"B2_S_bin{i}" for i in range(8)]
    names += [f"B2_V_bin{i}" for i in range(8)]
    names += ["B3_green", "B3_yellow", "B3_brown", "B3_dark",
              "B3_grey", "B3_purple", "B3_coverage", "B3_vitality"]
    for k, (dy, dx) in enumerate(GLCM_OFFSETS):
        for p in ("contrast", "dissim", "homog", "energy", "corr", "entropy"):
            names.append(f"B4_o{k}_{p}")
    names += [f"B5_lbp{i}" for i in range(10)]
    names += ["B6_area_frac", "B6_circularity", "B6_extent",
              "B6_eccentricity", "B6_aspect"]
    return names


N_FEATURES = 97


def extract_features(img_uint8: np.ndarray) -> np.ndarray:
    rgb = preprocess(img_uint8)
    hsv = rgb_to_hsv(rgb)
    gray = to_gray(rgb)
    mask = leaf_mask(rgb)
    vec = []
    vec += block1_colour_moments(rgb, hsv, mask)
    vec += block2_hsv_histogram(hsv, mask)
    vec += block3_lesion_chromaticity(hsv, mask)
    vec += block4_glcm(gray, mask)
    vec += block5_lbp(gray, mask)
    vec += block6_shape(mask)
    out = np.asarray(vec, dtype=np.float64)
    assert out.size == N_FEATURES, f"expected {N_FEATURES}, got {out.size}"
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_from_path(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return extract_features(arr)
