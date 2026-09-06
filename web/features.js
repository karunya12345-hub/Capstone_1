/*
 * AgriSense - M3: browser-side leaf feature extractor.
 *
 * EXACT MIRROR of src/features.py. Every operation is hand-written rather than
 * delegated to canvas/library helpers so that the 97-dimensional vector this
 * produces is identical to the one the notebook trained on. If you change one
 * file you MUST change the other and re-run src/parity_test.py.
 *
 * Contract: docs/01_Project_Outline.md section 7.
 */

export const IMG_SIZE = 160;
export const GLCM_LEVELS = 16;
export const GLCM_OFFSETS = [[0, 1], [1, 0], [1, 1], [1, -1]];   // [dy, dx]
export const N_FEATURES = 97;
const EPS = 1e-12;

/* ------------------------------------------------------------------ *
 * 0. Pre-processing: centre crop + half-pixel-centre bilinear resize   *
 * ------------------------------------------------------------------ */

/** imgData: {data: Uint8ClampedArray RGBA, width, height} -> Float64Array size*size*3 in [0,1] */
export function preprocess(imgData, size = IMG_SIZE) {
  const { data, width: W, height: H } = imgData;
  const s = Math.min(H, W);
  const top = Math.floor((H - s) / 2);
  const left = Math.floor((W - s) / 2);

  const out = new Float64Array(size * size * 3);
  const scale = s / size;
  for (let oy = 0; oy < size; oy++) {
    let yy = (oy + 0.5) * scale - 0.5;
    yy = Math.min(Math.max(yy, 0), s - 1);
    const y0 = Math.floor(yy);
    const y1 = Math.min(y0 + 1, s - 1);
    const wy = yy - y0;
    for (let ox = 0; ox < size; ox++) {
      let xx = (ox + 0.5) * scale - 0.5;
      xx = Math.min(Math.max(xx, 0), s - 1);
      const x0 = Math.floor(xx);
      const x1 = Math.min(x0 + 1, s - 1);
      const wx = xx - x0;

      const iA = ((top + y0) * W + (left + x0)) * 4;
      const iB = ((top + y0) * W + (left + x1)) * 4;
      const iC = ((top + y1) * W + (left + x0)) * 4;
      const iD = ((top + y1) * W + (left + x1)) * 4;
      const o = (oy * size + ox) * 3;
      for (let c = 0; c < 3; c++) {
        const v = data[iA + c] * (1 - wy) * (1 - wx) + data[iB + c] * (1 - wy) * wx
                + data[iC + c] * wy * (1 - wx) + data[iD + c] * wy * wx;
        out[o + c] = Math.min(Math.max(v / 255, 0), 1);
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * 1. Colour space                                                     *
 * ------------------------------------------------------------------ */

export function rgbToHsv(rgb, n) {
  const hsv = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const d = mx - mn;
    let h = 0;
    if (d > EPS) {
      if (mx === r) { h = ((g - b) / d) % 6; if (h < 0) h += 6; }
      else if (mx === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h = (h / 6) % 1;
      if (h < 0) h += 1;
    }
    hsv[i * 3] = h;
    hsv[i * 3 + 1] = mx > EPS ? d / Math.max(mx, EPS) : 0;
    hsv[i * 3 + 2] = mx;
  }
  return hsv;
}

export function toGray(rgb, n) {
  const g = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    g[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
  }
  return g;
}

/* ------------------------------------------------------------------ *
 * 2. Leaf segmentation                                                *
 * ------------------------------------------------------------------ */

function otsuThreshold(q) {
  const hist = new Float64Array(256);
  for (let i = 0; i < q.length; i++) hist[q[i]]++;
  const total = q.length;
  if (total <= 0) return 128;
  let sumAll = 0;
  for (let t = 0; t < 256; t++) sumAll += t * hist[t];
  let wB = 0, sumB = 0, bestVar = -1, bestT = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > bestVar) { bestVar = v; bestT = t; }
  }
  return bestT;
}

function largestComponent(mask, h, w) {
  const lab = new Int32Array(h * w);
  const stack = new Int32Array(h * w);
  let cur = 0, bestLabel = 0, bestSize = 0;
  for (let start = 0; start < h * w; start++) {
    if (!mask[start] || lab[start]) continue;
    cur++;
    let sp = 0, size = 0;
    stack[sp++] = start; lab[start] = cur;
    while (sp) {
      const p = stack[--sp];
      size++;
      const y = (p / w) | 0, x = p % w;
      if (y > 0)     { const q = p - w; if (mask[q] && !lab[q]) { lab[q] = cur; stack[sp++] = q; } }
      if (y < h - 1) { const q = p + w; if (mask[q] && !lab[q]) { lab[q] = cur; stack[sp++] = q; } }
      if (x > 0)     { const q = p - 1; if (mask[q] && !lab[q]) { lab[q] = cur; stack[sp++] = q; } }
      if (x < w - 1) { const q = p + 1; if (mask[q] && !lab[q]) { lab[q] = cur; stack[sp++] = q; } }
    }
    if (size > bestSize) { bestSize = size; bestLabel = cur; }
  }
  if (!bestLabel) return mask;
  const out = new Uint8Array(h * w);
  for (let i = 0; i < h * w; i++) out[i] = lab[i] === bestLabel ? 1 : 0;
  return out;
}

function fillHoles(mask, h, w) {
  const outside = new Uint8Array(h * w);
  const stack = [];
  const push = (y, x) => {
    const p = y * w + x;
    if (!mask[p] && !outside[p]) { outside[p] = 1; stack.push(p); }
  };
  for (let x = 0; x < w; x++) { push(0, x); push(h - 1, x); }
  for (let y = 0; y < h; y++) { push(y, 0); push(y, w - 1); }
  while (stack.length) {
    const p = stack.pop();
    const y = (p / w) | 0, x = p % w;
    if (y > 0) push(y - 1, x);
    if (y < h - 1) push(y + 1, x);
    if (x > 0) push(y, x - 1);
    if (x < w - 1) push(y, x + 1);
  }
  const out = new Uint8Array(h * w);
  for (let i = 0; i < h * w; i++) out[i] = (mask[i] || !outside[i]) ? 1 : 0;
  return out;
}

export function leafMask(rgb, size) {
  const n = size * size;
  const exgr = new Float64Array(n);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    const v = (2 * g - r - b) - (1.4 * r - g);
    exgr[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const full = new Uint8Array(n).fill(1);
  if (hi - lo < 1e-9) return full;

  const q = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    q[i] = Math.min(255, Math.max(0, Math.floor((exgr[i] - lo) / (hi - lo) * 255)));
  }
  const t = otsuThreshold(q);
  let mask = new Uint8Array(n);
  let cnt = 0;
  for (let i = 0; i < n; i++) { mask[i] = q[i] > t ? 1 : 0; cnt += mask[i]; }
  const frac = cnt / n;
  if (frac < 0.05 || frac > 0.98) return full;

  mask = largestComponent(mask, size, size);
  mask = fillHoles(mask, size, size);
  cnt = 0;
  for (let i = 0; i < n; i++) cnt += mask[i];
  if (cnt / n < 0.05) return full;
  return mask;
}

/* ------------------------------------------------------------------ *
 * 3. Feature blocks                                                   *
 * ------------------------------------------------------------------ */

function moments(vals) {
  const n = vals.length;
  if (n === 0) return [0, 0, 0];
  let m = 0;
  for (let i = 0; i < n; i++) m += vals[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (vals[i] - m) * (vals[i] - m);
  const s = Math.sqrt(v / n);
  let sk = 0;
  if (s > EPS) {
    for (let i = 0; i < n; i++) { const z = (vals[i] - m) / (s + EPS); sk += z * z * z; }
    sk /= n;
  }
  return [m, s, sk];
}

function channelValues(src, stride, offset, mask, n) {
  const out = [];
  for (let i = 0; i < n; i++) if (mask[i]) out.push(src[i * stride + offset]);
  return out;
}

function block1(rgb, hsv, mask, n) {
  const f = [];
  for (const [src, off] of [[rgb, 0], [rgb, 1], [rgb, 2], [hsv, 0], [hsv, 1], [hsv, 2]]) {
    f.push(...moments(channelValues(src, 3, off, mask, n)));
  }
  return f;
}

function block2(hsv, mask, n) {
  const hh = new Float64Array(16), hs = new Float64Array(8), hv = new Float64Array(8);
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    cnt++;
    hh[Math.min(15, Math.max(0, Math.trunc(hsv[i * 3] * 16)))]++;
    hs[Math.min(7, Math.max(0, Math.trunc(hsv[i * 3 + 1] * 8)))]++;
    hv[Math.min(7, Math.max(0, Math.trunc(hsv[i * 3 + 2] * 8)))]++;
  }
  const d = Math.max(cnt, 1);
  return [...hh, ...hs, ...hv].map(v => v / d);
}

function block3(hsv, mask, n) {
  let green = 0, yellow = 0, brown = 0, dark = 0, grey = 0, purple = 0, cnt = 0, cov = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i]) cov++;
    if (!mask[i]) continue;
    cnt++;
    const h = hsv[i * 3] * 360, s = hsv[i * 3 + 1], v = hsv[i * 3 + 2];
    if (h >= 75 && h < 160 && s > 0.25 && v > 0.15) green++;
    if (h >= 40 && h < 75 && s > 0.30 && v > 0.30) yellow++;
    if (h >= 15 && h < 40 && s > 0.20 && v >= 0.15 && v < 0.60) brown++;
    if (v < 0.22) dark++;
    if (s < 0.18 && v >= 0.22) grey++;
    if ((h >= 260 || h < 15) && s > 0.15) purple++;
  }
  const d = Math.max(cnt, 1);
  const g = green / d, y = yellow / d, b = brown / d;
  return [g, y, b, dark / d, grey / d, purple / d, cov / n, g / (g + b + y + EPS)];
}

function block4(gray, mask, size) {
  const L = GLCM_LEVELS;
  const q = new Int32Array(size * size);
  for (let i = 0; i < q.length; i++) {
    q[i] = Math.min(L - 1, Math.max(0, Math.trunc(gray[i] * L)));
  }
  const feats = [];
  for (const [dy, dx] of GLCM_OFFSETS) {
    const P = new Float64Array(L * L);
    let total = 0;
    const y0 = Math.max(0, -dy), y1 = size - Math.max(0, dy);
    const x0 = Math.max(0, -dx), x1 = size - Math.max(0, dx);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const p = y * size + x;
        const p2 = (y + dy) * size + (x + dx);
        if (!mask[p] || !mask[p2]) continue;
        P[q[p] * L + q[p2]] += 1;
        total++;
      }
    }
    if (total > 0) {
      // symmetrise then normalise (matches the NumPy reference)
      const S = new Float64Array(L * L);
      let sum = 0;
      for (let i = 0; i < L; i++) {
        for (let j = 0; j < L; j++) {
          const v = P[i * L + j] + P[j * L + i];
          S[i * L + j] = v;
          sum += v;
        }
      }
      for (let k = 0; k < L * L; k++) P[k] = S[k] / sum;
    }
    let contrast = 0, dissim = 0, homog = 0, asm_ = 0, ent = 0, mi = 0, mj = 0;
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        const p = P[i * L + j];
        const dd = i - j;
        contrast += p * dd * dd;
        dissim += p * Math.abs(dd);
        homog += p / (1 + dd * dd);
        asm_ += p * p;
        ent -= p * Math.log(p + EPS);
        mi += p * i;
        mj += p * j;
      }
    }
    let vi = 0, vj = 0, cov = 0;
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        const p = P[i * L + j];
        vi += p * (i - mi) * (i - mi);
        vj += p * (j - mj) * (j - mj);
        cov += p * (i - mi) * (j - mj);
      }
    }
    const corr = cov / (Math.sqrt(vi) * Math.sqrt(vj) + EPS);
    feats.push(contrast, dissim, homog, Math.sqrt(asm_), corr, ent);
  }
  return feats;
}

const LBP_NEIGHBOURS = [[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]];

function block5(gray, mask, size) {
  const hist = new Float64Array(10);
  let cnt = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const p = y * size + x;
      if (!mask[p]) continue;
      const c = gray[p];
      let code = 0;
      for (let bit = 0; bit < 8; bit++) {
        const [dy, dx] = LBP_NEIGHBOURS[bit];
        // np.roll(gray, -dy, 0) then -dx, 1 -> neighbour at (y+dy, x+dx) with wrap
        const ny = (y + dy + size) % size;
        const nx = (x + dx + size) % size;
        if (gray[ny * size + nx] >= c) code |= (1 << bit);
      }
      let transitions = 0, pop = 0;
      for (let b = 0; b < 8; b++) {
        const cb = (code >> b) & 1;
        const nb = (code >> ((b + 1) % 8)) & 1;
        if (cb !== nb) transitions++;
        pop += cb;
      }
      hist[transitions <= 2 ? pop : 9]++;
      cnt++;
    }
  }
  const d = Math.max(cnt, 1);
  return Array.from(hist, v => v / d);
}

function block6(mask, size) {
  let area = 0, minY = size, maxY = -1, minX = size, maxX = -1, sy = 0, sx = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!mask[y * size + x]) continue;
      area++; sy += y; sx += x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
    }
  }
  if (area < 1) return [0, 0, 0, 0, 0];
  let perimeter = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (!mask[p]) continue;
      const interior = y > 0 && y < size - 1 && x > 0 && x < size - 1
        && mask[p - size] && mask[p + size] && mask[p - 1] && mask[p + 1];
      if (!interior) perimeter++;
    }
  }
  const bh = maxY - minY + 1, bw = maxX - minX + 1;
  const cy = sy / area, cx = sx / area;
  let myy = 0, mxx = 0, mxy = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!mask[y * size + x]) continue;
      myy += (y - cy) * (y - cy);
      mxx += (x - cx) * (x - cx);
      mxy += (y - cy) * (x - cx);
    }
  }
  myy /= area; mxx /= area; mxy /= area;
  const tmp = Math.sqrt(Math.max((mxx - myy) * (mxx - myy) + 4 * mxy * mxy, 0));
  const l1 = (mxx + myy + tmp) / 2, l2 = (mxx + myy - tmp) / 2;
  const ecc = Math.sqrt(Math.max(1 - l2 / (l1 + EPS), 0));
  const circ = 4 * Math.PI * area / (perimeter * perimeter + EPS);
  return [area / (size * size), Math.min(circ, 4), area / (bh * bw), ecc, bw / bh];
}

/* ------------------------------------------------------------------ *
 * 4. Public API                                                       *
 * ------------------------------------------------------------------ */

export function featureNames() {
  const names = [];
  for (const ch of ['R', 'G', 'B', 'H', 'S', 'V']) {
    names.push(`B1_${ch}_mean`, `B1_${ch}_std`, `B1_${ch}_skew`);
  }
  for (let i = 0; i < 16; i++) names.push(`B2_H_bin${i}`);
  for (let i = 0; i < 8; i++) names.push(`B2_S_bin${i}`);
  for (let i = 0; i < 8; i++) names.push(`B2_V_bin${i}`);
  names.push('B3_green', 'B3_yellow', 'B3_brown', 'B3_dark',
             'B3_grey', 'B3_purple', 'B3_coverage', 'B3_vitality');
  for (let k = 0; k < GLCM_OFFSETS.length; k++) {
    for (const p of ['contrast', 'dissim', 'homog', 'energy', 'corr', 'entropy']) {
      names.push(`B4_o${k}_${p}`);
    }
  }
  for (let i = 0; i < 10; i++) names.push(`B5_lbp${i}`);
  names.push('B6_area_frac', 'B6_circularity', 'B6_extent',
             'B6_eccentricity', 'B6_aspect');
  return names;
}

/** Returns {vector: Float64Array(97), mask: Uint8Array, size} */
export function extractFeaturesDetailed(imgData, size = IMG_SIZE) {
  const rgb = preprocess(imgData, size);
  const n = size * size;
  const hsv = rgbToHsv(rgb, n);
  const gray = toGray(rgb, n);
  const mask = leafMask(rgb, size);
  const v = [
    ...block1(rgb, hsv, mask, n),
    ...block2(hsv, mask, n),
    ...block3(hsv, mask, n),
    ...block4(gray, mask, size),
    ...block5(gray, mask, size),
    ...block6(mask, size),
  ];
  if (v.length !== N_FEATURES) throw new Error(`expected ${N_FEATURES}, got ${v.length}`);
  const out = Float64Array.from(v, x => (Number.isFinite(x) ? x : 0));
  return { vector: out, mask, size };
}

export function extractFeatures(imgData, size = IMG_SIZE) {
  return extractFeaturesDetailed(imgData, size).vector;
}
