"""
AgriSense - M11: Python <-> JavaScript feature parity test.

Why this exists
---------------
The browser must compute EXACTLY the feature vector the model was trained on.
A silent divergence in, say, the resize filter or the GLCM normalisation does not
raise an error - it just makes every browser prediction quietly wrong. This test
is the gate: it renders reference images, extracts features in Python and in
Node.js, and asserts the two agree.

Usage
-----
    python src/parity_test.py                 # synthetic reference images
    python src/parity_test.py path/to/leaf.jpg [more.jpg ...]

Requires Node.js on PATH. Writes reports/parity_report.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

TOL = 1e-4
ROOT = Path(__file__).resolve().parent.parent

NODE_DRIVER = r"""
import { extractFeatures, featureNames } from '../web/features.js';
import { readFileSync, writeFileSync } from 'node:fs';

const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const c of cases) {
  const data = Uint8ClampedArray.from(c.rgba);
  const v = extractFeatures({ data, width: c.width, height: c.height });
  out.push(Array.from(v));
}
writeFileSync(process.argv[3], JSON.stringify({ names: featureNames(), vectors: out }));
"""


def synthetic_images(seed: int = 0):
    """Deterministic stand-ins for leaves: an elliptical blade with lesions."""
    rng = np.random.default_rng(seed)
    images = []
    specs = [
        (220, 220, [(80, 90, 12), (140, 130, 9), (105, 75, 7)], (0.42, 0.28, 0.12)),
        (256, 192, [(96, 140, 18)], (0.55, 0.50, 0.20)),
        (180, 240, [], (0.0, 0.0, 0.0)),
        (200, 200, [(60, 60, 6), (95, 140, 15), (150, 80, 11), (120, 110, 5)], (0.20, 0.16, 0.10)),
    ]
    for h, w, lesions, lesion_rgb in specs:
        img = np.empty((h, w, 3))
        img[..., 0], img[..., 1], img[..., 2] = 0.15, 0.12, 0.14
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        leaf = ((yy - cy) ** 2 / ((0.42 * h) ** 2) + (xx - cx) ** 2 / ((0.28 * w) ** 2)) < 1
        img[leaf] = [0.25, 0.55, 0.18]
        for ly, lx, r in lesions:
            spot = ((yy - ly) ** 2 + (xx - lx) ** 2) < r * r
            img[spot & leaf] = list(lesion_rgb)
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
        images.append((img * 255).astype(np.uint8))
    return images


def to_rgba(img: np.ndarray):
    h, w = img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = 255
    return dict(rgba=rgba.reshape(-1).tolist(), width=int(w), height=int(h))


def main(paths):
    if paths:
        from PIL import Image
        images = []
        for p in paths:
            with Image.open(p) as im:
                images.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        source = list(paths)
    else:
        images = synthetic_images()
        source = [f"synthetic_{i}" for i in range(len(images))]

    py = np.array([F.extract_features(im) for im in images], dtype=np.float64)

    tmp = Path(tempfile.mkdtemp())
    (tmp / "cases.json").write_text(json.dumps([to_rgba(im) for im in images]))
    driver = ROOT / "src" / "_parity_driver.mjs"
    driver.write_text(NODE_DRIVER)
    try:
        subprocess.run(
            ["node", str(driver), str(tmp / "cases.json"), str(tmp / "out.json")],
            check=True, cwd=str(ROOT / "src"),
        )
        res = json.loads((tmp / "out.json").read_text())
    finally:
        driver.unlink(missing_ok=True)

    js = np.array(res["vectors"], dtype=np.float64)
    names = res["names"]

    assert js.shape == py.shape, f"shape mismatch: py {py.shape} vs js {js.shape}"
    assert names == F.feature_names(), "feature name lists differ"

    absdiff = np.abs(py - js)
    scale = np.maximum(np.abs(py), 1.0)
    rel = absdiff / scale
    worst = int(np.argmax(rel.max(axis=0)))

    report = {
        "n_images": len(images),
        "sources": source,
        "max_abs_diff": float(absdiff.max()),
        "max_rel_diff": float(rel.max()),
        "worst_feature": names[worst],
        "worst_feature_rel_diff": float(rel[:, worst].max()),
        "tolerance": TOL,
        "passed": bool(rel.max() < TOL),
    }
    os.makedirs(ROOT / "reports", exist_ok=True)
    (ROOT / "reports" / "parity_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        order = np.argsort(-rel.max(axis=0))[:10]
        print("\nWorst offenders:")
        for i in order:
            print(f"  {names[i]:<24} py={py[0, i]: .8f}  js={js[0, i]: .8f}  rel={rel[:, i].max():.2e}")
        sys.exit(1)
    print("\nPARITY OK - browser inference will match the notebook.")


if __name__ == "__main__":
    main(sys.argv[1:])
