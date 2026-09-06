"""
AgriSense - M1: PlantVillage 3-crop subset manifest builder.

Expected on-disk layout (any of the common PlantVillage mirrors works):

    data/raw/PlantVillage/
        Pepper__bell___Bacterial_spot/
        Pepper__bell___healthy/
        Potato___Early_blight/
        ...
        Tomato___healthy/

Download options (pick one, extract into data/raw/):
    * Kaggle : https://www.kaggle.com/datasets/emmarex/plantdisease
    * GitHub : https://github.com/spMohanty/PlantVillage-Dataset
    * TFDS   : tfds.load('plant_village')
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

CROP_PREFIXES = ("Pepper", "Potato", "Tomato")
IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _pretty(raw: str) -> str:
    s = raw.replace("___", " | ").replace("__", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def find_root(base: str = "data/raw") -> Path:
    base = Path(base)
    for p in [base] + sorted(x for x in base.rglob("*") if x.is_dir()):
        subs = [d for d in p.iterdir() if d.is_dir()] if p.exists() else []
        if sum(d.name.startswith(CROP_PREFIXES) for d in subs) >= 5:
            return p
    raise FileNotFoundError(
        f"No PlantVillage class folders found under {base}. See module docstring."
    )


def build_manifest(base="data/raw", cap_per_class=400, seed=42,
                   splits=(0.70, 0.15, 0.15), out="data/processed/manifest.csv"):
    root = find_root(base)
    rng = np.random.default_rng(seed)
    rows = []
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if not d.name.startswith(CROP_PREFIXES):
            continue
        files = sorted(str(f) for f in d.iterdir() if f.suffix in IMG_EXT)
        if not files:
            continue
        idx = rng.permutation(len(files))[:cap_per_class]
        for i in idx:
            rows.append({"path": files[i], "class_raw": d.name,
                         "label": _pretty(d.name),
                         "crop": d.name.split("_")[0]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root}")

    # deterministic stratified split
    df["split"] = "train"
    for cls, g in df.groupby("class_raw"):
        order = rng.permutation(len(g))
        n = len(g)
        n_tr = int(round(splits[0] * n))
        n_va = int(round(splits[1] * n))
        ids = g.index.to_numpy()[order]
        df.loc[ids[n_tr:n_tr + n_va], "split"] = "val"
        df.loc[ids[n_tr + n_va:], "split"] = "test"

    os.makedirs(Path(out).parent, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    m = build_manifest()
    print(m.shape)
    print(m.groupby(["label", "split"]).size().unstack(fill_value=0))
