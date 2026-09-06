"""
AgriSense - M7: serialise a scikit-learn pipeline to JSON that web/inference.js
can evaluate. Two model families are supported:

  "linear"  : StandardScaler [-> PCA] -> LogisticRegression / LinearSVC
              exported as a coefficient matrix; a few kB.
  "forest"  : StandardScaler [-> PCA] -> RandomForest / ExtraTrees / DecisionTree
              exported as flat per-tree arrays; a few hundred kB.

Anything else must be converted to one of these before deployment.
"""

from __future__ import annotations

import json
import numpy as np


def _round(a, nd=None):
    """Round for size. nd=None keeps full float64 precision.

    IMPORTANT: scaler/PCA parameters and tree thresholds are NEVER rounded.
    A 1e-6 perturbation of a scaled feature can flip a near-boundary tree
    comparison, which changes the leaf a sample lands in and therefore the
    prediction. Leaf values are safe to round - they only affect the output
    magnitude, not the path taken.
    """
    a = np.asarray(a, dtype=np.float64)
    return (a if nd is None else np.round(a, nd)).tolist()


def _preproc(pipe):
    out = {}
    steps = dict(pipe.named_steps) if hasattr(pipe, "named_steps") else {}
    sc = steps.get("scaler")
    if sc is not None:
        out["scaler"] = {"mean": _round(sc.mean_), "scale": _round(sc.scale_)}
    pca = steps.get("pca")
    if pca is not None:
        out["pca"] = {"mean": _round(pca.mean_),
                      "components": [_round(c) for c in pca.components_]}
    return out


def _tree(t):
    tr = t.tree_
    value = tr.value.reshape(tr.node_count, -1)
    value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)
    return {
        "left": tr.children_left.astype(int).tolist(),
        "right": tr.children_right.astype(int).tolist(),
        "feature": tr.feature.astype(int).tolist(),
        "threshold": _round(tr.threshold),
        "value": [_round(v, 6) for v in value],
    }


def export_model(pipe, labels, kind, meta=None, path=None):
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    doc = {
        "format": "agrisense-model/1",
        "kind": kind,
        "labels": list(labels),
        "meta": meta or {},
    }
    doc.update(_preproc(pipe))

    if kind == "linear":
        coef = np.atleast_2d(clf.coef_)
        inter = np.atleast_1d(clf.intercept_)
        if coef.shape[0] == 1 and len(labels) == 2:
            # Binary sklearn models store a single row. softmax over
            # [0, z] reproduces sigmoid(z) exactly; softmax over [-z, z]
            # would give sigmoid(2z), which is a different model.
            coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
            inter = np.array([0.0, inter[0]])
        doc["coef"] = [_round(r, 8) for r in coef]
        doc["intercept"] = _round(inter, 8)
        doc["decision"] = "softmax" if hasattr(clf, "predict_proba") else "margin"
    elif kind == "forest":
        ests = getattr(clf, "estimators_", [clf])
        doc["trees"] = [_tree(e) for e in ests]
    else:
        raise ValueError(f"unsupported kind: {kind!r}")

    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
    return doc


def export_regressor(pipe, kind, meta=None, path=None):
    """Same idea for a scalar regressor (irrigation depth in mm)."""
    reg = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    doc = {"format": "agrisense-model/1", "kind": f"{kind}_regressor",
           "meta": meta or {}}
    doc.update(_preproc(pipe))
    if kind == "linear":
        doc["coef"] = _round(np.atleast_1d(reg.coef_).ravel(), 8)
        doc["intercept"] = float(np.atleast_1d(reg.intercept_)[0])
    elif kind == "forest":
        trees = []
        for e in getattr(reg, "estimators_", [reg]):
            tr = e.tree_
            trees.append({
                "left": tr.children_left.astype(int).tolist(),
                "right": tr.children_right.astype(int).tolist(),
                "feature": tr.feature.astype(int).tolist(),
                "threshold": _round(tr.threshold),
                "value": _round(tr.value.reshape(-1), 6),
            })
        doc["trees"] = trees
    else:
        raise ValueError(kind)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
    return doc


def predict_json(doc, X):
    """Reference implementation of the JSON evaluator - the JS must match this.
    Used by the parity test to prove inference.js is correct."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if "scaler" in doc:
        X = (X - np.array(doc["scaler"]["mean"])) / np.array(doc["scaler"]["scale"])
    if "pca" in doc:
        X = (X - np.array(doc["pca"]["mean"])) @ np.array(doc["pca"]["components"]).T
    kind = doc["kind"]
    if kind == "linear":
        z = X @ np.array(doc["coef"]).T + np.array(doc["intercept"])
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    if kind == "forest":
        acc = None
        for t in doc["trees"]:
            out = _walk(t, X, np.array(t["value"]))
            acc = out if acc is None else acc + out
        return acc / len(doc["trees"])
    if kind == "linear_regressor":
        return X @ np.array(doc["coef"]) + doc["intercept"]
    if kind == "forest_regressor":
        acc = None
        for t in doc["trees"]:
            out = _walk(t, X, np.array(t["value"])[:, None])[:, 0]
            acc = out if acc is None else acc + out
        return acc / len(doc["trees"])
    raise ValueError(kind)


def _walk(t, X, values):
    """Traverse one exported tree.

    NOTE: scikit-learn casts X to float32 inside its tree traversal, so the
    comparison at each node is float32(x) <= float64(threshold). We reproduce
    that cast here, and web/inference.js does the same with Math.fround().
    Without it, samples sitting within ~1e-7 of a split go the other way and
    the browser silently disagrees with the notebook.
    """
    left = np.array(t["left"]); right = np.array(t["right"])
    feat = np.array(t["feature"]); thr = np.array(t["threshold"])
    X32 = np.asarray(X, dtype=np.float32)
    out = np.zeros((X.shape[0], values.shape[1]))
    for i in range(X32.shape[0]):
        x = X32[i]
        n = 0
        while left[n] != -1:
            n = left[n] if float(x[feat[n]]) <= thr[n] else right[n]
        out[i] = values[n]
    return out
