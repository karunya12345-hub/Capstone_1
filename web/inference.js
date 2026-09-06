/*
 * AgriSense - M8: generic evaluator for models exported by src/export_json.py.
 *
 * Supported "kind" values:
 *   linear            softmax(W x + b)      -> class probabilities
 *   forest            mean of tree leaf distributions -> class probabilities
 *   linear_regressor  w . x + b             -> scalar
 *   forest_regressor  mean of tree leaf values -> scalar
 *
 * Pre-processing (scaler, then optional PCA) is applied first if present.
 */

/** Apply the exported StandardScaler / PCA, if any. */
function preprocess(model, x) {
  let v = Float64Array.from(x);
  const sc = model.scaler;
  if (sc) {
    for (let i = 0; i < v.length; i++) v[i] = (v[i] - sc.mean[i]) / sc.scale[i];
  }
  const pca = model.pca;
  if (pca) {
    const out = new Float64Array(pca.components.length);
    for (let k = 0; k < pca.components.length; k++) {
      const c = pca.components[k];
      let s = 0;
      for (let i = 0; i < c.length; i++) s += (v[i] - pca.mean[i]) * c[i];
      out[k] = s;
    }
    v = out;
  }
  return v;
}

/*
 * scikit-learn casts X to float32 inside its tree traversal, so every node
 * comparison is float32(x) <= float64(threshold). Math.fround reproduces that
 * cast. Without it, a sample within ~1e-7 of a split point takes the other
 * branch and the browser silently disagrees with the notebook.
 */
function walkTree(tree, v) {
  let n = 0;
  while (tree.left[n] !== -1) {
    n = Math.fround(v[tree.feature[n]]) <= tree.threshold[n] ? tree.left[n] : tree.right[n];
  }
  return tree.value[n];
}

function softmax(z) {
  let m = -Infinity;
  for (const t of z) if (t > m) m = t;
  const e = z.map(t => Math.exp(t - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(t => t / s);
}

/** Class probabilities as a plain Array. */
export function predictProba(model, x) {
  const v = preprocess(model, x);
  if (model.kind === 'linear') {
    const z = model.coef.map((row, k) => {
      let s = model.intercept[k];
      for (let i = 0; i < row.length; i++) s += row[i] * v[i];
      return s;
    });
    return softmax(z);
  }
  if (model.kind === 'forest') {
    const K = model.labels.length;
    const acc = new Array(K).fill(0);
    for (const t of model.trees) {
      const leaf = walkTree(t, v);
      for (let k = 0; k < K; k++) acc[k] += leaf[k];
    }
    return acc.map(a => a / model.trees.length);
  }
  throw new Error(`predictProba: unsupported kind ${model.kind}`);
}

/** Scalar prediction for the two regressor kinds. */
export function predictValue(model, x) {
  const v = preprocess(model, x);
  if (model.kind === 'linear_regressor') {
    let s = model.intercept;
    for (let i = 0; i < model.coef.length; i++) s += model.coef[i] * v[i];
    return s;
  }
  if (model.kind === 'forest_regressor') {
    let s = 0;
    for (const t of model.trees) s += walkTree(t, v);
    return s / model.trees.length;
  }
  throw new Error(`predictValue: unsupported kind ${model.kind}`);
}

/** [{label, prob}] sorted best-first, truncated to k. */
export function topK(model, x, k = 3) {
  const p = predictProba(model, x);
  return p
    .map((prob, i) => ({ label: model.labels[i], prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k);
}

/**
 * Contribution of each input feature to the winning class, for linear models.
 * Returns the signed term coef[cls][i] * scaled_x[i], which is exactly what
 * moved the decision - the interpretability that a CNN cannot give directly.
 */
export function linearContributions(model, x, classIndex, featureNames) {
  if (model.kind !== 'linear' || model.pca) return null;
  const v = preprocess(model, x);
  const row = model.coef[classIndex];
  return row
    .map((c, i) => ({ name: featureNames?.[i] ?? `f${i}`, value: c * v[i] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

export async function loadModel(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const m = await r.json();
  if (m.format !== 'agrisense-model/1') throw new Error('unrecognised model format');
  return m;
}
