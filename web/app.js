/*
 * AgriSense - M10: UI wiring.
 * Everything runs locally; no network call is made after the page and its two
 * model files have loaded.
 */

import { extractFeaturesDetailed, featureNames, IMG_SIZE } from './features.js';
import { loadModel, predictProba, predictValue, topK, linearContributions } from './inference.js';
import { buildIrrigationFeatures, fuse, STAGES, SOILS } from './advisory.js';

const $ = id => document.getElementById(id);
const fmt = (v, n = 1) => Number(v).toFixed(n);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const state = {
  diseaseModel: null,
  irrClf: null,
  irrReg: null,
  lastFeatures: null,
  lastDisease: null,
  lastIrrigation: null,
  names: featureNames(),
};

/* ------------------------------------------------------------------ *
 * Boot                                                                *
 * ------------------------------------------------------------------ */

async function boot() {
  const chips = [];
  try {
    state.irrClf = await loadModel('models/irrigation_classifier.json');
    state.irrReg = await loadModel('models/irrigation_regressor.json');
    const m = state.irrClf.meta || {};
    chips.push(`<span class="chip ok">Irrigation model · acc ${fmt((m.test_accuracy ?? 0) * 100, 1)}%</span>`);
  } catch (e) {
    chips.push('<span class="chip bad">Irrigation model failed to load</span>');
    console.error(e);
  }
  try {
    state.diseaseModel = await loadModel('models/disease_model.json');
    const m = state.diseaseModel.meta || {};
    chips.push(`<span class="chip ok">Disease model · ${state.diseaseModel.labels.length} classes` +
               (m.test_macro_f1 ? ` · macro-F1 ${fmt(m.test_macro_f1 * 100, 1)}%` : '') + '</span>');
  } catch {
    chips.push('<span class="chip bad">Disease model not trained yet</span>');
    $('noModelNote').hidden = false;
  }
  chips.push('<span class="chip">97 features · offline</span>');
  $('statusChips').innerHTML = chips.join('');
  renderAbout();
  updateIrrigation();
}

/* ------------------------------------------------------------------ *
 * Tabs                                                                *
 * ------------------------------------------------------------------ */

document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('is-active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('is-active'));
    t.classList.add('is-active');
    $(t.dataset.panel).classList.add('is-active');
    if (t.dataset.panel === 'panel-advisory') renderAdvisory();
  });
});

/* ------------------------------------------------------------------ *
 * Image input                                                         *
 * ------------------------------------------------------------------ */

const dz = $('dropzone');
['dragenter', 'dragover'].forEach(e =>
  dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach(e =>
  dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', ev => {
  const f = ev.dataTransfer?.files?.[0];
  if (f) handleImage(f);
});
$('fileInput').addEventListener('change', e => e.target.files[0] && handleImage(e.target.files[0]));
$('camInput').addEventListener('change', e => e.target.files[0] && handleImage(e.target.files[0]));

$('modelInput').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const m = JSON.parse(await f.text());
    if (m.format !== 'agrisense-model/1') throw new Error('unrecognised model format');
    state.diseaseModel = m;
    $('noModelNote').hidden = true;
    document.querySelector('.chip.bad')?.remove();
    $('statusChips').insertAdjacentHTML('beforeend',
      `<span class="chip ok">Disease model · ${m.labels.length} classes (loaded from file)</span>`);
    renderAbout();
    if (state.lastFeatures) runDisease(state.lastFeatures);
  } catch (err) {
    alert('Could not read that model file: ' + err.message);
  }
});

async function handleImage(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  c.getContext('2d', { willReadFrequently: true }).drawImage(bmp, 0, 0);
  const imgData = c.getContext('2d').getImageData(0, 0, bmp.width, bmp.height);

  const t0 = performance.now();
  const { vector, mask, size } = extractFeaturesDetailed(imgData, IMG_SIZE);
  const ms = performance.now() - t0;

  drawPreview(imgData, mask, size);
  state.lastFeatures = vector;
  renderFeatureTable(vector);
  $('featDetails').hidden = false;
  runDisease(vector, ms);
}

function drawPreview(imgData, mask, size) {
  // source: re-run the same centre-crop + bilinear path via a scratch canvas
  const src = $('canvasSrc').getContext('2d');
  const tmp = document.createElement('canvas');
  tmp.width = imgData.width; tmp.height = imgData.height;
  tmp.getContext('2d').putImageData(imgData, 0, 0);
  const s = Math.min(imgData.width, imgData.height);
  const top = Math.floor((imgData.height - s) / 2);
  const left = Math.floor((imgData.width - s) / 2);
  src.clearRect(0, 0, size, size);
  src.drawImage(tmp, left, top, s, s, 0, 0, size, size);

  const mctx = $('canvasMask').getContext('2d');
  const out = mctx.createImageData(size, size);
  const base = src.getImageData(0, 0, size, size).data;
  for (let i = 0; i < size * size; i++) {
    const on = mask[i];
    out.data[i * 4]     = on ? base[i * 4]     : 22;
    out.data[i * 4 + 1] = on ? base[i * 4 + 1] : 26;
    out.data[i * 4 + 2] = on ? base[i * 4 + 2] : 22;
    out.data[i * 4 + 3] = 255;
  }
  mctx.putImageData(out, 0, 0);
}

/* ------------------------------------------------------------------ *
 * Disease panel                                                       *
 * ------------------------------------------------------------------ */

function runDisease(vector, ms) {
  const out = $('diseaseOut');
  if (!state.diseaseModel) {
    state.lastDisease = null;
    out.innerHTML = `<div class="note info">Features extracted in ${fmt(ms ?? 0, 0)} ms.
      Segmentation is shown on the left. Train the classifier to get a diagnosis.</div>`;
    renderAdvisory();
    return;
  }
  const t0 = performance.now();
  const top = topK(state.diseaseModel, vector, 3);
  const infMs = performance.now() - t0;
  state.lastDisease = top[0];

  const bars = top.map(t => `
    <div class="bar">
      <span>${esc(prettyLabel(t.label))}</span><span class="pct">${fmt(t.prob * 100, 1)}%</span>
      <div class="track"><div class="fill" style="width:${(t.prob * 100).toFixed(1)}%"></div></div>
    </div>`).join('');

  const cls = top[0].prob >= 0.45
    ? (/healthy/i.test(top[0].label) ? 'go' : 'alert')
    : 'hold';

  let contrib = '';
  const idx = state.diseaseModel.labels.indexOf(top[0].label);
  const lc = linearContributions(state.diseaseModel, vector, idx, state.names);
  if (lc) {
    contrib = `<details><summary>What drove this prediction</summary><div class="feattable">` +
      lc.slice(0, 12).map(c =>
        `<div><span>${esc(c.name)}</span><span>${c.value >= 0 ? '+' : ''}${fmt(c.value, 3)}</span></div>`
      ).join('') + `</div></details>`;
  }

  out.innerHTML = `
    <div class="verdict ${cls}">
      <span class="big">${esc(prettyLabel(top[0].label))}</span>
      <span class="sub">${fmt(top[0].prob * 100, 1)}% confidence</span>
    </div>
    ${top[0].prob < 0.45 ? '<div class="note warn">Low confidence — re-photograph the leaf against a plain background, filling the frame, in even light.</div>' : ''}
    <div class="bars">${bars}</div>
    <p class="muted small">Features ${fmt(ms ?? 0, 0)} ms · inference ${fmt(infMs, 1)} ms · all on this device</p>
    ${contrib}`;
  renderAdvisory();
}

function prettyLabel(l) {
  return String(l).replace(/_{2,}/g, ' | ').replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
}

function renderFeatureTable(v) {
  $('featTable').innerHTML = state.names
    .map((n, i) => `<div><span>${esc(n)}</span><span>${fmt(v[i], 4)}</span></div>`).join('');
}

/* ------------------------------------------------------------------ *
 * Irrigation panel                                                    *
 * ------------------------------------------------------------------ */

const RANGE_UNITS = {
  in_vwc: ['o_vwc', v => `${v}% VWC`],
  in_stemp: ['o_stemp', v => `${v} °C`],
  in_atemp: ['o_atemp', v => `${v} °C`],
  in_rh: ['o_rh', v => `${v}%`],
  in_rain: ['o_rain', v => `${v} mm`],
  in_fcast: ['o_fcast', v => `${v} mm`],
  in_et0: ['o_et0', v => `${v} mm/day`],
  in_days: ['o_days', v => `${v} d`],
};

Object.keys(RANGE_UNITS).concat(['in_soil', 'in_stage']).forEach(id => {
  $(id).addEventListener('input', updateIrrigation);
});

document.querySelectorAll('[data-preset]').forEach(b => {
  b.addEventListener('click', () => {
    const p = {
      dry:        { in_vwc: 13, in_rain: 0, in_fcast: 0, in_et0: 6.2, in_days: 9, in_rh: 38, in_atemp: 36 },
      wet:        { in_vwc: 27, in_rain: 22, in_fcast: 14, in_et0: 2.6, in_days: 1, in_rh: 84, in_atemp: 27 },
      borderline: { in_vwc: 20, in_rain: 1, in_fcast: 4, in_et0: 4.8, in_days: 5, in_rh: 55, in_atemp: 32 },
    }[b.dataset.preset];
    Object.entries(p).forEach(([k, v]) => { $(k).value = v; });
    updateIrrigation();
  });
});

function readInputs() {
  Object.entries(RANGE_UNITS).forEach(([id, [outId, f]]) => { $(outId).textContent = f($(id).value); });
  return {
    soil: $('in_soil').value,
    stage: $('in_stage').value,
    soil_vwc_pct: +$('in_vwc').value,
    soil_temp_c: +$('in_stemp').value,
    air_temp_c: +$('in_atemp').value,
    humidity_pct: +$('in_rh').value,
    rain_24h_mm: +$('in_rain').value,
    rain_forecast_mm: +$('in_fcast').value,
    et0_mm: +$('in_et0').value,
    days_since_irrigation: +$('in_days').value,
  };
}

function updateIrrigation() {
  const o = readInputs();
  const { vector, derived } = buildIrrigationFeatures(o);
  const box = $('irrigationOut');

  if (!state.irrClf || !state.irrReg) {
    box.innerHTML = '<div class="note warn">Irrigation models could not be loaded. ' +
      'Serve this folder over HTTP (<code>python -m http.server</code>) rather than opening the file directly.</div>';
    return;
  }

  const p = predictProba(state.irrClf, vector)[1];
  const thr = state.irrClf.meta?.decision_threshold ?? 0.5;
  const irrigate = p >= thr;
  const depth = Math.max(0, predictValue(state.irrReg, vector));

  state.lastIrrigation = { irrigate, probability: p, depth_mm: irrigate ? depth : 0, derived };

  const pctOfRaw = Math.min(derived.depletion / Math.max(derived.raw, 1e-6), 1.4);
  box.innerHTML = `
    <div class="verdict ${irrigate ? 'go' : 'hold'}">
      <span class="big">${irrigate ? 'Irrigate today' : 'Hold — no irrigation'}</span>
      <span class="sub">${irrigate ? `${fmt(depth, 1)} mm` : `p(irrigate) = ${fmt(p, 2)} &lt; ${fmt(thr, 2)}`}</span>
    </div>
    <div class="gauge">
      <div class="fill" style="width:${(Math.min(pctOfRaw, 1) * 100).toFixed(0)}%"></div>
      <div class="mark" style="left:50%"></div>
    </div>
    <div class="gaugelab"><span>field capacity</span><span>trigger (½ RAW)</span><span>RAW limit</span></div>
    <dl class="kv">
      <dt>Root-zone depletion D<sub>r</sub></dt><dd>${fmt(derived.depletion, 1)} mm</dd>
      <dt>Readily available water RAW</dt><dd>${fmt(derived.raw, 1)} mm</dd>
      <dt>Total available water TAW</dt><dd>${fmt(derived.taw, 1)} mm</dd>
      <dt>Crop water use ET<sub>c</sub></dt><dd>${fmt(derived.etc, 2)} mm/day</dd>
      <dt>Projected end-of-day deficit</dt><dd>${fmt(derived.projected, 1)} mm</dd>
      <dt>Trigger depth</dt><dd>${fmt(derived.trigger, 1)} mm</dd>
      <dt>Model probability</dt><dd>${fmt(p, 3)}</dd>
      <dt>Soil / stage</dt><dd>${SOILS[o.soil].label} · ${STAGES[o.stage].label} (K<sub>c</sub> ${STAGES[o.stage].kc}, Z<sub>r</sub> ${STAGES[o.stage].zr} m)</dd>
    </dl>`;
  renderAdvisory();
}

/* ------------------------------------------------------------------ *
 * Fused advisory                                                      *
 * ------------------------------------------------------------------ */

function renderAdvisory() {
  const box = $('advisoryOut');
  if (!state.lastIrrigation) { box.innerHTML = '<div class="empty">Set the field inputs first.</div>'; return; }

  const f = fuse(state.lastIrrigation, state.lastDisease);
  const p = f.pathogen;
  const severe = p && (p.severity === 'high');

  box.innerHTML = `
    <div class="verdict ${f.irrigate ? (severe ? 'alert' : 'go') : 'hold'}">
      <span class="big">${f.irrigate ? `Irrigate ${fmt(f.depthMm, 1)} mm` : 'Hold irrigation'}</span>
      <span class="sub">${p ? esc(p.name) : 'no diagnosis yet — run a leaf image in tab 1'}</span>
    </div>
    <dl class="kv">
      <dt>Method</dt><dd>${esc(f.method)}</dd>
      <dt>Timing</dt><dd>${esc(f.timing)}</dd>
      <dt>Interval</dt><dd>${esc(f.intervalNote)}</dd>
      ${p && p.agent !== '-' ? `<dt>Causal agent</dt><dd>${esc(p.agent)}</dd>` : ''}
      ${p && p.signs !== '-' ? `<dt>What to look for</dt><dd>${esc(p.signs)}</dd>` : ''}
      ${p ? `<dt>Crop action</dt><dd>${esc(p.action)}</dd>` : ''}
    </dl>
    ${f.notes.map(n => `<div class="note info">${esc(n)}</div>`).join('')}
    ${!state.lastDisease ? '<div class="note info">No leaf diagnosed yet, so no disease modifier is applied. The irrigation decision above is the unmodified model output.</div>' : ''}`;
}

/* ------------------------------------------------------------------ *
 * About                                                               *
 * ------------------------------------------------------------------ */

function renderAbout() {
  const rows = [];
  const add = (name, m) => {
    if (!m) { rows.push(`<dt>${name}</dt><dd>not loaded</dd>`); return; }
    const meta = m.meta || {};
    const bits = [`kind: ${m.kind}`];
    if (m.labels) bits.push(`${m.labels.length} classes`);
    if (meta.test_accuracy != null) bits.push(`accuracy ${fmt(meta.test_accuracy * 100, 1)}%`);
    if (meta.test_f1 != null) bits.push(`F1 ${fmt(meta.test_f1, 3)}`);
    if (meta.test_macro_f1 != null) bits.push(`macro-F1 ${fmt(meta.test_macro_f1, 3)}`);
    if (meta.test_mae_mm != null) bits.push(`MAE ${meta.test_mae_mm} mm`);
    if (meta.test_r2 != null) bits.push(`R² ${meta.test_r2}`);
    rows.push(`<dt>${name}</dt><dd>${esc(bits.join(' · '))}</dd>`);
  };
  add('Disease classifier', state.diseaseModel);
  add('Irrigation classifier', state.irrClf);
  add('Irrigation depth regressor', state.irrReg);
  $('aboutOut').innerHTML = `<dl class="kv">${rows.join('')}</dl>`;
}

boot();
