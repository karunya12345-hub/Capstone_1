/*
 * AgriSense - M9: FAO-56 physics features (mirror of irrigation_sim.add_physics_features)
 * plus the pathogen-conditional fusion decision table.
 *
 * The fusion table is the part of this project that has no counterpart in the
 * surveyed literature (see docs/02_Literature_Survey.md section 7): the detected
 * disease does not change WHETHER the soil needs water, it changes HOW and WHEN
 * that water should be delivered.
 */

/* ------------------------------------------------------------------ *
 * FAO-56 soil and crop constants - must match src/irrigation_sim.py    *
 * ------------------------------------------------------------------ */

export const SOILS = {
  sandy:  { theta_fc: 0.16, theta_wp: 0.07, p: 0.60, label: 'Sandy' },
  loamy:  { theta_fc: 0.28, theta_wp: 0.14, p: 0.55, label: 'Loamy' },
  clayey: { theta_fc: 0.38, theta_wp: 0.23, p: 0.45, label: 'Clayey' },
};

export const STAGES = {
  initial:     { kc: 0.60, zr: 0.25, label: 'Initial (establishment)' },
  development: { kc: 0.90, zr: 0.45, label: 'Development' },
  mid:         { kc: 1.15, zr: 0.80, label: 'Mid-season (peak demand)' },
  late:        { kc: 0.85, zr: 1.00, label: 'Late season' },
  harvest:     { kc: 0.70, zr: 1.00, label: 'Harvest / ripening' },
};

export const TRIGGER_FRACTION = 0.50;

/** Order must match irrigation_sim.FEATURE_COLUMNS exactly. */
export const FEATURE_COLUMNS = [
  'soil_vwc_pct', 'soil_temp_c', 'air_temp_c', 'humidity_pct',
  'rain_24h_mm', 'rain_forecast_mm', 'et0_mm', 'kc', 'days_since_irrigation',
  'soil_sandy', 'soil_loamy', 'soil_clayey',
  'root_depth_m', 'etc_mm', 'est_taw_mm', 'est_raw_mm', 'est_depletion_mm',
  'depletion_ratio', 'projected_deficit_mm', 'projected_deficit_ratio',
  'forecast_adj_ratio',
];

/**
 * Build the model input row from the grower-facing inputs.
 * @param {{soil:string, stage:string, soil_vwc_pct:number, soil_temp_c:number,
 *          air_temp_c:number, humidity_pct:number, rain_24h_mm:number,
 *          rain_forecast_mm:number, et0_mm:number, days_since_irrigation:number}} o
 */
export function buildIrrigationFeatures(o) {
  const s = SOILS[o.soil];
  const st = STAGES[o.stage];
  const zr = st.zr;

  const taw = 1000 * (s.theta_fc - s.theta_wp) * zr;
  const raw = s.p * taw;
  const depletion = Math.max(1000 * (s.theta_fc - o.soil_vwc_pct / 100) * zr, 0);
  const etc = st.kc * o.et0_mm;
  const projected = depletion + etc - o.rain_24h_mm;
  const trigger = TRIGGER_FRACTION * raw;
  const credit = Math.min(o.rain_forecast_mm, trigger);

  const row = {
    soil_vwc_pct: o.soil_vwc_pct,
    soil_temp_c: o.soil_temp_c,
    air_temp_c: o.air_temp_c,
    humidity_pct: o.humidity_pct,
    rain_24h_mm: o.rain_24h_mm,
    rain_forecast_mm: o.rain_forecast_mm,
    et0_mm: o.et0_mm,
    kc: st.kc,
    days_since_irrigation: Math.min(o.days_since_irrigation, 21),
    soil_sandy: o.soil === 'sandy' ? 1 : 0,
    soil_loamy: o.soil === 'loamy' ? 1 : 0,
    soil_clayey: o.soil === 'clayey' ? 1 : 0,
    root_depth_m: zr,
    etc_mm: etc,
    est_taw_mm: taw,
    est_raw_mm: raw,
    est_depletion_mm: depletion,
    depletion_ratio: depletion / Math.max(raw, 1e-6),
    projected_deficit_mm: projected,
    projected_deficit_ratio: projected / Math.max(raw, 1e-6),
    forecast_adj_ratio: (projected - credit) / Math.max(raw, 1e-6),
  };
  return {
    vector: FEATURE_COLUMNS.map(k => row[k]),
    derived: { taw, raw, trigger, depletion, etc, projected },
  };
}

/* ------------------------------------------------------------------ *
 * Pathogen knowledge base                                             *
 * ------------------------------------------------------------------ */

/*
 * Keyed by substring so the table survives the different class-name
 * conventions used by the various PlantVillage mirrors
 * ("Tomato_Late_blight", "Tomato___Late_blight", "Tomato | Late blight", ...).
 */
const PATHOGENS = [
  {
    match: ['late blight'],
    name: 'Late blight',
    agent: 'Phytophthora infestans (oomycete)',
    group: 'wetness-driven',
    severity: 'high',
    signs: 'Water-soaked grey-green lesions spreading fast, white sporulation on the leaf underside in humid weather.',
    action: 'Act within 24-48 h. Remove and destroy affected foliage; apply a protectant fungicide on a preventative schedule during cool, wet spells. Do not compost infected material.',
    irrigation: {
      method: 'Drip or furrow only - stop all overhead irrigation',
      timing: 'Early morning, so the canopy dries within the day',
      depthFactor: 1.0,
      intervalNote: 'Keep the soil-water schedule unchanged; a water-stressed plant defends itself less well.',
      why: 'Sporangia need several hours of continuous leaf wetness to germinate. Overhead or evening irrigation supplies exactly that.',
    },
  },
  {
    match: ['early blight'],
    name: 'Early blight',
    agent: 'Alternaria solani (fungus)',
    group: 'wetness-driven',
    severity: 'medium',
    signs: 'Dark concentric "target" rings on older, lower leaves; yellow halo; progresses upward.',
    action: 'Remove lower affected leaves, mulch to stop soil splash, maintain nitrogen, rotate away from solanaceous crops next season.',
    irrigation: {
      method: 'Drip preferred; mulch the bed surface',
      timing: 'Early morning',
      depthFactor: 1.0,
      intervalNote: 'Unchanged. Fewer, deeper irrigations beat frequent shallow ones here.',
      why: 'Conidia are splash-dispersed from infected soil and debris; mulch plus drip breaks that path.',
    },
  },
  {
    match: ['septoria'],
    name: 'Septoria leaf spot',
    agent: 'Septoria lycopersici (fungus)',
    group: 'wetness-driven',
    severity: 'medium',
    signs: 'Many small circular spots with pale centres and dark margins, lowest leaves first.',
    action: 'Strip the worst lower leaves, improve air movement through the canopy, mulch, and apply a protectant fungicide if spread continues.',
    irrigation: {
      method: 'Drip only',
      timing: 'Early morning',
      depthFactor: 1.0,
      intervalNote: 'Unchanged.',
      why: 'Spores are released and dispersed by splashing water and require leaf wetness to infect.',
    },
  },
  {
    match: ['leaf mold', 'leaf mould'],
    name: 'Leaf mould',
    agent: 'Passalora fulva / Fulvia fulva (fungus)',
    group: 'humidity-driven',
    severity: 'medium',
    signs: 'Pale yellow blotches on the upper surface with olive-green velvety growth underneath.',
    action: 'Increase ventilation - this is primarily a protected-cultivation disease. Reduce canopy density, remove affected leaves.',
    irrigation: {
      method: 'Drip only',
      timing: 'Early morning',
      depthFactor: 0.95,
      intervalNote: 'Slightly longer interval is acceptable to lower canopy humidity, provided depletion stays below the trigger.',
      why: 'Infection needs relative humidity above roughly 85%. Every litre applied overhead raises it further.',
    },
  },
  {
    match: ['target spot'],
    name: 'Target spot',
    agent: 'Corynespora cassiicola (fungus)',
    group: 'wetness-driven',
    severity: 'medium',
    signs: 'Brown lesions with concentric rings, on leaves, stems and fruit alike.',
    action: 'Remove affected tissue, improve airflow, apply a protectant fungicide where pressure is high.',
    irrigation: {
      method: 'Drip only',
      timing: 'Early morning',
      depthFactor: 1.0,
      intervalNote: 'Unchanged.',
      why: 'Prolonged leaf wetness drives sporulation and infection.',
    },
  },
  {
    match: ['bacterial spot'],
    name: 'Bacterial spot',
    agent: 'Xanthomonas spp. (bacterium)',
    group: 'splash-dispersed',
    severity: 'high',
    signs: 'Small dark water-soaked spots, often with a yellow halo; scabby lesions on fruit.',
    action: 'No curative treatment. Use certified seed, remove affected plants, avoid working the crop while foliage is wet, consider copper-based protectants.',
    irrigation: {
      method: 'Drip only - overhead irrigation actively spreads this pathogen plant to plant',
      timing: 'Early morning',
      depthFactor: 1.0,
      intervalNote: 'Extend the interval slightly where soil water allows; never work in a wet canopy.',
      why: 'The bacterium is dispersed by water splash and enters through wounds and stomata in wet tissue.',
    },
  },
  {
    match: ['spider mite', 'two-spotted', 'two spotted'],
    name: 'Two-spotted spider mite',
    agent: 'Tetranychus urticae (arthropod pest)',
    group: 'drought-favoured',
    severity: 'medium',
    signs: 'Fine pale stippling on the upper surface, bronzing, webbing on the underside.',
    action: 'Introduce or conserve predatory mites, wash the canopy, use a miticide only if numbers keep rising. Avoid broad-spectrum insecticides, which kill the predators.',
    irrigation: {
      method: 'Overhead irrigation is HELPFUL here - a canopy wash suppresses mite populations',
      timing: 'Mid-morning, so foliage still dries before evening',
      depthFactor: 1.10,
      intervalNote: 'Shorten the interval. This is the one case where the advisory increases water.',
      why: 'Mite populations explode in hot, dry, water-stressed canopies. Relieving that stress is itself a control measure.',
    },
  },
  {
    match: ['yellow leaf curl', 'yellowleaf'],
    name: 'Tomato yellow leaf curl virus',
    agent: 'TYLCV, transmitted by whitefly (Bemisia tabaci)',
    group: 'viral',
    severity: 'high',
    signs: 'Upward leaf curling, marginal yellowing, stunting, heavy flower drop.',
    action: 'No cure. Control whitefly, use resistant varieties and reflective mulch, remove infected plants promptly (bag them - do not shake).',
    irrigation: {
      method: 'Unchanged',
      timing: 'Unchanged',
      depthFactor: 1.05,
      intervalNote: 'Keep the plant comfortable; avoid additional water stress on top of the infection.',
      why: 'Irrigation does not move the virus. Water stress does worsen symptom expression and yield loss.',
    },
  },
  {
    match: ['mosaic'],
    name: 'Tomato mosaic virus',
    agent: 'ToMV (mechanically transmitted)',
    group: 'viral',
    severity: 'high',
    signs: 'Mottled light and dark green mosaic, leaf distortion, fern-leaf narrowing.',
    action: 'No cure. Remove infected plants, disinfect tools and hands, avoid tobacco products near the crop, use certified seed.',
    irrigation: {
      method: 'Unchanged',
      timing: 'Unchanged',
      depthFactor: 1.05,
      intervalNote: 'Maintain steady moisture; avoid handling the crop while wet, as tools and hands spread this virus mechanically.',
      why: 'ToMV spreads by contact, not by water. The irrigation change here is about handling, not hydrology.',
    },
  },
  {
    match: ['healthy'],
    name: 'Healthy',
    agent: '-',
    group: 'healthy',
    severity: 'none',
    signs: 'No lesions, chlorosis or distortion detected.',
    action: 'Continue routine scouting - twice weekly, looking at the underside of lower leaves.',
    irrigation: {
      method: 'Grower preference',
      timing: 'Early morning is still best practice',
      depthFactor: 1.0,
      intervalNote: 'Follow the model recommendation as computed.',
      why: 'No disease pressure detected, so no modifier is applied.',
    },
  },
];

const UNKNOWN = {
  name: 'Unrecognised',
  agent: '-',
  group: 'unknown',
  severity: 'unknown',
  signs: '-',
  action: 'Confirm with a local extension officer before treating.',
  irrigation: {
    method: 'Drip preferred (precautionary)',
    timing: 'Early morning',
    depthFactor: 1.0,
    intervalNote: 'Unchanged pending confirmation.',
    why: 'Default conservative guidance when the class is not in the knowledge base.',
  },
};

/** Look up the knowledge-base entry for a predicted class label. */
export function pathogenFor(label) {
  const s = String(label).toLowerCase().replace(/[_|]+/g, ' ');
  for (const p of PATHOGENS) {
    if (p.match.some(m => s.includes(m))) return p;
  }
  return UNKNOWN;
}

/* ------------------------------------------------------------------ *
 * Fusion                                                              *
 * ------------------------------------------------------------------ */

/**
 * Combine the irrigation decision with the disease diagnosis.
 * @param {{irrigate:boolean, probability:number, depth_mm:number, derived:object}} irr
 * @param {{label:string, prob:number}|null} disease
 * @param {number} confidenceFloor  below this probability the diagnosis is treated as uncertain
 */
export function fuse(irr, disease, confidenceFloor = 0.45) {
  const confident = !!disease && disease.prob >= confidenceFloor;
  const p = confident ? pathogenFor(disease.label) : null;

  const base = irr.depth_mm;
  const factor = p ? p.irrigation.depthFactor : 1.0;
  const depth = irr.irrigate ? Math.max(0, base * factor) : 0;

  const notes = [];
  if (!confident && disease) {
    notes.push(
      `Diagnosis confidence is only ${(disease.prob * 100).toFixed(0)}%. ` +
      'Irrigation guidance is left unmodified - re-photograph the leaf against a plain ' +
      'background, filling the frame, in even light.'
    );
  }
  if (p && p.group !== 'healthy' && p.group !== 'unknown') {
    notes.push(`${p.name} detected: ${p.irrigation.why}`);
    if (Math.abs(factor - 1) > 0.001) {
      notes.push(
        `Applied depth adjusted by ${factor > 1 ? '+' : ''}${((factor - 1) * 100).toFixed(0)}% ` +
        `(${base.toFixed(1)} mm -> ${depth.toFixed(1)} mm).`
      );
    }
  }

  return {
    irrigate: irr.irrigate,
    probability: irr.probability,
    baseDepthMm: base,
    depthMm: depth,
    method: p ? p.irrigation.method : 'Grower preference',
    timing: p ? p.irrigation.timing : 'Early morning',
    intervalNote: p ? p.irrigation.intervalNote : 'Follow the model recommendation as computed.',
    pathogen: p,
    confident,
    notes,
  };
}
