# AgriSense

**AI-powered plant disease detection and disease-aware smart irrigation — classical machine learning, CPU-only, running entirely in a browser.**

A final-year CS project. Two models, one advisory. No GPU, no deep learning framework, no server, no network at inference time.

---

## What it does

1. **Leaf disease detection.** A photograph of a leaf goes in; a diagnosis across 15 crop–disease classes (tomato, potato, bell pepper) comes out, with the top-3 probabilities and the features that drove the decision. 97 hand-designed colour, lesion-chromaticity, GLCM texture, LBP and shape features feed a conventional classifier.
2. **Irrigation decision.** Soil moisture, weather and crop stage go in; an irrigate/hold decision and an applied depth in millimetres come out, grounded in the FAO-56 root-zone water balance.
3. **The fusion layer — the part that is new.** The diagnosis does not change *whether* the soil needs water. It changes *how* and *when* the water is delivered: late blight suppresses overhead irrigation and moves it to early morning; bacterial spot forces drip because the pathogen is splash-dispersed; spider mite — favoured by drought — is the one case where the advisory *increases* water. Every surveyed paper treats these two problems separately.

Everything runs client-side. The exported irrigation models are **3 kB combined**; the disease model is compacted to fit a stated deployment budget.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Irrigation half — works immediately, no downloads
jupyter lab notebooks/02_irrigation_model.ipynb

# 2. Disease half — needs the dataset first (see below)
jupyter lab notebooks/01_disease_model.ipynb

# 3. The demo site
cd web && python -m http.server 8000     # then open http://localhost:8000
```

> Serve `web/` over HTTP rather than double-clicking `index.html`: browsers block ES modules and `fetch` on `file://` URLs.

### Getting the dataset

Download the PlantVillage 3-crop subset and extract it into `data/raw/`, so you have folders like `data/raw/PlantVillage/Tomato___Late_blight/`.

* Kaggle — <https://www.kaggle.com/datasets/emmarex/plantdisease>
* GitHub — <https://github.com/spMohanty/PlantVillage-Dataset>
* TFDS — `tfds.load('plant_village')`

For the out-of-distribution stress test in notebook 01 §8, put field photographs into `data/ood/<class name>/` — [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) or your own phone pictures.

---

## Repository layout

```
docs/
  01_Project_Outline.md      scope, architecture, feature contract, timeline, risks
  02_Literature_Survey.md    18 papers reviewed, comparison tables, research gap
notebooks/
  01_disease_model.ipynb     dataset -> features -> model comparison -> ablation
                             -> OOD stress test -> size budget -> export
  02_irrigation_model.ipynb  FAO-56 simulation -> models -> Bayes-limit analysis
                             -> water-saving simulation -> export   [pre-run]
src/
  features.py                97-feature extractor (pure NumPy, mirrored in JS)
  dataset.py                 PlantVillage manifest + seeded stratified split
  irrigation_sim.py          FAO-56 water balance + stochastic weather generator
  export_json.py             scikit-learn -> browser-evaluable JSON
  parity_test.py             asserts Python and JavaScript agree  ← the safety gate
web/
  index.html  styles.css     the demo site
  features.js                exact mirror of src/features.py
  inference.js               generic JSON model evaluator
  advisory.js                FAO-56 mirror + pathogen fusion decision table
  app.js                     UI wiring
  models/*.json              exported models
reports/                     figures and tables produced by the notebooks
```

---

## The one thing that will break this project

`src/features.py` and `web/features.js` must compute **the same 97 numbers**. If they drift, nothing errors — the browser just quietly predicts the wrong class, and you will not notice until the viva.

```bash
python src/parity_test.py            # requires Node.js on PATH
```

Current status: agreement to `2.3e-12`. Run it after touching either file. Notebook 01 runs it automatically before exporting and refuses to deploy if it fails.

---

## Results worth defending

**Irrigation.** The model reaches the *ceiling imposed by the soil-moisture probe*: applying the exact FAO-56 rule to the same noisy observations scores F1 0.608, and the learned model scores 0.612. The residual error is sensor noise, not model capacity — no deeper network can recover it. Notebook 02 §4 proves this and sweeps the sensor-noise level to show both curves degrading together.

For context, Glória et al. (2021) report 84.6% on six years of real field data, with a Random Forest *ahead* of a neural network. Numbers in the mid-80s to low-90s are what this problem honestly looks like.

Simulated over a 180-day season, the learned policy applies **47% less water than a fixed 5-day calendar schedule with zero additional crop-stress days**.

**Disease.** Report the operating point, not the accuracy. Every paper in the literature already reports 99% on PlantVillage; Mohanty et al.'s own model fell to ~31% on images taken under different conditions, and Ahmed et al. (2021) went from 98.79% cross-validated to 82.47% on their own field photographs. Notebook 01 measures that gap (§8) instead of omitting it, quantifies what each feature block contributes (§7), and shows exactly what the browser size budget costs in accuracy (§9).

---

## Design decisions, and why

| Decision | Reason |
|---|---|
| Classical ML, not a CNN | Trains on a CPU in minutes; deploys as kilobytes of JSON; every feature is interpretable; runs offline |
| Segment the leaf before extracting features | Removes PlantVillage's uniform background, the main source of the background-memorisation effect Barbedo (2018) documents |
| Hand-written NumPy instead of scikit-image | The JavaScript mirror has to reproduce it exactly; library internals cannot be relied on to match |
| Synthesised irrigation data | FAO-56 is the international reference standard, fully documented and regenerable — unlike an unvetted scraped CSV |
| Physics-informed features | A tree ensemble cannot form `depletion + ETc − rainfall` from axis-aligned splits. Computing it explicitly is what closes the gap to the ceiling |
| `Math.fround` in the JS tree evaluator | scikit-learn casts to float32 inside tree traversal; without the matching cast, samples near a split take the other branch and the browser silently disagrees |
| Deployment size as a selection constraint | A 15 MB model is not a deployed model. Notebook 01 §9 makes the trade-off explicit |

---

## Limitations

* The disease model is trained on laboratory-condition imagery. Field accuracy is lower — measure it, do not assume it.
* The irrigation model learns a reference policy, not observed grower behaviour. Validating against real sensor logs is the obvious next step.
* Single-leaf classification only: no severity estimate, no whole-canopy assessment.
* This is decision support. Confirm any treatment with a local extension officer before applying agro-chemicals.

---

## Key references

Full list with DOIs in `docs/02_Literature_Survey.md` §10.

Allen et al. (1998) FAO-56 · Mohanty et al. (2016) *Front. Plant Sci.* · Ferentinos (2018) *Comput. Electron. Agric.* · Barbedo (2018) *Comput. Electron. Agric.* · Singh et al. (2020) PlantDoc, CoDS-COMAD · Ahmed et al. (2021) *Wireless Pers. Commun.* · Goldstein et al. (2018) *Precision Agriculture* · Glória et al. (2021) *Sensors* · Sridhar & Angamuthu (2025) *Scientific Reports*
