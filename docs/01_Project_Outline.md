# Project Outline

## AgriSense — AI-Powered Plant Disease Detection and Smart Irrigation Advisory
### A Classical Machine Learning Approach for Low-Resource Deployment

| Field | Value |
|---|---|
| Project code | AGRISENSE-2026 |
| Domain | Machine Learning · Computer Vision · Precision Agriculture |
| Category | Software-only (no hardware dependency) |
| Compute budget | CPU-only, commodity laptop |
| Deployment | Fully client-side static web application |
| Document version | 1.0 |
| Date | 30 August 2026 |

---

## 1. Problem Statement

Two decisions dominate day-to-day smallholder crop management, and both are currently made by eye:

1. **Is this plant diseased, and with what?** Misdiagnosis leads to the wrong agro-chemical being applied, or to no action until the infection is systemic. The FAO attributes 20–40% of global crop production loss annually to pests and pathogens.
2. **Should I irrigate today, and how much?** Irrigation on a fixed calendar schedule ignores rainfall, evaporative demand and the actual water already held in the root zone. Over-irrigation wastes water and energy; it also raises canopy humidity and leaf wetness duration, which is precisely the condition that most fungal foliar pathogens require to sporulate.

These two problems are almost always solved separately in the literature. They are not independent: **the disease state of a crop should change the irrigation recommendation**, and irrigation practice changes disease pressure. This project treats them as one coupled advisory problem.

### 1.1 The engineering constraint that shapes this project

Nearly all recent published work solves the disease-detection half with deep convolutional networks (see Literature Survey, §4). Those models need a GPU to train, tens to hundreds of megabytes to store, and a server to run inference. That excludes exactly the deployment target that matters — an offline, low-end device in the field.

This project therefore deliberately adopts a **classical machine-learning pipeline**: explicit, hand-designed colour/texture/shape features followed by a conventional classifier (SVM / Random Forest / Gradient Boosting). The consequences are not merely a compromise; they are the contribution:

- Trains in minutes on a CPU, with no GPU and no cloud cost.
- The final model is small enough (tens to a few hundred kB of JSON) to be **shipped inside a web page** and executed in the browser with zero backend.
- Every feature is interpretable, so a wrong prediction can be explained ("lesion-brown pixel fraction was high, GLCM contrast was high") rather than being a black box.
- The whole system runs **offline**, which is the realistic connectivity assumption in a field.

---

## 2. Objectives

| # | Objective | Measurable success criterion |
|---|---|---|
| O1 | Build a reproducible dataset pipeline over a 3-crop subset of PlantVillage (Tomato, Potato, Bell Pepper — 15 classes) | Deterministic, seeded train/val/test split; class manifest committed to the repo |
| O2 | Design an interpretable feature extractor (colour, lesion-chromaticity, GLCM texture, LBP) computable identically in Python and JavaScript | ≤ 120 features; Python↔JS feature parity error < 1e-4 on a fixed test image set |
| O3 | Train and compare ≥ 5 classical classifiers for disease classification | Best model ≥ 90% macro-F1 on a held-out test split; full comparison table reported |
| O4 | Build a physically-grounded irrigation decision model (FAO-56 soil water balance) and learn it with a classical regressor/classifier | Irrigation-decision accuracy ≥ 95% vs. the agronomic reference policy; water-depth MAE ≤ 2 mm |
| O5 | Fuse the two models into a single advisory that modifies irrigation guidance based on detected disease | Documented, rule-explicit fusion layer with a decision table |
| O6 | Deploy as a fully client-side web application with live prediction | Page loads and predicts with no network calls after first load; works from `file://` and from GitHub Pages |
| O7 | Produce a rigorous evaluation, including honest failure analysis | Confusion matrix, per-class report, cross-validation with std-dev, and a documented limitations section |

---

## 3. Scope

### 3.1 In scope
- Leaf-level disease classification from a single RGB image, 15 classes across 3 crops.
- Irrigation decision (irrigate / do not irrigate) and depth (mm) from 9 agro-meteorological inputs.
- A rule-based fusion layer coupling the two.
- A static, offline-capable web front-end for live demonstration.
- Full evaluation, documentation and a reproducible training notebook.

### 3.2 Out of scope (and why)
- **Deep learning / CNNs** — excluded by design (see §1.1). A CNN baseline is *cited* from literature for comparison but not trained.
- **Physical IoT hardware** — the irrigation model consumes sensor *values*; whether they arrive from an ESP32, a manual reading or the demo sliders is an interface detail. A hardware integration spec is given in §14 as future work.
- **Disease severity quantification / lesion segmentation for area %** — classification only.
- **Field-condition images (PlantDoc-style)** — used only as a *stress test* to report the generalisation gap honestly, not as a training target.
- **Multi-leaf / whole-canopy imagery.**

### 3.3 Assumptions
- Input image contains a single, roughly centred leaf occupying > 25% of the frame.
- Images are RGB, ≥ 128×128 px.
- Sensor inputs to the irrigation model are pre-calibrated (volumetric water content in %, not raw ADC counts).

---

## 4. System Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           OFFLINE  (training)            │
                          └──────────────────────────────────────────┘

  PlantVillage subset          src/features.py            scikit-learn
  (Tomato/Potato/Pepper)  ──►  ┌───────────────┐   ──►   ┌──────────────────┐
  15 classes, ~400 img/cls     │ Segment leaf  │         │ LinearSVC / RBF  │
                               │ Colour stats  │         │ SVM / RandomFor. │  ──┐
                               │ HSV histogram │         │ HistGradBoost    │    │
                               │ Lesion ratios │         │ (grid-searched)  │    │
                               │ GLCM (4 dir)  │         └──────────────────┘    │
                               │ Uniform LBP   │                                 │
                               └───────────────┘                                 │
                                                                                 ▼
  src/irrigation_sim.py                                              src/export_json.py
  FAO-56 water balance   ──►   9 tabular features   ──►  RF classifier + RF     │
  + stochastic weather         (VWC, ET0, rain, …)      regressor              │
                                                                                 │
                          ┌──────────────────────────────────────────┐          │
                          │      ONLINE  (browser, no server)        │  ◄───────┘
                          └──────────────────────────────────────────┘          models/*.json

     web/features.js  ── mirrors features.py exactly (parity-tested)
             │
             ▼
   ┌───────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
   │ <canvas> pixel    │   │ irrigation sliders │   │  FUSION LAYER        │
   │ extraction        │   │ / sensor JSON      │   │  disease × water     │
   └─────────┬─────────┘   └─────────┬──────────┘   │  → final advisory    │
             │                       │              └──────────┬───────────┘
             ▼                       ▼                         ▼
     disease_model.json      irrigation_model.json      Rendered advisory card
     (JS inference)          (JS inference)             + top-3 probabilities
                                                        + treatment guidance
```

Everything to the right of the dashed boundary is a `.json` file and plain JavaScript. There is no Python at run time, no API, and no server.

---

## 5. Module Breakdown

| ID | Module | File(s) | Responsibility |
|---|---|---|---|
| M1 | Dataset acquisition & manifest | `src/dataset.py` | Download/locate PlantVillage subset, cap per-class counts, build seeded stratified splits, write `data/processed/manifest.csv` |
| M2 | Feature extraction (Python) | `src/features.py` | Leaf segmentation + 4 feature blocks → fixed-length vector with named columns |
| M3 | Feature extraction (JavaScript) | `web/features.js` | Bit-for-bit mirror of M2 for browser inference |
| M4 | Disease model training | `notebooks/01_disease_model.ipynb` | Feature caching, model comparison, hyper-parameter search, evaluation, export |
| M5 | Irrigation data synthesis | `src/irrigation_sim.py` | FAO-56 root-zone water balance driven by stochastic weather → labelled dataset |
| M6 | Irrigation model training | `notebooks/02_irrigation_model.ipynb` | RF classifier (irrigate?) + RF regressor (mm), evaluation, export |
| M7 | Model export | `src/export_json.py` | Serialise scaler + PCA + linear model *or* tree ensemble into compact JSON |
| M8 | Inference engine (JS) | `web/inference.js` | Generic JSON-model evaluator: linear and tree-ensemble paths |
| M9 | Fusion & advisory logic | `web/advisory.js` | Decision table coupling disease class → irrigation modifier + treatment text |
| M10 | Web UI | `web/index.html`, `web/app.js`, `web/styles.css` | Upload/camera, live prediction, irrigation panel, explainability panel |
| M11 | Parity test | `src/parity_test.py` + `web/parity.html` | Assert Python and JS produce the same feature vector |

---

## 6. Dataset Plan

### 6.1 Disease images — PlantVillage (3-crop subset)

| Crop | Classes | Notes |
|---|---|---|
| Tomato | 10 (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy) | The hardest and most agronomically relevant crop in the set |
| Potato | 3 (Early blight, Late blight, Healthy) | Shares *Alternaria*/*Phytophthora* pathogens with tomato — tests cross-crop confusion |
| Bell Pepper | 2 (Bacterial spot, Healthy) | Adds a third leaf morphology |
| **Total** | **15 classes** | ≈ 20,600 images available; capped at 400/class ⇒ ≈ 6,000 images used |

- **Source:** `https://www.kaggle.com/datasets/emmarex/plantdisease` (the standard "PlantVillage" 3-crop segmented subset), or the canonical repository `https://github.com/spMohanty/PlantVillage-Dataset`, or `tfds` catalogue entry `plant_village`.
- **Licence:** CC0 1.0 Public Domain (original PlantVillage release).
- **Capping rationale:** 400 images/class keeps the feature-extraction pass under ~10 minutes on a CPU and mitigates the 5:1 class imbalance in the raw set.
- **Known limitation, stated up front:** PlantVillage images are captured on uniform laboratory backgrounds. Models trained on it are known to over-fit background and lighting cues (Barbedo 2018; Singh et al. 2020). This is addressed by (a) background removal in M2 so the classifier never sees the background, and (b) a stress-test on out-of-distribution field images reported in the evaluation.

### 6.2 Irrigation data — synthesised, physically grounded

There is no widely-adopted public benchmark that pairs soil-moisture sensor traces with expert irrigation decisions for a specific crop and soil. Rather than adopt an unvetted scraped CSV, this project **generates** its dataset from the FAO Irrigation & Drainage Paper 56 root-zone water-balance model, which is the international reference method for irrigation scheduling:

```
Dr,i = Dr,i-1 − (P − RO)i − Ii − CRi + ETc,i + DPi      (FAO-56, Eq. 85)
ETc  = Kc · ET0            ET0 estimated by Hargreaves-Samani from Tmax, Tmin, Ra
Irrigate when   Dr ≥ RAW = p · TAW,   TAW = 1000 (θFC − θWP) Zr
```

A stochastic weather generator (seasonal temperature sinusoid + Markov-chain rainfall occurrence + gamma rainfall depth) drives the balance over multiple simulated seasons and soil types. The reference agronomic policy provides the labels; the ML model must **learn that policy from noisy sensor observations** — which is the realistic task, since a field deployment sees noisy sensors, not a perfect water balance.

| Property | Value |
|---|---|
| Rows | 30,000 (3 soil types × 5 crops-stages × 2,000 days) |
| Features | soil VWC %, soil temp °C, air temp °C, RH %, rain last 24 h, forecast rain next 24 h, ET0 mm, crop-stage Kc, days since last irrigation, soil type (one-hot) |
| Labels | `irrigate` ∈ {0,1}; `depth_mm` ∈ ℝ⁺ |
| Realism injected | Gaussian sensor noise on VWC (σ = 1.5%), 2% label flip, 3% missing forecast |

Two public datasets are cross-referenced as an optional external validation: the Kaggle *Smart Agriculture / auto-irrigation* sensor logs and the *IoT Agriculture 2024* set. These are documented in the notebook as an appendix, not as the primary source.

---

## 7. Feature Specification (M2/M3 contract)

This is the interface both the Python trainer and the JavaScript inference engine must satisfy. Any change here must be made in both files and re-verified by M11.

**Pre-processing:** resize to 160×160 (bilinear, aspect-preserving with centre crop) → RGB float [0,1].

**Leaf mask:** Excess-Green minus Excess-Red index, `ExGR = 2G − R − B − (1.4R − G)`, thresholded by Otsu, then largest-connected-component + hole fill. Falls back to the whole frame if the mask covers < 5% of pixels.

| Block | Features | Count |
|---|---|---|
| B1 · Colour moments | mean, std, skewness of R, G, B, H, S, V over leaf pixels | 18 |
| B2 · HSV histogram | H (16 bins) + S (8) + V (8), L1-normalised over leaf pixels | 32 |
| B3 · Lesion chromaticity | pixel fraction in each of 6 pathology-relevant HSV boxes (healthy-green, chlorotic-yellow, necrotic-brown, dark-lesion, grey-mould, purple/bronze), + mask coverage, + green/(green+brown) ratio | 8 |
| B4 · GLCM texture | grey quantised to 16 levels; 4 offsets ((1,0),(0,1),(1,1),(1,−1)); contrast, dissimilarity, homogeneity, energy, correlation, entropy per offset | 24 |
| B5 · Uniform LBP | P=8, R=1, 10-bin uniform-pattern histogram over leaf pixels | 10 |
| B6 · Shape | mask area fraction, perimeter², circularity, extent, eccentricity of the leaf blob | 5 |
| | **Total** | **97** |

---

## 8. Model Selection Plan

Candidates, all CPU-cheap:

1. Logistic Regression (multinomial, L2) — linear baseline, tiniest export
2. Linear SVM (`LinearSVC`, one-vs-rest)
3. RBF-kernel SVM (`SVC`, probability-calibrated)
4. Random Forest (300 trees, depth-limited)
5. Extra Trees (300 trees)
6. Histogram Gradient Boosting
7. k-NN (k=5, distance-weighted) — sanity baseline

Protocol: `StandardScaler` → optional `PCA(0.98 variance)` → classifier. 5-fold stratified CV on the training split for model selection and `RandomizedSearchCV` for the top two; final numbers reported **once** on the untouched test split.

**Export constraint:** whichever model wins, it must serialise to JSON that `web/inference.js` can evaluate. Linear models export as a coefficient matrix; tree ensembles export as flat arrays per tree. Kernel SVM exports as support vectors + duals (acceptable if the support-vector count stays under ~2,000, otherwise the runner-up is deployed and the trade-off is documented).

---

## 9. Evaluation Methodology

| Aspect | Metric / method |
|---|---|
| Disease — primary | Macro-F1 on held-out test split (class-imbalance-robust) |
| Disease — secondary | Accuracy, per-class precision/recall, 15×15 confusion matrix, top-3 accuracy |
| Disease — stability | 5-fold CV mean ± std |
| Disease — ablation | Drop each feature block B1–B6 and re-measure; report Δ macro-F1 (proves each block earns its place) |
| Disease — explainability | Permutation importance over the 97 named features |
| Disease — honesty check | Accuracy on out-of-distribution field photographs; report the drop, do not hide it |
| Disease — literature context | Compare against published CNN numbers on the same subset (cited, not re-trained) |
| Irrigation — decision | Accuracy, precision/recall on the "irrigate" class, confusion matrix |
| Irrigation — depth | MAE, RMSE in mm |
| Irrigation — impact | Simulated seasonal water use vs. a fixed-calendar baseline (target: measurable % saving) |
| System | Browser inference latency (ms), model payload size (kB), cold-load time |

---

## 10. Deployment Architecture

Static files only. `web/` can be opened directly from disk or served from GitHub Pages / Netlify / any static host.

```
web/
├── index.html          UI shell, three panels
├── styles.css
├── features.js         M3 — mirror of features.py
├── inference.js        M8 — generic JSON model evaluator
├── advisory.js         M9 — fusion decision table + treatment knowledge base
├── app.js              wiring, canvas handling, camera capture, rendering
└── models/
    ├── disease_model.json      exported by notebook 01
    ├── irrigation_model.json   exported by notebook 02
    └── labels.json
```

No build step, no npm, no bundler, no framework. This is deliberate: the examiner can open one file and read the whole run-time system.

---

## 11. Technology Stack

| Layer | Choice | Justification |
|---|---|---|
| Language (training) | Python 3.10+ | Ecosystem |
| Numerics | NumPy, Pandas | — |
| Imaging | Pillow, scikit-image (GLCM, LBP, Otsu) | Reference implementations of the exact texture descriptors specified in §7 |
| ML | scikit-learn 1.4+ | All seven candidate models, CPU |
| Notebooks | JupyterLab | Deliverable requirement |
| Plots | Matplotlib | Report figures |
| Front-end | Vanilla HTML/CSS/JS (ES modules) | Zero-dependency, auditable, offline |
| Hosting | GitHub Pages (static) | Free, no server |
| VCS | Git | — |

---

## 12. Timeline (12 weeks)

| Week | Milestone | Exit criterion |
|---|---|---|
| 1 | Literature survey, problem finalisation | This document + Literature Survey signed off |
| 2 | Dataset acquisition, manifest, EDA | `manifest.csv` + class-distribution figure |
| 3 | `features.py` implemented, segmentation validated visually | Feature matrix cached to `.npy` for full subset |
| 4 | Baseline models trained, first confusion matrix | ≥ 80% macro-F1 achieved |
| 5 | Model comparison + hyper-parameter search | Comparison table complete, best model chosen |
| 6 | Ablation + permutation importance | Ablation table complete |
| 7 | `irrigation_sim.py`, FAO-56 balance validated | Water-balance trace figure; dataset generated |
| 8 | Irrigation models trained and evaluated | Decision accuracy + depth MAE recorded |
| 9 | JSON export + `inference.js` + parity test | Parity error < 1e-4 |
| 10 | Web UI, fusion advisory layer | End-to-end live demo working offline |
| 11 | OOD stress test, latency benchmarks, figures | Evaluation chapter drafted |
| 12 | Report, viva demo rehearsal, buffer | Final submission |

---

## 13. Deliverables

1. `docs/01_Project_Outline.md` — this document
2. `docs/02_Literature_Survey.md` — surveyed prior art and research gap
3. `notebooks/01_disease_model.ipynb` — end-to-end disease pipeline
4. `notebooks/02_irrigation_model.ipynb` — end-to-end irrigation pipeline
5. `src/` — reusable, importable modules
6. `web/` — deployable client-side demonstrator
7. `models/` + `web/models/` — exported model artefacts
8. `reports/` — figures, tables, evaluation outputs
9. Final project report and presentation

---

## 14. Risks and Mitigations

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | Classical features plateau well below CNN accuracy | M | M | This is an expected, *reportable* result. Frame the trade-off quantitatively (accuracy vs. model size, energy, offline capability). Ablation shows where the ceiling comes from. |
| R2 | Python↔JS feature mismatch causes silently wrong browser predictions | H | H | M11 parity test is a hard gate; the browser also ships a self-check page comparing against 5 committed reference vectors. |
| R3 | Feature extraction too slow for ~6,000 images on CPU | M | M | 160×160 resize, 16-level GLCM quantisation, vectorised NumPy, cached `.npy` feature matrix so extraction runs once. |
| R4 | Kernel SVM wins but exports too large for the browser | M | L | Fall back to the best tree/linear model; document the accuracy cost. |
| R5 | Irrigation model criticised as "trained on synthetic data" | M | M | Pre-empt it: the generator is the FAO-56 international reference standard, fully documented and cited; add optional validation on public sensor logs. |
| R6 | PlantVillage background bias undermines the result's credibility | H | M | Background is removed before feature extraction; an OOD stress test is reported explicitly rather than omitted. |
| R7 | Scope creep into hardware | M | M | Hardware is fenced into §3.2 / future work from day one. |

---

## 15. Stated Contributions

1. A **97-feature interpretable descriptor** for leaf disease that is specified precisely enough to be implemented identically in two languages, with a parity test proving it.
2. A demonstration that **classical ML deployed entirely client-side** is a viable operating point for plant disease triage, with the accuracy/size/offline trade-off quantified rather than assumed.
3. A **coupled disease–irrigation advisory**: an explicit decision table in which detected foliar pathology modifies irrigation timing and method (e.g. suppressing evening overhead irrigation under late-blight risk), addressing a gap where the two problems are almost universally treated in isolation.
4. A reproducible, physically-grounded **FAO-56 irrigation dataset generator** released with the project.

---

## 16. Future Work

- ESP32 + capacitive soil-moisture / DHT22 hardware node publishing over MQTT into the same model interface.
- Fine-tuning on field-condition imagery (PlantDoc, Tomato-Village) to close the OOD gap.
- Disease severity regression (lesion area %) to drive dosage, not just identification.
- A quantised MobileNet baseline as a direct on-device deep-learning comparison point.
- Multilingual advisory text and a PWA install target for genuine field use.
