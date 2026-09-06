# Literature Survey

## AI-Powered Plant Disease Detection and Smart Irrigation
### Prior art, comparative analysis and identification of the research gap

| Field | Value |
|---|---|
| Document | Literature Survey (Deliverable 2) |
| Project | AgriSense — AGRISENSE-2026 |
| Version | 1.0 |
| Date | 30 August 2026 |
| Papers reviewed in depth | 18 |

---

## 1. Purpose and Scope of the Survey

This survey establishes what is already known about two research streams — automated plant disease recognition from leaf imagery, and data-driven irrigation scheduling — and locates the gap that this project addresses. Three questions guide it:

- **RQ1.** How far can *classical* (hand-crafted feature + conventional classifier) pipelines go on leaf disease classification, and what exactly do they lose relative to deep learning?
- **RQ2.** What is the state of practice in machine-learning-based irrigation decision support, and what data does that literature actually train on?
- **RQ3.** Has anyone coupled the two — that is, allowed a detected disease state to alter the irrigation recommendation?

### 1.1 Search methodology

| Parameter | Setting |
|---|---|
| Sources | IEEE Xplore, ScienceDirect (Elsevier), SpringerLink, MDPI, Frontiers, Nature Portfolio, ACM DL, arXiv |
| Keyword clusters | (`plant disease` OR `leaf disease`) × (`SVM` OR `random forest` OR `GLCM` OR `texture features` OR `CNN` OR `deep learning`); (`irrigation scheduling` OR `smart irrigation`) × (`machine learning` OR `soil moisture` OR `IoT` OR `evapotranspiration`) |
| Period | 2010 – 2026, with priority to 2016 – 2026 |
| Inclusion | Peer-reviewed; reports a quantitative result on a stated dataset; method reproducible from the text |
| Exclusion | No quantitative evaluation; proprietary dataset with no description; predatory venues |

---

## 2. Taxonomy of Approaches

```
Automated crop health / water management
│
├── A. Disease recognition from imagery
│   ├── A1. Spectral / hyperspectral + classical ML        (Rumpf 2010)
│   ├── A2. RGB leaf image + segmentation + hand-crafted features + classical ML
│   │        colour moments · HSV histograms · GLCM · LBP · wavelets
│   │        classifiers: SVM · RF · LGBM · ANN · k-NN
│   ├── A3. End-to-end deep CNNs                            (Mohanty 2016 → present)
│   └── A4. Dataset & generalisation critiques              (Barbedo 2018; PlantDoc 2020)
│
└── B. Irrigation decision support
    ├── B1. Agronomic reference models (soil water balance)  (FAO-56)
    ├── B2. Threshold / rule-based sensor controllers
    ├── B3. Supervised ML on sensor + weather data           (Goldstein 2018; Glória 2021)
    └── B4. IoT + deep learning irrigation controllers       (Kashyap 2021)

    ✗ C. Coupled disease-aware irrigation  ← essentially unoccupied
```

---

## 3. Stream A2 — Classical Feature Engineering for Leaf Disease Classification

This is the stream the present project extends, so it is reviewed in the most detail.

### 3.1 Rumpf et al. (2010) — the spectral origin

Rumpf and colleagues established the template for the whole classical stream: derive vegetation indices from hyperspectral reflectance of sugar beet leaves, feed them to a Support Vector Machine, and classify *Cercospora* leaf spot, leaf rust and powdery mildew — crucially, **before visible symptoms appear**. Classification accuracy reached up to 97% for discriminating diseased from healthy, and above 86% among the three diseases. The lesson carried forward is that a small number of physically meaningful features, well chosen, is sufficient for an SVM to separate pathologies. The limitation is instrumental: hyperspectral sensing is not available on a farmer's phone.

### 3.2 Singh and Misra (2017) — segmentation-first RGB pipeline

Singh and Misra defined what became the canonical RGB pipeline: colour-space conversion, **genetic-algorithm-optimised image segmentation** to isolate the lesion region, extraction of colour and texture descriptors from the segmented region only, and classification with a minimum-distance criterion and SVM. Reported accuracy is in the mid-90s across banana, beans, jackfruit, lemon, mango, potato, tomato and sapota leaves (the exact figure is widely re-quoted at second hand — confirm it against the published paper before citing a number). Their central methodological claim — that segmenting the diseased region *before* extracting features materially improves accuracy — is adopted directly in this project's M2 module, though with a cheaper deterministic segmenter (Otsu on an excess-green index) rather than a genetic algorithm, to preserve CPU-time and JavaScript portability.

### 3.3 Ahmed, Asif and Saleem (2021) — the colour + GLCM reference point

This is the closest published analogue to the present project's feature design. The authors combine **6 colour features with 22 GLCM-derived texture features** and classify with an SVM. On a laboratory-condition dataset they report **98.79% ± 0.57 accuracy under 10-fold cross-validation**. The important number, however, is what happens off-benchmark: on their own self-collected images the same model achieves only **82.47%** for disease identification and **91.40%** for the simpler healthy-versus-diseased decision. That ~16-point collapse between benchmark and field is the single most instructive result in the classical literature, and it is why the present project treats an out-of-distribution stress test as a mandatory deliverable rather than an optional extra (Project Outline §9).

### 3.4 Tabbakh and Barpanda (2022) — systematic classifier comparison

Tabbakh and Barpanda evaluate six classifiers (LightGBM, SVM, Random Forest, Logistic Regression, AdaBoost, Decision Tree) over a **modified GLCM plus wavelet-based statistical feature** set, computed on both original and segmented images. **LightGBM reaches 94.76%** and SVM 93.51% when all feature types are combined. Two findings are directly actionable here: (i) gradient boosting is competitive with, and can exceed, SVM on this class of feature vector — which is why HistGradientBoosting is in the present project's candidate set; and (ii) combining features from segmented *and* unsegmented views outperforms either alone.

### 3.5 Sridhar and Angamuthu (2025) — current state of the art for classical pipelines

The most recent work surveyed, and evidence that the classical approach is still an active research line rather than a historical curiosity. The pipeline is bilateral filtering → GraphCut segmentation in Y\*Cb\*Cr space → **hybrid GLCM + LBP** texture features → multiclass SVM, with linear, RBF, quadratic and cubic kernels compared. On a 9,111-image four-crop dataset (cotton, wheat, sugarcane, cashew) the **linear kernel achieves 99.0% accuracy** (98.6% precision, 98.7% recall, 98.6% F1) under stratified 5-fold cross-validation. Notably the *linear* kernel wins, implying the hybrid GLCM+LBP representation is already close to linearly separable — a strong argument for the present project's decision to combine both descriptor families, and a practical gift, since a linear model exports to a browser in a few kilobytes.

### 3.6 Descriptor foundations

The texture descriptors used throughout stream A2 trace to two canonical papers. Haralick, Shanmugam and Dinstein (1973) introduced the grey-level co-occurrence matrix and the statistical measures (contrast, correlation, energy, homogeneity, entropy) computed from it. Ojala, Pietikäinen and Mäenpää (2002) introduced rotation-invariant uniform Local Binary Patterns, which capture micro-texture — lesion granularity, mould felt, mite stippling — at a cost of a few operations per pixel. Both are cheap enough to reimplement in JavaScript, which is a hard requirement of this project's deployment target.

### 3.7 Synthesis of stream A2

| Study | Features | Classifier | Dataset | Reported accuracy | Off-benchmark evidence |
|---|---|---|---|---|---|
| Rumpf et al. (2010) | 9 spectral vegetation indices | SVM (RBF) | Sugar beet, hyperspectral | ≤ 97% (binary), > 86% (3-class) | Pre-symptomatic detection demonstrated |
| Singh & Misra (2017) | Colour + texture on GA-segmented lesion | SVM / min-distance | 8 crops, mixed | mid-90s (verify) | Not reported |
| Ahmed et al. (2021) | 6 colour + 22 GLCM | SVM | PlantVillage-style + self-collected | 98.79% ± 0.57 (10-fold) | **82.47%** on self-collected |
| Tabbakh & Barpanda (2022) | Modified GLCM + wavelet statistics | LightGBM (best of 6) | Plant disease imagery | 94.76% | Segmented vs. original compared |
| Sridhar & Angamuthu (2025) | GLCM + LBP hybrid | Multiclass SVM, linear kernel | 9,111 images, 4 crops | 99.0% | 5-fold stratified CV |

**Reading of the table.** Classical pipelines built on colour statistics plus GLCM/LBP texture reliably reach the low-to-high 90s on curated datasets. They do *not* reliably transfer to uncontrolled imagery — and only Ahmed et al. actually measured that. The honest performance envelope for this project is therefore "≥ 90% macro-F1 on a controlled benchmark, with a measured and reported drop in the field", not "99%".

---

## 4. Stream A3/A4 — Deep Learning and Its Documented Limits

### 4.1 The founding results

**Hughes and Salathé (2015)** released the PlantVillage corpus — over 54,000 expert-annotated leaf images spanning 38 crop–disease pairs across 14 species — and in doing so created the benchmark that the entire field standardised on.

**Mohanty, Hughes and Salathé (2016)**, in *Frontiers in Plant Science*, trained AlexNet and GoogLeNet on that corpus and reported **99.35% accuracy** on a held-out split. It is one of the most cited papers in agricultural computing. It also contains its own most important caveat: on images taken under conditions different from the training set, accuracy fell to around **31%**.

**Ferentinos (2018)**, in *Computers and Electronics in Agriculture* (145, 311–318), extended this to an expanded 87,848-image, 58-class database and reported **99.53%** with a VGG-derived architecture — while noting that performance on real field conditions was substantially lower.

**Too et al. (2019)** (*Computers and Electronics in Agriculture*, 161, 272–279) systematically compared fine-tuned VGG16, Inception-V4, ResNet-50/101/152 and DenseNet-121, finding DenseNet best on accuracy-per-parameter. Their contribution to the present project is a *cost* baseline: even the most parameter-efficient of these models is orders of magnitude larger than anything that can be shipped inside a static web page.

### 4.2 The critiques that matter

**Barbedo (2018)** (*Computers and Electronics in Agriculture*, 153, 46–53) tested how dataset size and, more importantly, *variety* affect deep and transfer learning for plant disease classification. The finding: performance is limited far more by the lack of variety in the imagery — uniform backgrounds, single leaves, controlled lighting, one image per condition — than by the number of images. Adding more of the same kind of image does not produce a model that works in a field.

**Singh, Jain, Jain, Kayal, Kumawat and Batra (2020)**, in their CoDS-COMAD paper introducing **PlantDoc**, made the point empirically. PlantDoc contains ~2,598 images of 13 species and 17 disease classes photographed *in situ*, with real backgrounds, occlusion and variable lighting. Models trained on PlantVillage degrade badly on it; training that includes in-the-wild data reduced classification error substantially. PlantDoc is now the standard instrument for measuring the laboratory-to-field gap.

### 4.3 What this means for the present project

The deep-learning literature has effectively converged on two conclusions, both of which the present project builds on rather than contests:

1. On PlantVillage-style data, the benchmark is **saturated**. Reporting another 99% on it adds nothing.
2. The remaining hard problems are **generalisation** and **deployability** — neither of which is solved by a larger network.

This reframes the classical approach. Choosing hand-crafted features over a CNN costs a few points of accuracy on a saturated benchmark and buys: a model measured in kilobytes rather than megabytes, CPU-only training, offline browser execution with no server, and per-feature interpretability. Under Barbedo's finding — that the benchmark ceiling is a dataset artefact, not a model artefact — that trade is defensible rather than merely economical.

---

## 5. Datasets — Comparative Review

| Dataset | Size | Classes | Conditions | Role in this project |
|---|---|---|---|---|
| PlantVillage (Hughes & Salathé 2015) | 54,309 | 38 (14 species) | Laboratory, uniform background | **Primary training source** (3-crop, 15-class subset) |
| PlantDoc (Singh et al. 2020) | ~2,598 | 17 (13 species) | In the wild, real backgrounds | **OOD stress test only** — never trained on |
| Tomato-Village | Field tomato imagery | Tomato-specific | Field | Optional secondary stress test |
| Self-collected field photographs | Small | Subset of the 15 | Uncontrolled, phone camera | Qualitative demo + failure analysis |

**Justification for the 3-crop subset (Tomato, Potato, Bell Pepper — 15 classes).** Restricting scope is deliberate, not a shortcut. Tomato contributes ten classes including four pathologies with visually similar early symptoms (Early blight, Late blight, Septoria leaf spot, Target spot), which is where the interesting confusion lives. Potato shares the *Alternaria solani* and *Phytophthora infestans* pathogens with tomato, so cross-crop confusion becomes measurable. Bell pepper adds a third leaf morphology. The result is ~6,000 images after capping — small enough to extract 97 features from on a CPU in minutes, large enough to support a 15-class problem with a real confusion structure.

---

## 6. Stream B — Machine Learning for Irrigation Decision Support

### 6.1 The agronomic reference model

Any credible irrigation work is measured against **FAO Irrigation and Drainage Paper 56** (Allen, Pereira, Raes and Smith, 1998), which defines reference evapotranspiration ET₀, the crop coefficient approach ETc = Kc · ET₀, and the root-zone water-balance equation used to decide when depletion has reached the readily-available-water threshold. Where full meteorological data is unavailable, the **Hargreaves–Samani (1985)** temperature-based estimator of ET₀ is the standard substitute. This project uses exactly this pair to generate labels, so that the learned model is anchored to an internationally accepted agronomic standard rather than to an arbitrary rule.

### 6.2 Adeyemi et al. (2017) — the framing review

Published in *Sustainability* (9(3), 353), this review surveys monitoring and management systems for precision irrigation and argues for a shift from static scheduling toward dynamic, model-based and data-driven control that closes the loop with real-time soil and weather sensing. It is the standard citation for *why* ML-based irrigation is worth doing at all.

### 6.3 Goldstein et al. (2018) — learning the agronomist's tacit knowledge

Published in *Precision Agriculture* (19(3), 421–444), this is the methodological anchor for the present project's irrigation module. Over nearly two years, data from 22 soil sensors across four jojoba plots was combined with meteorological records and the irrigation instructions actually issued by a human agronomist. Regression and classification algorithms were then trained to reproduce those recommendations. **Gradient Boosted Regression Trees reached ~93%** on the regression formulation and a **Boosted Tree Classifier ~95%** on the classification formulation.

The framing is what matters: irrigation is treated as *learning a decision policy* from sensor observations, not as forecasting soil moisture. The present project adopts the same formulation, with the FAO-56 water balance standing in for the agronomist as the source of ground-truth decisions — a substitution that is more reproducible, and one the authors themselves suggest when noting the models' potential to encode expert reasoning.

### 6.4 Glória, Cardoso and Sebastião (2021) — end-to-end system with measured savings

Published in *Sensors* (21(9), 3079). A LoRa-based wireless sensor network with sensor, actuator and aggregation nodes feeds both a formula-based evapotranspiration algorithm and ML classifiers. On **105,217 records collected over six years**, the comparison is: **Random Forest 84.6%**, Neural Network 80.2%, SVM 79.3%, Decision Tree 77.0%. The combined formula + Random Forest system achieved **up to 60% water savings** over manual irrigation in a 90-day field trial (formula alone: ~50%; prior non-ML work: ~40%).

Two things are decisive for the present project. First, **Random Forest beat the neural network** on real tabular sensor data — which validates using a classical tree ensemble rather than deep learning for the irrigation half. Second, the accuracies here are in the **77–85%** band, not the 95%+ band, because real agronomist decisions are noisy and partly unobservable. Any project reporting 99% irrigation-decision accuracy on real data should be treated with suspicion; this project reports against a *known* reference policy and says so explicitly.

### 6.5 Kashyap et al. (2021) — the deep-learning counterpoint

*IEEE Sensors Journal* (21(16), 17479–17491) presents an IoT-enabled intelligent irrigation system using a deep recurrent architecture over soil-moisture and weather sequences, reporting improved irrigation prediction over conventional baselines. It is included to represent the deep-learning position fairly. Its relevance here is as a boundary marker: it requires a sensor time-series of meaningful length and server-class inference, whereas the present project targets a single-timestep decision computable in a browser.

### 6.6 Synthesis of stream B

| Study | Data | Method | Result | Take-away for this project |
|---|---|---|---|---|
| Allen et al. (1998), FAO-56 | — | Analytical water balance | International reference standard | Used as the label-generating policy |
| Adeyemi et al. (2017) | Review | — | Case for dynamic, data-driven irrigation | Motivation |
| Goldstein et al. (2018) | 22 sensors, ~2 yr, real agronomist logs | GBRT / Boosted Tree Classifier | ~93% regression, ~95% classification | Formulation: learn the *decision policy* |
| Glória et al. (2021) | 105,217 records, 6 yr, LoRa WSN | RF / NN / SVM / DT | RF 84.6%; up to 60% water saved | RF > NN on tabular sensor data |
| Kashyap et al. (2021) | IoT sensor sequences | Deep neural network | Improved prediction | Needs sequences + a server — out of scope here |

---

## 7. Stream C — The Gap: Disease-Aware Irrigation

Searching the intersection of the two streams returns very little. Papers that address both do so by *co-locating* them — an IoT platform that happens to host a disease classifier and an irrigation controller as independent services — not by *coupling* them. No surveyed work lets the output of the disease model change the input, timing or method of the irrigation decision.

This is a substantive omission, because the agronomy is well established:

- Foliar fungal and oomycete pathogens — *Phytophthora infestans* (late blight), *Alternaria solani* (early blight), *Septoria lycopersici*, *Fulvia fulva* (leaf mould) — require extended **leaf wetness duration** and high canopy humidity to germinate and sporulate. Overhead irrigation, and especially evening irrigation that leaves foliage wet overnight, directly increases that risk.
- Bacterial pathogens such as *Xanthomonas* spp. (bacterial spot) are **splash-dispersed**; overhead irrigation actively spreads them between plants.
- Conversely, **spider mite** infestation is favoured by hot, dry, water-stressed conditions, so the correct response is the opposite: maintain rather than reduce moisture.
- And a plant already under water stress has reduced capacity to mount a defence response, so blanket irrigation reduction under disease pressure is also wrong.

The correct advisory is therefore neither "irrigate less" nor "irrigate more" but **pathogen-conditional**: keep the soil-water decision as computed, and modify the *method and timing* (switch from overhead to drip; move irrigation to early morning so foliage dries within the day; extend the interval slightly for splash-dispersed bacteria). No system in the surveyed literature does this.

**This is the gap the present project occupies.** Not a new classifier and not a new irrigation model, but an explicit, auditable **fusion decision table** in which the detected disease class conditions the irrigation recommendation — implemented, deployed and demonstrable end-to-end offline in a browser.

---

## 8. Identified Research Gaps

| # | Gap | Evidence | How this project responds |
|---|---|---|---|
| G1 | The PlantVillage benchmark is saturated; incremental accuracy on it is not a contribution | Mohanty 99.35%; Ferentinos 99.53%; Sridhar 99.0% | Do not compete on benchmark accuracy. Compete on deployability, interpretability and honest OOD reporting |
| G2 | Laboratory-to-field generalisation is rarely measured, and collapses when it is | Mohanty ~31% on unseen conditions; Ahmed 98.79% → 82.47% | OOD stress test is a mandatory, pre-registered deliverable (Outline §9) |
| G3 | State-of-the-art models are undeployable where they are most needed — offline, low-end devices, no server | Too et al.'s model-size comparison | Entire run-time system is a static web page: ~100 kB of JSON plus vanilla JS, no backend |
| G4 | Hand-crafted feature pipelines are rarely specified precisely enough to be reimplemented exactly | Feature counts given, exact definitions usually not | A formal 97-feature contract (Outline §7) implemented twice — Python and JavaScript — with an automated parity test |
| G5 | Disease detection and irrigation control are treated as independent problems despite well-documented agronomic coupling | No coupled system found in streams A or B | An explicit pathogen-conditional fusion decision table linking the two models |

---

## 9. Positioning of the Proposed Work

| Dimension | Prevailing literature | This project |
|---|---|---|
| Model family | Deep CNN (VGG, ResNet, DenseNet, Inception) | Hand-crafted features + SVM / RF / gradient boosting |
| Training hardware | GPU | CPU laptop |
| Deployed artefact | 10–500 MB weights + inference server | ~100 kB JSON in a static page |
| Connectivity at inference | Required | **None** |
| Interpretability | Post-hoc saliency | 97 named, physically meaningful features + permutation importance |
| Headline accuracy | 99%+ on PlantVillage | Target ≥ 90% macro-F1 — with the field gap measured and published |
| Irrigation coupling | Absent or co-located | Explicit pathogen-conditional fusion table |
| Irrigation ground truth | Agronomist logs (Goldstein) or unvetted CSVs | FAO-56 reference water balance, fully documented and regenerable |

The claim of this work is deliberately narrow and defensible: **not that classical machine learning is better than deep learning for plant disease recognition, but that at a fraction of the size, cost and infrastructure it reaches an accuracy that is useful for field triage — and that within that budget there is room left over to do something the larger models have not done, namely couple the diagnosis to the irrigation decision.**

---

## 10. References

[1] T. Rumpf, A.-K. Mahlein, U. Steiner, E.-C. Oerke, H.-W. Dehne, and L. Plümer, "Early detection and classification of plant diseases with Support Vector Machines based on hyperspectral reflectance," *Computers and Electronics in Agriculture*, vol. 74, no. 1, pp. 91–99, 2010. doi:10.1016/j.compag.2010.06.009

[2] V. Singh and A. K. Misra, "Detection of plant leaf diseases using image segmentation and soft computing techniques," *Information Processing in Agriculture*, vol. 4, no. 1, pp. 41–49, 2017. doi:10.1016/j.inpa.2016.10.005

[3] N. Ahmed, H. M. S. Asif, and G. Saleem, "Leaf image-based plant disease identification using color and texture features," *Wireless Personal Communications*, 2021. Preprint: arXiv:2102.04515

[4] A. Tabbakh and S. S. Barpanda, "Evaluation of machine learning models for plant disease classification using modified GLCM and wavelet based statistical features," *Traitement du Signal*, vol. 39, no. 6, pp. 1893–1905, 2022. doi:10.18280/ts.390602

[5] P. Sridhar and P. Angamuthu, "Enhancing image based classification for crop disease detection using a multiclass SVM approach with kernel comparison," *Scientific Reports*, vol. 15, art. 40055, 2025. doi:10.1038/s41598-025-23568-w

[6] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC-3, no. 6, pp. 610–621, 1973.

[7] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 24, no. 7, pp. 971–987, 2002.

[8] D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," arXiv:1511.08060, 2015.

[9] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, art. 1419, 2016. doi:10.3389/fpls.2016.01419

[10] K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318, 2018. doi:10.1016/j.compag.2018.01.009

[11] E. C. Too, L. Yujian, S. Njuki, and L. Yingchun, "A comparative study of fine-tuning deep learning models for plant disease identification," *Computers and Electronics in Agriculture*, vol. 161, pp. 272–279, 2019. doi:10.1016/j.compag.2018.03.032

[12] J. G. A. Barbedo, "Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification," *Computers and Electronics in Agriculture*, vol. 153, pp. 46–53, 2018. doi:10.1016/j.compag.2018.08.013

[13] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A dataset for visual plant disease detection," in *Proc. 7th ACM IKDD CoDS and 25th COMAD*, 2020, pp. 249–253. doi:10.1145/3371158.3371196

[14] R. G. Allen, L. S. Pereira, D. Raes, and M. Smith, *Crop Evapotranspiration — Guidelines for Computing Crop Water Requirements*, FAO Irrigation and Drainage Paper 56. Rome: FAO, 1998.

[15] G. H. Hargreaves and Z. A. Samani, "Reference crop evapotranspiration from temperature," *Applied Engineering in Agriculture*, vol. 1, no. 2, pp. 96–99, 1985.

[16] O. Adeyemi, I. Grove, S. Peets, and T. Norton, "Advanced monitoring and management systems for improving sustainability in precision irrigation," *Sustainability*, vol. 9, no. 3, art. 353, 2017. doi:10.3390/su9030353

[17] A. Goldstein, L. Fink, A. Meitin, S. Bohadana, O. Lutenberg, and G. Ravid, "Applying machine learning on sensor data for irrigation recommendations: revealing the agronomist's tacit knowledge," *Precision Agriculture*, vol. 19, no. 3, pp. 421–444, 2018. doi:10.1007/s11119-017-9527-4

[18] A. Glória, J. Cardoso, and P. Sebastião, "Sustainable irrigation system for farming supported by machine learning and real-time sensor data," *Sensors*, vol. 21, no. 9, art. 3079, 2021. doi:10.3390/s21093079

[19] P. K. Kashyap et al., "Towards precision agriculture: IoT-enabled intelligent irrigation systems using deep learning neural network," *IEEE Sensors Journal*, vol. 21, no. 16, pp. 17479–17491, 2021. doi:10.1109/JSEN.2021.3069266

[20] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

### Note on citation verification

All references were located and their bibliographic details checked against publisher or indexing records during preparation of this survey.

**Numeric results confirmed directly from the publisher record:** [3] (98.79% ± 0.57 CV, 82.47% self-collected, 91.40% binary), [4] (LightGBM 94.76%, SVM 93.51%), [5] (99.0% linear kernel, 9 111 images), [17] (GBRT ~93%, boosted-tree classifier ~95%, 22 sensors), [18] (RF 84.6% / NN 80.2% / SVM 79.3% / DT 77.0%, 105 217 records, up to 60% water saving).

**Figures quoted at second hand and to be verified before final submission:** the accuracy in [2]; the 97% / >86% split in [1]; the ~31% out-of-condition figure attributed to [9]; the full author list for [19], abbreviated here with *et al.*

No reported accuracy has been independently reproduced. Where a number matters to an argument in this project, the argument is written so that it survives a small correction to the number.
