# demo_images — upload-ready samples for the web demo

30 images, **2 per class × 15 classes**, drawn from the **test split** of
`data/processed/manifest.csv` (seed 7). None of these were seen during training,
so they are a fair live demonstration.

The filename **is** the ground truth: `<PlantVillage_class>__<n>.jpg`. Alphabetical
order groups them by class, so the file picker in the browser reads like a menu.

## How to demo

```bash
cd web && python -m http.server 8000     # then open http://localhost:8000
```

Tab 1 → "Choose file" → navigate to `demo_images/` → pick any image. The filename
tells you the right answer; the page tells you what the model said.

## What to expect

`INDEX.csv` records the prediction the deployed `web/models/disease_model.json`
actually produces for each file, verified by running the exported JSON model
through the same 97-feature extractor the browser uses.

**27 / 30 correct (90%), mean confidence 0.853** — in line with the 86.7% test-set
accuracy. All three misses are the known-weak classes from the confusion matrix,
and in **all three the correct class is the runner-up**, so top-2 is 30/30.

| File | Model says | Should be |
|---|---|---|
| `Potato___healthy__1.jpg` | Pepper bell healthy (0.74) | Potato healthy |
| `Tomato_Early_blight__1.jpg` | Tomato Target Spot (0.70) | Tomato Early blight |
| `Tomato_Late_blight__1.jpg` | Tomato Septoria leaf spot (0.80) | Tomato Late blight |

Keep those three in the demo rather than hiding them — being able to explain *why*
a model failed (small class; overlapping early-blight symptomatology) is worth more
in a viva than a clean sweep.

## Good ones to open with

`Tomato_healthy__1.jpg`, `Tomato__Tomato_YellowLeaf__Curl_Virus__1.jpg`,
`Potato___Early_blight__1.jpg` — all predicted at 1.00 confidence with clean
segmentation masks.
