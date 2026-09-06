# CLAUDE.md — AgriSense

Working notes for Claude Code. Read this before touching anything.

## What this is

A final-year CS project: leaf disease detection + FAO-56 irrigation advisory, both
running as **classical ML models exported to JSON and evaluated in the browser**.
Two Jupyter notebooks train the models; `web/` is a static site that loads the
exported JSON and does inference client-side. There is no backend, and inference
must never need one.

Current state: both models trained and exported, demo site working, everything
committed. The next phase adds a redesigned frontend, Supabase auth, and a
Cloudflare Pages deploy. See `PLAN.md`.

## The one thing that will silently break this project

`src/features.py` and `web/features.js` compute the same 97-dimensional feature
vector. **They must stay behaviourally identical.** If they drift, nothing throws —
the browser just quietly predicts the wrong class, and you will not notice.

```bash
python src/parity_test.py      # requires Node.js on PATH
```

Currently agrees to `2.3e-12`. **Run it after touching either file, and before
declaring any frontend work done.** Notebook 01 runs it automatically and refuses
to export if it fails.

Consequences:

- Never "simplify" `features.py` by swapping hand-written NumPy for a scikit-image
  call. The plain-NumPy implementations exist *precisely* so JS can mirror them.
- Never let a bundler, minifier or transpiler change float semantics in
  `features.js`. Math must stay float64 except where noted below.
- `inference.js` uses `Math.fround()` when comparing a feature to a tree
  threshold. That is not a mistake — scikit-learn casts X to float32 inside its
  tree traversal, and without the matching cast, samples within ~1e-7 of a split
  take the other branch. Do not "clean it up".

## Golden rules

1. **Inference stays client-side.** Supabase is for auth and stored history only.
   Never route a prediction through a server. Pull the plug on the network mid-demo
   and diagnosis must still work.
2. **Model JSONs are build artefacts.** `web/models/*.json` are written by the
   notebooks. Never hand-edit them. To change a model, change the notebook and
   re-run its export cell.
3. **`src/*.py` is the reference implementation.** JS mirrors Python, not the
   other way round.
4. **No new runtime dependencies without a reason.** The site is deliberately
   vanilla — no framework, no build step. That is a documented design decision
   (Project Outline §10), not laziness. The Supabase JS client via ESM CDN is the
   one accepted addition.
5. **Don't touch `data/raw/`.** It is gitignored, ~20k PlantVillage images, and
   re-downloading is slow.

## Layout

```
docs/         Project Outline + Literature Survey (markdown and .docx)
notebooks/    01_disease_model.ipynb, 02_irrigation_model.ipynb — both fully run
src/          features.py  dataset.py  irrigation_sim.py  export_json.py
              parity_test.py  ← the gate
web/          index.html styles.css features.js inference.js advisory.js app.js
              models/*.json  ← generated, do not edit
demo_images/  30 held-out test images for the live demo; INDEX.csv has the
              prediction each one should produce
reports/      figures and tables the notebooks emit
data/raw/     PlantVillage (gitignored)
```

## Commands

```bash
cd web && python -m http.server 8000     # serve the site — NOT file://
python src/parity_test.py                # Python <-> JS feature parity gate
jupyter lab notebooks/                   # retrain (01 needs data/raw/)
```

`file://` will not work: browsers block ES modules and `fetch` on that scheme.

## Numbers worth knowing (don't contradict them in UI copy)

| | |
|---|---|
| Disease model | Logistic Regression, 15 classes, 97 features, **22.2 kB** |
| | test accuracy **86.73%**, macro-F1 0.8666, top-3 97.67% |
| Irrigation classifier | Logistic Regression, decision threshold **0.32**, accuracy 90.5%, F1 0.612 |
| Irrigation depth | Ridge, MAE **1.74 mm**, R² 0.948 |
| Combined model payload | ~25 kB |
| Water saving vs calendar | 47%, zero extra crop-stress days |

The disease model is trained on laboratory-condition imagery. Accuracy on field
photographs is substantially lower. Don't write UI copy that implies otherwise —
the low-confidence warning under 45% exists for this reason.

## Supabase specifics

- The **anon key is public by design**. It ships in the client. What protects data
  is Row Level Security, not key secrecy. Never add a service-role key to
  anything under `web/`.
- Every table gets RLS enabled and a policy scoped to `auth.uid()`. A table
  without RLS is readable by anyone holding the anon key — which is everyone.
- Keep config in `web/config.js` (URL + anon key). It is committed deliberately.

## Before you say you're done

- [ ] `python src/parity_test.py` passes
- [ ] Site loads over `python -m http.server`, no console errors
- [ ] Upload 3 files from `demo_images/` — predictions match `INDEX.csv`
- [ ] Irrigation tab: "dry spell" preset says irrigate, "after rain" says hold
- [ ] Network tab: no requests after initial load except Supabase auth
- [ ] Works in both light and dark browser themes
