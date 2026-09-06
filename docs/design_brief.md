# Claude Design brief — AgriSense web app

Paste the block below into Claude Design. Keep this file as the record of what
was asked for.

---

Design a modern, eye-catching web app called **AgriSense** — a plant disease
diagnosis and smart irrigation advisor for growers. Six artboards, desktop first,
plus two mobile variants.

**Product in one line:** upload a photo of a crop leaf, get a disease diagnosis
and a matching irrigation recommendation, all computed in the browser.

**Audience:** agronomy students, extension officers and smallholder growers.
Someone standing in a field on a phone, and an examiner at a desk. It should look
credible and modern, not like a student project — but it is a scientific
instrument, not a consumer app. Confident, calm, precise. No stock-photo farmers,
no cartoon leaves.

**Visual direction:** agricultural without cliché. A deep botanical green as the
primary, a warm amber for warnings, a clear red for high-severity disease. Generous
white space, a strong typographic hierarchy, soft cards over hard borders.
Data-dense areas must stay readable — this app shows probabilities, millimetres and
97 numbered features. Must work in both **light and dark** themes.

## Artboards

**1 · Login.** Email + password, sign-up toggle, and — important — a clearly
visible secondary action: **"Continue without signing in"**. The app's whole claim
is that it works with no server, so guest access is a feature, not a fallback.
A short line of positioning copy. No social login.

**2 · Diagnose — empty state.** Large drag-and-drop zone, "Choose file" and "Use
camera" buttons, and a hint that sample images are available. Two small square
canvas placeholders labelled "Pre-processed 160×160" and "Leaf segmentation".

**3 · Diagnose — result state.** The hero is the predicted class with a confidence
percentage. Below it, the top-3 classes as labelled probability bars. Beside it,
the two 160×160 canvases now filled — the second shows the leaf isolated on a dark
background, which is genuinely striking and should be given room. A collapsible
"What drove this prediction" list of feature contributions, and a collapsible table
of all 97 extracted features. A muted line reading "Features 118 ms · inference
0.4 ms · all on this device".
Include a **low-confidence warning banner** variant — amber, calm, non-alarming.

**4 · Irrigation.** Ten inputs on the left: two dropdowns (soil type, growth stage)
and eight sliders with live value readouts (soil moisture %, soil temp, air temp,
humidity, rain last 24 h, rain forecast, reference ET₀, days since last
irrigation). Three preset chips: "Dry spell", "After rain", "Borderline".
On the right, the verdict — "Irrigate today · 38.5 mm" or "Hold — no irrigation" —
above a horizontal depletion gauge with a marked trigger threshold, then a
definition list of FAO-56 values (root-zone depletion, readily available water,
total available water, crop water use, projected deficit).

**5 · Field advisory.** The payoff screen, where diagnosis changes irrigation.
One decisive card: recommended depth, then **Method** ("Drip only — stop all
overhead irrigation"), **Timing** ("Early morning, so the canopy dries within the
day"), **Interval**, the causal agent in italics (*Phytophthora infestans*), what
to look for, and the crop action. Design it so a long explanatory sentence sits
comfortably — this is the most valuable screen in the app.

**6 · History.** A simple reverse-chronological list of past scans: date,
predicted class, confidence, irrigation decision, delete control. Include an empty
state.

**7 & 8 · Mobile.** Login and Diagnose-result at phone width. The camera button
matters most here.

## Constraints

- Vanilla HTML and CSS. No framework, no build step, no icon library — inline SVG
  or nothing.
- Persistent header with the AgriSense mark, the signed-in email, a sign-out
  control, and small status chips ("Disease model · 15 classes", "97 features ·
  offline").
- Navigation is four tabs: Diagnose, Irrigation, Advisory, History.
- Every number shown must have room to be wrong: this model is 86.7% accurate, so
  the design must make confidence and alternatives visible rather than presenting
  a single answer as certain.
