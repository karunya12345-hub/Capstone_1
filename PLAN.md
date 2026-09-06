# PLAN.md — AgriSense frontend, auth and deployment

Target: a modern, presentable web app with user login, developed locally, then
deployed to **Cloudflare Pages** (static hosting) with **Supabase** (auth +
database) on free tiers.

Read `CLAUDE.md` first. Work phase by phase; each phase has an acceptance check
that must pass before moving on.

---

## 0 · The architectural decision, and why it matters

The project's stated contribution is that inference runs **entirely in the
browser with no server**. Adding login could quietly destroy that claim, so it
must not.

**The rule: Supabase authenticates the user; it never sees an image and never
makes a prediction.**

```
  Browser (Cloudflare Pages — static files only)
  ├── features.js  ──►  97 features from the uploaded image   ← never leaves the device
  ├── inference.js ──►  disease_model.json  (22 kB, loaded once)
  ├── advisory.js  ──►  irrigation models + fusion table
  └── supabase-js  ──►  Supabase:  sign-up / sign-in / session
                                   optional: saved scan history (text only)
```

Nothing here needs a server of your own — Cloudflare Pages serves static files and
the Supabase client talks to Supabase directly from the browser. **No Workers, no
Pages Functions, no API routes.** If you find yourself writing a server endpoint,
stop: the design has gone wrong.

What the honest claim becomes, and it is still strong:

> Diagnosis and irrigation inference require no network. Only sign-in and history
> sync do. Sign in once, then run the whole advisory offline.

Keep a **"Continue without signing in"** path on the login screen. It is the
single most valuable thing in the demo: it proves the claim live.

---

## 1 · Phase 1 — Frontend rebuild (localhost, no auth yet)

Claude Design produces the visual layer. Your job is to wire it to the existing
logic **without touching the maths**.

- Replace `web/index.html` and `web/styles.css` with the new design.
- Keep `features.js`, `inference.js`, `advisory.js` **unchanged**. Only `app.js`
  may be rewritten, and only its DOM-wiring half.
- Preserve every existing capability: drag-and-drop + file picker + camera
  capture, the 160×160 source and segmentation-mask canvases, top-3 probability
  bars, the 97-feature inspector, the "what drove this prediction" panel, the ten
  irrigation inputs with the three presets, the FAO-56 water-balance readout, and
  the fused advisory card.
- Keep it buildless: plain ES modules, no bundler, no framework. Import
  `@supabase/supabase-js` from an ESM CDN in Phase 2.

**Accept when:** `python src/parity_test.py` passes; three images from
`demo_images/` predict exactly what `INDEX.csv` says; the dry/wet/borderline
presets behave; no console errors; light and dark themes both look right.

---

## 2 · Phase 2 — Supabase auth (still localhost)

Free tier, hosted project (there is no local Postgres to run).

1. Create a project at supabase.com. Note the **Project URL** and **anon key**.
2. Auth → Providers → enable **Email**. For a demo, turn *off* "Confirm email"
   so a new account works immediately; note that you did.
3. Add `web/config.js` — committed on purpose, the anon key is public:

```js
export const SUPABASE_URL = "https://<project>.supabase.co";
export const SUPABASE_ANON_KEY = "<anon key>";
```

4. Add `web/auth.js`: `signUp`, `signInWithPassword`, `signOut`,
   `getSession`, `onAuthStateChange`. Nothing else.
5. Gate the app behind a login screen, with the guest path from §0.
6. Show the signed-in email and a sign-out control in the header.

**Accept when:** sign-up, sign-out and sign-in all work; a refresh keeps you
signed in; guest mode reaches the full app; wrong password gives a readable
error, not a stack trace.

---

## 3 · Phase 3 — Scan history (optional, high demo value)

Only if Phase 2 is solid. Stores the *result*, never the image.

```sql
create table public.scans (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null references auth.users(id) on delete cascade,
  created_at     timestamptz not null default now(),
  predicted_class text not null,
  confidence     real not null,
  top3           jsonb,
  crop           text,
  irrigate       boolean,
  depth_mm       real,
  notes          text
);

alter table public.scans enable row level security;

create policy "own rows: select" on public.scans
  for select using (auth.uid() = user_id);
create policy "own rows: insert" on public.scans
  for insert with check (auth.uid() = user_id);
create policy "own rows: delete" on public.scans
  for delete using (auth.uid() = user_id);

create index scans_user_created_idx on public.scans (user_id, created_at desc);
```

RLS is not optional. Without it the anon key — which ships to every visitor —
reads the whole table.

A fourth tab, "History", lists recent scans with a delete control. Guest mode
keeps history in memory only and says so.

**Accept when:** a scan appears in history; signing in as a second account shows
an empty list (proves RLS); delete works; guest mode doesn't error.

---

## 4 · Phase 4 — Deploy

**Cloudflare Pages** — connect the GitHub repo:

| Setting | Value |
|---|---|
| Build command | *(leave empty)* |
| Build output directory | `web` |
| Framework preset | None |

There is no build step, so Pages just serves `web/` as-is.

**Then, in Supabase** — Authentication → URL Configuration:
- Site URL → your `*.pages.dev` URL
- Redirect URLs → add both the Pages URL and `http://localhost:8000`

Forgetting this is the classic first-deploy failure: auth works locally and
breaks in production.

**Accept when:** the Pages URL loads; sign-in works there; a `demo_images/` file
predicts the same class it did locally; DevTools shows no request to anything but
Cloudflare and Supabase.

---

## 5 · Phase 5 — Demo hardening

- A first-run hint pointing at `demo_images/`, since a visitor has no leaf photo.
- Loading state while the 22 kB model fetches; a clear error if it 404s.
- Friendly failure if the image isn't a leaf — the mask coverage in
  `B3_coverage` is a usable signal.
- Keep the low-confidence banner under 45%. **Do not remove it to make the demo
  look better.** Confidence carries little signal on non-PlantVillage images and
  the banner is the honest hedge.
- Mobile: the camera capture input already exists; check it on a phone.
- Lighthouse pass for obvious accessibility misses (contrast, labels, focus).

---

## Free-tier limits (both are fine here)

| | Limit | This project |
|---|---|---|
| Cloudflare Pages | 500 builds/month, unlimited bandwidth | trivial |
| Supabase | 2 projects, 500 MB database, 50k monthly active users | text rows only |

Supabase pauses a free project after ~1 week of no activity. **Open the app the
day before your viva** so it is warm.

---

## Out of scope

- Retraining. The models are done; `notebooks/` and `src/` are frozen for this
  phase apart from the parity gate.
- Server-side inference, Workers, Pages Functions.
- Image upload to storage — results only, never the photo.
- OAuth providers, password reset, email templates. Email + password is enough.
- Any change to the 97-feature contract.

---

## Definition of done

- [ ] Parity test passes
- [ ] Deployed on Cloudflare Pages, sign-in works in production
- [ ] Predictions in production match `demo_images/INDEX.csv`
- [ ] Aeroplane-mode test: signed in, network off, diagnosis still runs
- [ ] Second account sees an empty history (RLS proven)
- [ ] No service-role key anywhere in the repo
