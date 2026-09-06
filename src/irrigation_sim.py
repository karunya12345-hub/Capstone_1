"""
AgriSense - M5: FAO-56 grounded irrigation dataset generator.

Rather than train on an unvetted scraped CSV, we generate a dataset from the
FAO-56 root-zone soil water balance (Allen et al., 1998), driven by a stochastic
weather generator. The reference agronomic policy supplies the labels; the ML
model must recover that policy from NOISY sensor observations, which is the
realistic field task (cf. Goldstein et al., 2018).

  ET0   : Hargreaves-Samani (1985) from Tmax, Tmin and extraterrestrial radiation
  ETc   : Kc * ET0                                            (FAO-56 Eq. 56)
  Dr,i  : Dr,i-1 - (P - RO) - I + ETc + DP                     (FAO-56 Eq. 85)
  TAW   : 1000 * (theta_FC - theta_WP) * Zr                    (FAO-56 Eq. 82)
  RAW   : p * TAW    -> irrigate when Dr >= RAW                (FAO-56 Eq. 83)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Soil hydraulic properties (FAO-56 Table 19, representative values) -----
SOILS = {
    "sandy":  dict(theta_fc=0.16, theta_wp=0.07, p=0.60, infil=60.0),
    "loamy":  dict(theta_fc=0.28, theta_wp=0.14, p=0.55, infil=35.0),
    "clayey": dict(theta_fc=0.38, theta_wp=0.23, p=0.45, infil=12.0),
}
SOIL_IDS = {name: i for i, name in enumerate(SOILS)}

# Fraction of Readily Available Water at which irrigation is triggered.
TRIGGER_FRACTION = 0.50

# Std. dev. of the soil-moisture sensor error, in percentage points of VWC.
# 0.7 pp corresponds to a calibrated FDR/TDR probe (e.g. Campbell CS655, +/-1%)
# whose readings are averaged over a 24 h window. Raise to 1.5-3.0 to simulate an
# uncalibrated capacitive probe - see the sensitivity study in notebook 02.
SENSOR_NOISE_PCT = 0.7

# Probability that the operator deviates from the reference policy.
LABEL_NOISE = 0.015

# --- Crop growth stages: (Kc, root depth Zr in m) ---------------------------
# FAO-56 Table 12 style values for a tomato/potato/pepper-like row crop.
STAGES = {
    "initial":     dict(kc=0.60, zr=0.25),
    "development": dict(kc=0.90, zr=0.45),
    "mid":         dict(kc=1.15, zr=0.80),
    "late":        dict(kc=0.85, zr=1.00),
    "harvest":     dict(kc=0.70, zr=1.00),
}
STAGE_IDS = {name: i for i, name in enumerate(STAGES)}

# Raw sensor / weather observables available at inference time.
SENSOR_COLUMNS = [
    "soil_vwc_pct", "soil_temp_c", "air_temp_c", "humidity_pct",
    "rain_24h_mm", "rain_forecast_mm", "et0_mm", "kc", "days_since_irrigation",
    "soil_sandy", "soil_loamy", "soil_clayey",
]

# Physics-informed features derived from the observables using FAO-56. These are
# computable in the browser from exactly the same inputs, so they add agronomic
# structure without adding any information the deployed system would not have.
PHYSICS_COLUMNS = [
    "root_depth_m", "etc_mm", "est_taw_mm", "est_raw_mm", "est_depletion_mm",
    "depletion_ratio", "projected_deficit_mm", "projected_deficit_ratio",
    "forecast_adj_ratio",
]

FEATURE_COLUMNS = SENSOR_COLUMNS + PHYSICS_COLUMNS

# kc -> root depth lookup, used at inference time (the grower selects the stage).
KC_TO_ZR = {0.60: 0.25, 0.90: 0.45, 1.15: 0.80, 0.85: 1.00, 0.70: 1.00}


def add_physics_features(df):
    """Append the FAO-56 derived columns. Mirrored exactly in web/advisory.js."""
    import numpy as _np
    theta_fc = (df["soil_sandy"] * SOILS["sandy"]["theta_fc"]
                + df["soil_loamy"] * SOILS["loamy"]["theta_fc"]
                + df["soil_clayey"] * SOILS["clayey"]["theta_fc"])
    theta_wp = (df["soil_sandy"] * SOILS["sandy"]["theta_wp"]
                + df["soil_loamy"] * SOILS["loamy"]["theta_wp"]
                + df["soil_clayey"] * SOILS["clayey"]["theta_wp"])
    p_frac = (df["soil_sandy"] * SOILS["sandy"]["p"]
              + df["soil_loamy"] * SOILS["loamy"]["p"]
              + df["soil_clayey"] * SOILS["clayey"]["p"])

    # Late and harvest stages share zr = 1.0 m; disambiguate by kc value.
    zr = df["kc"].map(lambda k: KC_TO_ZR.get(round(float(k), 2), 0.80)).astype(float)

    theta_obs = df["soil_vwc_pct"] / 100.0
    taw = 1000.0 * (theta_fc - theta_wp) * zr
    raw = p_frac * taw
    depletion = _np.maximum(1000.0 * (theta_fc - theta_obs) * zr, 0.0)
    etc = df["kc"] * df["et0_mm"]

    df = df.copy()
    df["root_depth_m"] = zr
    df["etc_mm"] = etc
    df["est_taw_mm"] = taw
    df["est_raw_mm"] = raw
    df["est_depletion_mm"] = depletion
    df["depletion_ratio"] = depletion / _np.maximum(raw, 1e-6)

    # Projected end-of-day deficit: the quantity the FAO-56 trigger is actually
    # compared against. A tree ensemble cannot form this linear combination of
    # depletion, ETc and rainfall from axis-aligned splits, so we compute it
    # explicitly. This is the single most important feature in the model.
    projected = depletion + etc - df["rain_24h_mm"]
    df["projected_deficit_mm"] = projected
    df["projected_deficit_ratio"] = projected / _np.maximum(raw, 1e-6)

    # Same quantity after crediting the (imperfect) rainfall forecast, capped at
    # the trigger depth so a huge forecast cannot drive the term negative.
    trigger = TRIGGER_FRACTION * raw
    credit = _np.minimum(df["rain_forecast_mm"], trigger)
    df["forecast_adj_ratio"] = (projected - credit) / _np.maximum(raw, 1e-6)
    return df


def extraterrestrial_radiation(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """Ra in MJ m-2 day-1 (FAO-56 Eq. 21)."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(x)
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def et0_hargreaves(tmax, tmin, ra):
    """Reference ET in mm/day (Hargreaves & Samani, 1985)."""
    tmean = (tmax + tmin) / 2.0
    return np.maximum(0.0023 * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0))
                      * ra * 0.408, 0.0)


def weather_series(n_days: int, lat: float, rng: np.random.Generator):
    """Seasonal temperature sinusoid + two-state Markov rainfall occurrence."""
    doy = (np.arange(n_days) % 365) + 1
    seasonal = 28.0 + 6.0 * np.sin(2 * np.pi * (doy - 105) / 365.0)
    tmean = seasonal + rng.normal(0, 2.0, n_days)

    wet = np.zeros(n_days, dtype=bool)
    p_wet_given_dry, p_wet_given_wet = 0.12, 0.42
    state = False
    for i in range(n_days):
        p = p_wet_given_wet if state else p_wet_given_dry
        monsoon = 1.0 + 1.6 * np.exp(-((doy[i] - 200) ** 2) / (2 * 45.0 ** 2))
        state = rng.random() < min(p * monsoon, 0.85)
        wet[i] = state
    rain = np.where(wet, rng.gamma(shape=1.3, scale=9.0, size=n_days), 0.0)

    drange = np.where(wet, rng.uniform(4, 8, n_days), rng.uniform(9, 15, n_days))
    tmax = tmean + drange / 2.0
    tmin = tmean - drange / 2.0
    rh = np.clip(52 + 22 * wet + rng.normal(0, 7, n_days) - 0.6 * (tmean - 28), 15, 98)

    ra = extraterrestrial_radiation(lat, doy)
    et0 = et0_hargreaves(tmax, tmin, ra)
    return dict(doy=doy, tmean=tmean, tmax=tmax, tmin=tmin, rain=rain, rh=rh, et0=et0)


def simulate(n_days=2000, soil="loamy", lat=10.94, seed=0, irrigation_efficiency=0.85):
    """Run the FAO-56 water balance for one soil and return a labelled frame."""
    rng = np.random.default_rng(seed)
    sp = SOILS[soil]
    w = weather_series(n_days, lat, rng)

    stage_names = list(STAGES)
    stage_len = [20, 30, 45, 30, 15]           # a 140-day crop cycle
    cycle = np.concatenate([[i] * L for i, L in enumerate(stage_len)])

    rows = []
    dr = 0.0                                    # root-zone depletion, mm
    days_since = 0
    for i in range(n_days):
        st = stage_names[cycle[i % len(cycle)]]
        kc, zr = STAGES[st]["kc"], STAGES[st]["zr"]
        taw = 1000.0 * (sp["theta_fc"] - sp["theta_wp"]) * zr
        raw = sp["p"] * taw

        etc = kc * w["et0"][i]
        rain = w["rain"][i]
        runoff = max(0.0, rain - sp["infil"])
        infiltrated = rain - runoff

        forecast = w["rain"][i + 1] if i + 1 < n_days else 0.0
        # imperfect forecast: multiplicative error + occasional total miss
        forecast_obs = max(0.0, forecast * rng.normal(1.0, 0.35))
        if rng.random() < 0.03:
            forecast_obs = 0.0

        # ---- reference agronomic policy (the label) -------------------------
        # "Light and frequent" drip strategy: trigger at half of RAW and refill
        # only the current depletion, rather than waiting for full RAW and
        # refilling to field capacity. This is the FAO-56 threshold applied with
        # a safety factor, and it is what micro-irrigation practice actually does.
        trigger = TRIGGER_FRACTION * raw
        dr_projected = dr + etc - infiltrated
        effective_forecast = min(forecast_obs, trigger)
        irrigate = 1 if (dr_projected >= trigger and
                         dr_projected - effective_forecast >= 0.5 * trigger) else 0
        depth = 0.0
        if irrigate:
            depth = dr_projected / irrigation_efficiency
            depth = float(np.clip(depth, 3.0, 40.0))

        # ---- observable state BEFORE today's irrigation ---------------------
        theta = sp["theta_wp"] + max(taw - dr, 0.0) / (1000.0 * zr)
        vwc_true = 100.0 * theta
        vwc_obs = float(np.clip(vwc_true + rng.normal(0, SENSOR_NOISE_PCT), 1.0, 60.0))
        soil_temp = w["tmean"][i] - 2.0 + 0.10 * (30.0 - vwc_true) + rng.normal(0, 1.0)

        rows.append(dict(
            day=i, soil=soil, stage=st,
            soil_vwc_pct=vwc_obs,
            soil_temp_c=float(soil_temp),
            air_temp_c=float(w["tmean"][i]),
            humidity_pct=float(w["rh"][i]),
            rain_24h_mm=float(infiltrated),
            rain_forecast_mm=float(forecast_obs),
            et0_mm=float(w["et0"][i]),
            kc=float(kc),
            days_since_irrigation=int(min(days_since, 21)),
            soil_sandy=int(soil == "sandy"),
            soil_loamy=int(soil == "loamy"),
            soil_clayey=int(soil == "clayey"),
            taw_mm=float(taw), raw_mm=float(raw), depletion_mm=float(dr),
            policy_irrigate=int(irrigate), irrigate=int(irrigate),
            depth_mm=float(depth),
        ))

        # ---- advance the balance -------------------------------------------
        dr = dr + etc - infiltrated - (depth * irrigation_efficiency if irrigate else 0.0)
        deep_perc = max(0.0, -dr)
        dr = float(np.clip(dr, 0.0, taw * 1.2))
        days_since = 0 if irrigate else days_since + 1

    df = pd.DataFrame(rows)
    # 2% label noise: real operators do not follow the model perfectly
    flip = rng.random(len(df)) < LABEL_NOISE
    df.loc[flip, "irrigate"] = 1 - df.loc[flip, "irrigate"]
    return df


def build_dataset(days_per_soil=2000, seed=42, lat=10.94) -> pd.DataFrame:
    frames = [simulate(days_per_soil, soil=s, lat=lat, seed=seed + k)
              for k, s in enumerate(SOILS)]
    df = pd.concat(frames, ignore_index=True)
    df = add_physics_features(df)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    d = build_dataset()
    print(d.shape)
    print(d["irrigate"].value_counts(normalize=True).round(3).to_dict())
    print(d.groupby("soil")["depth_mm"].mean().round(2).to_dict())
