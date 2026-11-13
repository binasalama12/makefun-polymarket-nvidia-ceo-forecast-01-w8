#!/usr/bin/env python3
"""Bright Falcon forecaster for Polymarket's "Next CEO of NVIDIA?"
event (make.fun polymarket-nvidia-ceo-forecast-01, wallet w8).

Reads one case from the input directory: every market of the event
with its Yes price at each minute up to the checkpoint. Writes, for
each market and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

The Bright Falcon variant pairs the family's widest memory with a
gentle distrust of the last print:

- Volatility and the empirical h-step change distribution are
  estimated over the last 2160 minutes (36 hours).
- The centre is shrunk just 5% from the last price toward the
  60-minute mean (0.95 * last + 0.05 * mean60): enough to soften a
  one-minute spike, not enough to fight a real level shift.
- The blend leans empirical, 65/35: with 36 hours of history the
  empirical table is well populated at every anchor.
- Sigma is the mean-absolute-change estimator with the wide band,
  floor 0.0015 and cap 0.005.
- The regime factor (last 6 hours vs the whole history) gets the
  wide clamp [0.6, 1.8], trusting the long window to have priced
  the normal range already.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded
to 6 decimals. Deterministic, standard library only.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

BASE_SIGMA = 0.0025
WINDOW = 2160            # minutes of history for vol estimates
ANCHORS = [1, 5, 15, 30, 60, 120, 240]
EMP_WEIGHT = 0.65
SIGMA_FLOOR = 0.0015
SIGMA_CAP = 0.005
MIN_EMP_HISTORY = 300
SHRINK = 0.05            # centre = (1-SHRINK)*last + SHRINK*mean60
MEAN_SPAN = 60


def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def price6(v):
    return round(clamp01(v) * 1e6) / 1e6 + 0.0


def quantile_of(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def avg_abs_change(prices):
    if len(prices) < 2:
        return 0.0
    return sum(abs(prices[i + 1] - prices[i])
               for i in range(len(prices) - 1)) / (len(prices) - 1)


def centre_of(prices):
    last = prices[-1]
    span = prices[-MEAN_SPAN:] if len(prices) > MEAN_SPAN else prices
    return (1.0 - SHRINK) * last + SHRINK * (sum(span) / len(span))


def build_anchors(hist):
    table = {}
    n = len(hist)
    for h in ANCHORS:
        if h >= n:
            continue
        changes = sorted(hist[i + h] - hist[i] for i in range(n - h))
        table[h] = [quantile_of(changes, tau) for tau in TAUS]
    return table


def emp_row_at(table, h):
    keys = sorted(table)
    if not keys:
        return None
    if h <= keys[0]:
        return table[keys[0]]
    if h >= keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= h <= hi:
            f = (math.sqrt(h) - math.sqrt(lo)) / (math.sqrt(hi) - math.sqrt(lo))
            return [table[lo][j] * (1.0 - f) + table[hi][j] * f
                    for j in range(len(TAUS))]
    return table[keys[-1]]


def sigma_for(prices):
    window = prices[-WINDOW:] if len(prices) > WINDOW else prices
    return min(max(1.2533 * avg_abs_change(window), SIGMA_FLOOR), SIGMA_CAP)


def regime_factor(prices):
    if len(prices) <= 720:
        return 1.0
    overall = avg_abs_change(prices)
    if overall <= 1e-9:
        return 1.0
    recent = avg_abs_change(prices[-361:])
    return min(1.8, max(0.6, recent / overall))


def predict_market(prices, horizon, sigma):
    c = centre_of(prices)
    window = prices[-WINDOW:] if len(prices) > WINDOW else list(prices)
    table = build_anchors(window) if len(window) >= MIN_EMP_HISTORY else {}
    scale = regime_factor(prices)
    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        gauss = [c + z * sigma * sq for z in Z]
        emp = emp_row_at(table, h)
        if emp is None:
            row = gauss
        else:
            row = [EMP_WEIGHT * (c + emp[j]) + (1.0 - EMP_WEIGHT) * gauss[j]
                   for j in range(len(TAUS))]
        rows.append(sorted(price6(c + (v - c) * scale) for v in row))
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text())
        if list(case["quantiles"]) != TAUS:
            raise ValueError(f"{path.name} asks for quantiles "
                             f"{case['quantiles']}, not {TAUS}")
        horizon = int(case["horizon_minutes"])
        parsed = []
        for market in case["markets"]:
            prices = [float(p) for p in market["prices"]]
            if not prices:
                raise ValueError(f"market {market['id']} has no prices")
            parsed.append((market["id"], prices))
        sigmas = {mid: sigma_for(prices) for mid, prices in parsed
                  if len(prices) >= MIN_EMP_HISTORY}
        if sigmas:
            ordered = sorted(sigmas.values())
            fallback_sigma = ordered[len(ordered) // 2]
        else:
            fallback_sigma = BASE_SIGMA
        forecasts = [{"id": mid,
                      "q": predict_market(prices, horizon,
                                          sigmas.get(mid, fallback_sigma))}
                     for mid, prices in parsed]
        out = json.dumps({"forecasts": forecasts}) + "\n"
        (OUTPUT_DIR / f"{case_id}.out").write_text(out)


if __name__ == "__main__":
    main()
