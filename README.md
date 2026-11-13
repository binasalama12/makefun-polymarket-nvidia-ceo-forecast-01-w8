# polymarket-nvidia-ceo-forecast-01 entry — Bright Falcon (w8)

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of
that minute's price to the output directory.

## Model

Wide memory, gentle distrust of the last print:

- **Long memory:** the last 2160 minutes (36 hours) feed the
  empirical change table (anchors 1, 5, 15, 30, 60, 120, 240,
  interpolated in sqrt(h)) and the sigma estimate (mean absolute
  one-minute change, floor 0.0015, cap 0.005).
- **Slightly shrunk centre:** 0.95 × last price + 0.05 × the
  60-minute mean — enough to soften a one-minute spike, not enough
  to fight a real level shift.
- **Spread = a 65/35 blend** leaning empirical: with 36 hours of
  history the empirical table is well populated at every anchor.
- **Volatility regime:** last 6 hours vs the whole history,
  clamped to [0.6, 1.8].
- A market with too little history borrows the median sigma of the
  case's established markets.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. Deterministic; well under a second per case.

Backtested on the kit's practice inputs (later cases as ground
truth), this scores a mean pinball loss of about 0.375 against the
house baseline's 0.399 on the same folds.

## License

MIT — see `LICENSE`.
