# Data

None of the benchmarks are redistributed in this repository. Download them from
their original releases and place them here:

```
data/
  Time-IMM/     Time-IMM benchmark  (primary evaluation, 7 datasets)
  Time-MMD/     Time-MMD            (appendix)
  CGTSF/        CGTSF               (appendix)
  synthetic/    Paired (series, description) corpora for Phase 1
```

Dataset roots and per-dataset presets (lookback, horizon, stride, time unit,
resampling interval, target channel) are declared in `config.py`; override the
roots there if your copies live elsewhere.

## Time-IMM

The primary benchmark. Each dataset holds one directory per record, and each
record holds the two files the loader reads:

```
data/Time-IMM/
  ILINet/processed/US/time_series.csv
                      text.csv
  EPA-Air/processed/Bexar/...
```

`experiments/data_timeimm.py` performs all preprocessing at load time and caches
the result:

- record-level preprocessing following the benchmark, then per-channel
  standardization and sliding windows;
- numerical observations mapped to the dataset-specific regular grid (nearest
  location, duplicates averaged); empty locations are filled with the
  training-split mean and stay marked as missing, so they contribute to neither
  the loss nor the metrics;
- text records keep their raw timestamps; each record is assigned to the first
  grid location at or after its arrival, and from there to the corresponding
  backbone token;
- chronological 60/20/20 train/validation/test split within each record;
- only windows containing at least one text record in the history are kept, and
  exactly this subset is used for both the text and no-text conditions.

Text embeddings are computed once with the frozen `Qwen/Qwen3-Embedding-4B`
encoder and cached alongside the windows, so the encoder runs only on the first
pass over a dataset.

## Time-MMD and CGTSF

Used for the appendix experiments on sample- and window-aligned text. Both keep
their original splits, lookbacks, and horizons, and use training-split
normalization.

## Synthetic corpora

Phase 1 reads paired (series, description) records from `data/synthetic/` as
JSONL, one record per line. Construction is described in the paper appendix;
`experiments/data_synthetic.py` expects:

- `synthetic_pairs.jsonl` — full-series pairs, with fields `id`, `timeseries`,
  `text`, and `norm_stats: {mean, std}`;
- `synthetic_pairs_history_forecast_split.jsonl` — forecast-style pairs, with
  fields `id`, `history`, `forecast`, `text`, and
  `joint_norm_stats: {mean, std}`.

Series are z-normalized in the file; `norm_stats` / `joint_norm_stats` preserve
the pre-normalization scale. Text embeddings are computed on first use and
cached next to the corpus.
