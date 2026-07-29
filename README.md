# AffCon: Steering Time Series Foundation Models Beyond Numerical Context with Affiliated Tokens

Reference implementation accompanying the paper *"AffCon: Steering Time Series
Foundation Models Beyond Numerical Context with Affiliated Tokens"*.

AffCon turns asynchronous, timestamped text records into **affiliated tokens**
that a **frozen** decoder-only time series foundation model (TSFM) reads through
its own self-attention. Three ingredients:

1. **Anchoring** — each text record is bound to the first time series token
   whose reference time is at or after the record's timestamp.
2. **Temporal-causal masking** — a numerical state may attend to an affiliated
   token only from its anchor time onward.
3. **AffFlow** — a flow-based generator that transports the anchor token under a
   text-conditioned velocity field to produce the affiliated token.

The backbone is never modified: all trainable capacity lives in AffFlow and, in
Phase 2, in rank-8 LoRA adapters.

This repository contains the AffCon method and the training/evaluation pipeline
for the benchmarks used in the paper. It does not include third-party baseline
implementations, datasets, cached embeddings, or trained checkpoints.

---

## Repository layout

```
run_exp.py                    Single entry point for every experiment
config.py                     Dataset presets, backbone IDs, AffCon hyperparameters

models/
  coms_text.py                Anchoring, token layout, and the 4D temporal-causal mask
  coms_affflow.py             AffFlow: the velocity field and Euler integration
  coms_text_residual.py       Anchor-conditioned residual-MLP control (single-shot)
  coms_lora.py                LoRA injection into the frozen backbone
  coms_rope.py                Shared RoPE utilities (position IDs for affiliated tokens)
  model_sundial.py            Sundial wrapper + backbone/text-encoder loading
  model_timemoe.py            TimeMoE wrapper
  model_timesfm.py            TimesFM-2.5 wrapper
  coms_sundial.py             Inlined Sundial backbone (position IDs + 4D mask threaded through)
  coms_timemoe.py             Inlined TimeMoE backbone (same)
  coms_timesfm.py             Inlined TimesFM transformer (same)

experiments/
  experiment_base.py          Shared training / evaluation loop, metrics, checkpointing
  experiment_timeimm.py       Time-IMM runner (primary benchmark)
  experiment_synthetic.py     Phase 1 synthetic pretraining runner
  experiment_timemmd.py       Time-MMD runner (appendix)
  experiment_cgtsf.py         CGTSF runner (appendix)
  data_timeimm.py             Time-IMM windowing, resampling, text association, embedding cache
  data_synthetic.py           Synthetic corpus loading + lazy text-embedding cache
  data_timemmd.py             Time-MMD loading
  data_cgtsf.py               CGTSF loading
  data_utils.py, utils.py     Normalization, masking, text-encoder helpers
  crps.py                     CRPS / PIT for the probabilistic evaluation
  training_env.py             Training profiles (optimizer, schedule, freezing policy)
```

### Where each part of the paper lives

| Paper | Code |
| --- | --- |
| Anchoring rule, Eq. (3) | `find_anchor_indices` in [models/coms_text.py](models/coms_text.py) |
| Token layout, Eq. (4) | `build_affiliated_token_inputs` in [models/coms_text.py](models/coms_text.py) |
| Temporal-causal mask, Eq. (5) | `build_temporal_causal_4d_mask` in [models/coms_text.py](models/coms_text.py) (`text_attention_mode="temporal_causal"`) |
| AffFlow velocity field + Euler integration, Eq. (6)–(8) and Alg. 1 | `AffFlow` in [models/coms_affflow.py](models/coms_affflow.py) |
| Randomized integration grid | `affflow_train_random_grid`, `affflow_train_steps_min/max` in [config.py](config.py) |
| Residual-MLP control, Eq. (9) | [models/coms_text_residual.py](models/coms_text_residual.py) |
| Phase 1 pretraining | [experiments/experiment_synthetic.py](experiments/experiment_synthetic.py), profile `p1_affflow` |
| Phase 2 adaptation (AffFlow + LoRA) | [experiments/experiment_base.py](experiments/experiment_base.py), profile `p2_lora_affflow` |
| Synthetic-to-real mixing schedule `r_0` | `--synthetic-mix-ratio` |
| Training hyperparameters (App. A.4) | [experiments/training_env.py](experiments/training_env.py) |

Naming note: `affflow` on the command line and `AffFlow` in the source both
refer to the paper's AffFlow generator. Code identifiers follow the manuscript's
vocabulary throughout — anchoring, affiliated tokens, the temporal-causal mask,
and AffFlow all carry the names used in the paper.

---

## Setup

Python 3.11 or 3.12, a CUDA GPU, and PyTorch 2.x.

```bash
pip install -r requirements.txt
```

TimesFM-2.5 is installed from source (pinned in `requirements.txt`); it is only
needed for `--tsfm-model timesfm`.

All pretrained weights are pulled from the Hugging Face Hub on first use:

| Role | Model |
| --- | --- |
| Backbone | `thuml/sundial-base-128m` |
| Backbone | `Maple728/TimeMoE-50M` |
| Backbone | `google/timesfm-2.5-200m-pytorch` |
| Text encoder (frozen) | `Qwen/Qwen3-Embedding-4B` |

The first run downloads these weights, about 8 GB in total. The text encoder
then runs once per corpus; embeddings are cached to disk and reused.

---

## Data

Place the benchmarks under `data/` (see [data/README.md](data/README.md)):

```
data/Time-IMM/     Time-IMM benchmark (primary)
data/Time-MMD/     Time-MMD (appendix)
data/CGTSF/        CGTSF (appendix)
```

None of these are redistributed here; download them from their original
releases. Windowing, resampling to the dataset-specific grid, text association,
and the chronological 60/20/20 split are all handled by
`experiments/data_timeimm.py` at load time and cached.

Phase 1 additionally reads a corpus of paired (series, description) records from
`data/synthetic/` as JSONL, one record per line. Construction is described in
the paper appendix; the loaders in
[experiments/data_synthetic.py](experiments/data_synthetic.py) expect:

- `synthetic_pairs.jsonl` — full-series pairs, with fields `id`, `timeseries`,
  `text`, and `norm_stats: {mean, std}`;
- `synthetic_pairs_history_forecast_split.jsonl` — forecast-style pairs, with
  fields `id`, `history`, `forecast`, `text`, and
  `joint_norm_stats: {mean, std}`.

Text embeddings for these records are computed once with the frozen encoder and
cached next to the corpus.

---

## Running experiments

Everything goes through `run_exp.py`.

**Single-phase run — train AffFlow directly on one target dataset.** The
backbone and text encoder stay frozen, with no synthetic pretraining and no
LoRA, so all trainable capacity sits in AffFlow.

```bash
python run_exp.py \
    --run-mode train --run-stage sp --train-profile sp_affflow \
    --active-dataset time_imm --tsfm-model timemoe \
    --affflow-num-blocks 1 --batch-size 16 \
    --run-tag sp_affflow
```

**Phase 1 — pretrain AffFlow on synthetic pairs.** The backbone and text encoder
stay frozen; the flow starts from a dummy token because no numerical anchor
exists in this setting. Produces the checkpoint used by both settings below.

```bash
python run_exp.py \
    --run-mode train --run-stage p1 --train-profile p1_affflow \
    --active-dataset synthetic --tsfm-model timemoe \
    --run-tag p1_affflow
```

**Phase 2 — adapt AffFlow together with LoRA on Time-IMM.** This is the primary
text-conditioned forecasting setting. AffFlow is frozen for the first epoch to
warm-start LoRA, then both are optimized jointly.

```bash
python run_exp.py \
    --run-mode train --run-stage p2 --train-profile p2_lora_affflow \
    --active-dataset time_imm --tsfm-model timemoe \
    --use-lora --batch-size 32 \
    --synthetic-mix-ratio 0.5 \
    --pretrain-checkpoint /path/to/phase1_timemoe.pt \
    --run-tag p2_affflow
```

**Zero-shot transfer.** Evaluate the Phase 1 checkpoint directly, without
training on the target datasets.

```bash
python run_exp.py \
    --run-mode zeroshot --run-stage p1 \
    --active-dataset time_imm --tsfm-model timemoe \
    --pretrain-checkpoint /path/to/phase1_timemoe.pt \
    --run-tag zs_affflow
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--tsfm-model {sundial,timemoe,timesfm}` | Frozen backbone |
| `--train-profile` | Optimizer, schedule, and freezing policy (`p1_affflow`, `sp_affflow`, `p2_lora_affflow`, …) |
| `--text-attention-mode {temporal_causal,global_prefix}` | Temporal-causal mask vs. global visibility |
| `--text-position-mode {token_aligned,raw_timestamp,zero}` | Position index assigned to affiliated tokens |
| `--affiliated-token-control {none,random}` | Replaces generated tokens with Gaussian vectors |
| `--affflow-num-blocks`, `--affflow-mlp-ratio` | AffFlow velocity-field capacity |
| `--affflow-nfe` | Inference NFE (paper uses 50) |
| `--affflow-train-grid {random,fixed}` | Randomized vs. fixed training integration grid |
| `--generator-type {affflow,residual_mlp}` | AffFlow vs. the single-shot residual-MLP control |
| `--synthetic-mix-ratio` | Initial Phase 2 synthetic-to-real ratio `r_0` |

`python run_exp.py --help` lists the full set, including the flags behind the
component ablation and the AffFlow design studies.

Each run writes a self-contained bundle to
`results/run_traces/<run_mode>_<backbone>_<dataset>_<data_mode>_<stage>_<tag>_<timestamp>/`
holding the resolved config manifest, logs, per-epoch metrics, forecasts, and
the best checkpoint. Metrics are also appended to `results/metrics/`.

