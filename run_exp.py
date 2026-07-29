
from __future__ import annotations

import argparse
import copy
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config import CONFIG
from experiments.experiment_timemmd import TimeMMDExperiment
from experiments.experiment_timeimm import TimeIMMExperiment
from experiments.experiment_cgtsf import CGTSFExperiment
from experiments.experiment_synthetic import SyntheticExperiment
from experiments.training_env import (
    profile_names,
    resolve_batch_plan,
    validate_and_resolve_train_profile,
)

PROJ_PATH = Path(__file__).resolve().parent


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or zeroshot evaluate time-series forecasting models."
    )
    parser.add_argument(
        "--run-mode",
        choices=["train", "zeroshot"],
        default="train",
        help="Run training or zeroshot evaluation.",
    )
    parser.add_argument(
        "--active-dataset",
        choices=["time_imm", "time_mmd", "cgtsf", "synthetic"],
        default="time_imm",
        help="Select dataset family for the run.",
    )
    parser.add_argument(
        "--synthetic-phase1-mode",
        choices=["full_series", "forecast"],
        default="full_series",
        help="Synthetic-only Phase 1 recipe: full-series generative pretraining "
             "or forecast-style history+text->future pretraining.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional domain filter for Time-MMD / CGTSF runs.",
    )
    parser.add_argument(
        "--timeimm-datasets",
        nargs="+",
        default=None,
        help="Optional dataset filter for Time-IMM runs (e.g. EPA-Air GDELT).",
    )
    parser.add_argument(
        "--timeimm-selected-chunk-ids",
        nargs="+",
        default=None,
        help="Optional explicit Time-IMM chunk IDs to evaluate (e.g. 20079585_chunk0). "
             "When provided, zeroshot evaluation reports metrics on exactly these chunks.",
    )
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Path to append metrics as CSV. If omitted, a run-specific CSV is created under results/metrics/.",
    )
    parser.add_argument(
        "--trace-root",
        default=None,
        help="Optional directory to collect all manifests/logs/forecasts/checkpoints for this run.",
    )
    parser.add_argument(
        "--helper-script",
        dest="helper_scripts",
        action="append",
        default=None,
        help="Optional helper script / jobs file to snapshot into the run trace bundle. May be repeated.",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Optional campaign identifier grouping multiple submitted jobs under one launcher event.",
    )
    parser.add_argument(
        "--campaign-spec-path",
        default=None,
        help="Optional job-matrix / campaign-spec file associated with this run.",
    )
    parser.add_argument(
        "--campaign-label",
        default=None,
        help="Optional short human-facing campaign label.",
    )
    parser.add_argument(
        "--campaign-note",
        default=None,
        help="Optional short human-facing campaign note or purpose.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional append-only campaign registry JSONL path.",
    )
    parser.add_argument(
        "--patch-text-embeds",
        dest="use_patch_text_embeds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use cached patch-level text embeddings. Default is off; patch-level "
             "text is treated as an explicit Sundial ablation rather than the "
             "main default text representation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Effective training batch size.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Per-step device batch size. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps. If omitted, derived from batch/micro batch sizes.",
    )
    parser.add_argument(
        "--auto-batch-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable auditable OOM fallback by reducing micro-batch size and increasing accumulation.",
    )
    parser.add_argument(
        "--initial-micro-batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lookback-multiplier",
        type=int,
        help="CGTSF only: set lookback = horizon * lookback_multiplier for all domains.",
    )
    parser.add_argument(
        "--run-all-lookback-multipliers",
        action="store_true",
        help="CGTSF only: loop over all allowed lookback multipliers sequentially.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Required short run label used to distinguish variants (e.g. debug, withaff, rerun1).",
    )
    parser.add_argument(
        "--run-stage",
        choices=["plain", "p1", "sp", "p2"],
        default=None,
        help="Required model/checkpoint provenance encoded in run ids: plain, p1, sp, or p2.",
    )
    parser.add_argument(
        "--train-profile",
        choices=profile_names(),
        default=None,
        help="Required v5 training profile for --run-mode train.",
    )
    parser.add_argument(
        "--tsfm-model",
        choices=["sundial", "timemoe", "timesfm"],
        default="sundial",
        help="Model choice for all experiment runs.",
    )
    parser.add_argument(
        "--timeimm-irregularity",
        choices=["resampled", "raw"],
        default="resampled",
        help="Time-IMM data mode: 'resampled' (uniform grids) or 'raw' (irregular).",
    )
    parser.add_argument(
        "--patch-position-mode",
        choices=["timestamp", "contiguous"],
        default="contiguous",
        help="Patch backbone position-ID policy for Sundial and TimesFM: "
             "'contiguous' restores native patch indices 0..N-1 and is the "
             "active default, while 'timestamp' keeps patch-end timestamps "
             "as position IDs.",
    )
    parser.add_argument(
        "--text-attention-mode",
        choices=["temporal_causal", "global_prefix"],
        default="temporal_causal",
        help="Prefix-text attention policy: 'temporal_causal' keeps TS->text temporal restriction, "
             "while 'global_prefix' makes text globally visible to all later TS positions.",
    )
    parser.add_argument(
        "--text-position-mode",
        choices=["token_aligned", "raw_timestamp", "zero"],
        default="token_aligned",
        help="Runtime position-ID policy for injected text tokens.",
    )
    parser.add_argument(
        "--text-injection-layout",
        choices=["distributed", "collapsed_global"],
        default="distributed",
        help="Whether text is injected across its aligned positions or collapsed into a single global prefix slot.",
    )
    parser.add_argument(
        "--text-embedding-control",
        choices=["none", "random_fixed"],
        default="none",
        help="Optional ablation on incoming text embeddings before AffFlow/projection.",
    )
    parser.add_argument(
        "--text-embedding-random-seed",
        type=int,
        default=0,
        help="Base seed for deterministic random text-embedding controls.",
    )
    parser.add_argument(
        "--affiliated-token-control",
        choices=["none", "random"],
        default="none",
        help="Optional ablation on generated affiliated tokens. 'random' bypasses AffFlow and inserts Gaussian token vectors.",
    )
    parser.add_argument(
        "--fill-policy",
        choices=["global_mean", "sequence_mean"],
        default="global_mean",
        help="Value written into empty resampled grid cells. 'global_mean' (default) "
             "uses the training-set mean; 'sequence_mean' uses the window's own observed "
             "mean (history only). Empty cells stay masked either way and never enter "
             "the loss or evaluation metrics.",
    )
    parser.add_argument(
        "--observed-missing-policy",
        choices=["mask_aware", "seq_mean_impute"],
        default="mask_aware",
        help="AffCon-side handling of missing positions after resampling. "
             "'mask_aware' keeps missing history slots invalid for the backbone; "
             "'seq_mean_impute' makes missing in-window slots visible as normalized-mean values. "
             "The imputed policy is currently supported only for Time-IMM resampled AffCon runs.",
    )
    parser.add_argument(
        "--force-flip-invariance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TimesFM: mirror native compiled_decode by averaging the odd component "
             "(F(x) - F(-x)) / 2 at inference time. The project default is on; "
             "use --no-force-flip-invariance to disable it explicitly.",
    )
    parser.add_argument(
        "--timesfm-infer-is-positive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TimesFM: mirror native infer_is_positive by clamping the forecast "
             "to >= 0 when the conditioning history is entirely non-negative. "
             "Default off in the wrapper.",
    )
    parser.add_argument(
        "--timesfm-decoupled-flip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TimesFM: use the 3-decode decoupled flip so the text delta is not "
             "cancelled by flip symmetrization. Default off in the wrapper.",
    )
    parser.add_argument(
        "--compute-crps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Distributional metrics (CRPS / PIT) on the final model over the "
             "test split, for Sundial and TimesFM. The project default is on; "
             "use --no-compute-crps to skip them in broad campaigns.",
    )
    parser.add_argument(
        "--crps-num-samples",
        type=int,
        default=None,
        help="Sundial predictive-ensemble size for sample CRPS (TimesFM ignores it). "
             "Defaults to cfg.affcon.crps_num_samples.",
    )
    parser.add_argument(
        "--crps-dump",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write CRPS/PIT per-window .npz dumps beside the run metrics. "
             "Required for calibration forecast-band figures.",
    )
    parser.add_argument(
        "--chunk-anchor-date",
        type=str,
        default=None,
        help="Phase the Time-IMM sliding window (YYYY-MM-DD) so that a chunk's "
             "forecast starts on this date. Used by the calibration figure so "
             "the event's explaining text falls inside the history window.",
    )
    parser.add_argument(
        "--synthetic-mix-ratio",
        type=float,
        default=None,
        help="Epoch-0 synthetic:real batch ratio for downstream real-data training. "
             "The ratio halves each epoch and validation/test stay real-only.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs) on validation MSE. Defaults to the selected --train-profile.",
    )
    parser.add_argument(
        "--early-stop-min-delta-pct",
        type=float,
        default=0.0,
        help="Relative val-MSE improvement (percent) required to count as a new best "
             "for early stopping and checkpoint selection. 0 keeps the plain '<' rule.",
    )
    parser.add_argument(
        "--affflow-text-concat",
        action="store_true",
        default=True,
        help="AffFlow: concatenate text embedding into MLP input (True) or use AdaLN (False).",
    )
    parser.add_argument(
        "--generator-type",
        choices=["affflow", "residual_mlp"],
        default=None,
        help="Affiliated-token generator. 'residual_mlp' selects the "
             "single-shot anchor-conditioned control (TimeMoE only).",
    )
    parser.add_argument(
        "--affflow-num-blocks",
        type=int,
        default=None,
        help="Number of generator residual blocks. Used by both AffFlow and the "
             "residual-MLP parameter-frontier sweep.",
    )
    parser.add_argument(
        "--affflow-mlp-ratio",
        type=int,
        default=None,
        help="Generator residual-block MLP expansion ratio.",
    )
    parser.add_argument(
        "--no-affflow-text-concat",
        dest="affflow_text_concat",
        action="store_false",
        help="AffFlow: use AdaLN for text conditioning instead of concatenation.",
    )
    parser.add_argument(
        "--affflow-text-mode",
        choices=["concatext", "adaln", "filmtext", "affconcat"],
        default="affconcat",
        help="AffFlow text-conditioning mode. "
             "Default 'affconcat' is the affflow-style block (shared-norm concat, "
             "gate-free, time affine after MLP), i.e. the paper's AffFlow generator.",
    )
    parser.add_argument(
        "--affflow-prior-type",
        choices=["standard", "textprior"],
        default="standard",
        help="AffFlow source distribution: standard Gaussian or text-conditioned Gaussian prior.",
    )
    parser.add_argument(
        "--affflow-nfe",
        type=int,
        default=None,
        help="Euler integration steps (NFE) for AffFlow at inference "
             "(the Campaign C4 NFE sweep varies this; default keeps the config value).",
    )
    parser.add_argument(
        "--affflow-train-grid",
        choices=["random", "fixed"],
        default=None,
        help="Training integration grid for AffFlow. 'random' "
             "(default) resamples the step count/points per batch; 'fixed' "
             "integrates on a uniform affflow-nfe grid every batch — the "
             "fixed-grid control for the step-budget study.",
    )
    parser.add_argument(
        "--flow-init",
        choices=["anchor", "noise"],
        default=None,
        help="AffFlow x0: embedded anchor token or token-space Gaussian noise.",
    )
    parser.add_argument(
        "--affflow-dummy-token",
        action="store_true",
        default=False,
        help="Phase 1 AffFlow option: append one embedded dummy TS token after text tokens and decode from it; forced on for affflow synthetic Phase 1.",
    )
    parser.add_argument(
        "--affflow-dummy-init",
        choices=["zero", "randn"],
        default="zero",
        help="Phase 1 dummy TS input: zeros or Gaussian noise before embedding.",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Path to a Phase 1 synthetic AffFlow checkpoint (.pt) to load before real-data training or conditioned zeroshot.",
    )
    parser.add_argument(
        "--text-gate",
        action="store_true",
        default=False,
        help="Add a learnable scalar offset to the attention logits of the text "
             "prefix. Trained to negative values it removes the text tokens from "
             "attention, making the text-free model reachable by training.",
    )
    parser.add_argument(
        "--stage-a-lora-checkpoint",
        type=Path,
        default=None,
        help="Path to a ts_lora_only Stage-A checkpoint (.pt). Its LoRA weights are "
             "loaded into the backbone and frozen (dropout disabled); requires "
             "--use-lora and the p2_frozen_lora_affflow profile.",
    )
    parser.add_argument(
        "--affflow-loss-weight",
        type=float,
        default=None,
        help="AffFlow regularization weight for real-data training with a pretrained AffFlow (config default 0.0).",
    )
    parser.add_argument(
        "--affflow-adapter",
        action="store_true",
        default=False,
        help="Add a trainable linear adapter between frozen AffFlow/cVAE output and backbone input.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=False,
        help="Apply LoRA adapters to the backbone during real-data training.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default 8).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (default 16).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default 0.05).",
    )
    parser.add_argument(
        "--lora-modality-routed",
        action="store_true",
        default=False,
        help="Route text-prefix positions through separate K_text/V_text "
             "projections so they get their own LoRA adapters distinct from "
             "the TS-path K/V adapters. Requires --use-lora.",
    )
    parser.add_argument(
        "--llm-model-fusion",
        type=str,
        default=None,
        help="Override the text-encoder alias used for embeddings (see experiments/utils.py). "
             "Affects which cached .pt file is loaded.",
    )

    args = parser.parse_args()
    if args.affflow_text_mode is None:
        args.affflow_text_mode = "concatext" if args.affflow_text_concat else "adaln"
    args.affflow_text_concat = args.affflow_text_mode == "concatext"
    if not args.run_tag:
        parser.error("--run-tag is required.")
    if not args.run_stage:
        parser.error("--run-stage is required. Use one of: plain, p1, sp, p2.")
    if args.lora_modality_routed and not args.use_lora:
        parser.error("--lora-modality-routed requires --use-lora.")
    if args.affflow_num_blocks is not None and args.affflow_num_blocks < 1:
        parser.error("--affflow-num-blocks must be >= 1.")
    if args.affflow_mlp_ratio is not None and args.affflow_mlp_ratio < 1:
        parser.error("--affflow-mlp-ratio must be >= 1.")
    if args.generator_type == "residual_mlp":
        if args.tsfm_model != "timemoe":
            parser.error(
                "--generator-type residual_mlp is currently implemented only "
                "for --tsfm-model timemoe."
            )
        if args.flow_init == "noise":
            parser.error(
                "--generator-type residual_mlp requires the matched anchor "
                "initialization; do not use --flow-init noise."
            )
    if args.observed_missing_policy == "seq_mean_impute":
        if args.active_dataset != "time_imm" or args.timeimm_irregularity != "resampled":
            parser.error(
                "--observed-missing-policy seq_mean_impute is currently supported "
                "only with --active-dataset time_imm --timeimm-irregularity resampled."
            )
    try:
        validate_and_resolve_train_profile(args)
        resolve_batch_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run(cfg, args) -> None:

    trace_root = Path(args.trace_root)
    metrics_csv_path = (
        Path(args.metrics_csv)
        if args.metrics_csv is not None
        else trace_root / "metrics.csv"
    )

    if args.active_dataset == "time_imm" and args.timeimm_datasets:
        missing = [
            d for d in args.timeimm_datasets if d not in cfg.timeimm.dataset_presets]
        if missing:
            raise ValueError(f"Unknown Time-IMM datasets: {missing}")
        cfg.timeimm.datasets = list(args.timeimm_datasets)

    if args.active_dataset == "time_mmd" and args.domains:
        missing = [
            d for d in args.domains if d not in cfg.time_mmd.dataset_presets]
        if missing:
            raise ValueError(f"Unknown Time-MMD domains: {missing}")
        cfg.time_mmd.dataset_presets = {
            domain: cfg.time_mmd.dataset_presets[domain]
            for domain in args.domains
        }

    if args.active_dataset == "cgtsf" and args.domains:
        missing = [d for d in args.domains if d not in cfg.cgtsf.dataset_presets]
        if missing:
            raise ValueError(f"Unknown CGTSF domains: {missing}")
        cfg.cgtsf.dataset_presets = {
            domain: cfg.cgtsf.dataset_presets[domain]
            for domain in args.domains
        }

    if args.active_dataset == "synthetic":
        experiment = SyntheticExperiment(
            cfg=cfg,
            args=args,
            metrics_csv=metrics_csv_path,
        )
    else:
        experiment_classes = {
            "time_mmd": TimeMMDExperiment,
            "time_imm": TimeIMMExperiment,
            "cgtsf": CGTSFExperiment,
        }

        experiment_cls = experiment_classes.get(args.active_dataset)
        if experiment_cls is None:
            raise ValueError(f"Unknown dataset: {args.active_dataset}")

        experiment = experiment_cls(
            cfg=cfg,
            args=args,
            metrics_csv=metrics_csv_path,
        )

    experiment.run()


if __name__ == "__main__":
    args = parse_args()
    if args.campaign_id is None:
        args.campaign_id = os.getenv("CAMPAIGN_ID")
    if args.campaign_spec_path is None:
        args.campaign_spec_path = os.getenv("CAMPAIGN_SPEC_PATH")
    if args.campaign_label is None:
        args.campaign_label = os.getenv("CAMPAIGN_LABEL")
    if args.campaign_note is None:
        args.campaign_note = os.getenv("CAMPAIGN_NOTE")
    if args.registry_path is None and os.getenv("CAMPAIGN_REGISTRY_PATH"):
        args.registry_path = Path(os.getenv("CAMPAIGN_REGISTRY_PATH", ""))
    if args.trace_root is None:
        trace_timestamp = os.getenv("TRACE_TIMESTAMP") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.trace_timestamp = trace_timestamp
        args.trace_root = str(
            PROJ_PATH
            / "results"
            / "run_traces"
            / f"{args.run_mode}_{args.tsfm_model}_{args.active_dataset}_affflow_{args.run_stage}_{args.run_tag}_{trace_timestamp}"
        )
    else:
        args.trace_timestamp = os.getenv("TRACE_TIMESTAMP") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    set_global_seed(args.seed)


    cfg = copy.deepcopy(CONFIG)
    if args.affflow_loss_weight is not None:
        cfg.train.affflow_loss_weight = args.affflow_loss_weight
    cfg.affcon.patch_position_mode = args.patch_position_mode
    cfg.affcon.text_attention_mode = args.text_attention_mode
    cfg.affcon.text_position_mode = args.text_position_mode
    cfg.affcon.text_injection_layout = args.text_injection_layout
    cfg.affcon.text_embedding_control = args.text_embedding_control
    cfg.affcon.text_embedding_random_seed = args.text_embedding_random_seed
    cfg.affcon.affiliated_token_control = args.affiliated_token_control
    cfg.affcon.observed_missing_policy = args.observed_missing_policy
    cfg.affcon.affflow_dummy_token = args.affflow_dummy_token
    cfg.affcon.affflow_dummy_init = args.affflow_dummy_init
    if args.force_flip_invariance is not None:
        cfg.affcon.force_flip_invariance = args.force_flip_invariance
    if args.timesfm_infer_is_positive is not None:
        cfg.affcon.timesfm_infer_is_positive = args.timesfm_infer_is_positive
    if args.timesfm_decoupled_flip is not None:
        cfg.affcon.timesfm_decoupled_flip = args.timesfm_decoupled_flip
    if args.compute_crps is not None:
        cfg.affcon.compute_crps = args.compute_crps
    if args.crps_num_samples is not None:
        cfg.affcon.crps_num_samples = args.crps_num_samples
    if args.crps_dump is not None:
        cfg.affcon.crps_dump = args.crps_dump
    if args.synthetic_mix_ratio is not None:
        if args.synthetic_mix_ratio < 0:
            raise ValueError("--synthetic-mix-ratio must be non-negative.")
        cfg.affcon.synthetic_mix_ratio = args.synthetic_mix_ratio
    if args.generator_type is not None:
        cfg.affcon.generator_type = args.generator_type
    if args.affflow_num_blocks is not None:
        cfg.affcon.affflow_num_blocks = args.affflow_num_blocks
    if args.affflow_mlp_ratio is not None:
        cfg.affcon.affflow_mlp_ratio = args.affflow_mlp_ratio
    if args.affflow_nfe is not None:
        cfg.affcon.affflow_nfe = args.affflow_nfe
    if args.affflow_train_grid is not None:
        cfg.affcon.affflow_train_random_grid = args.affflow_train_grid == "random"
    if args.flow_init is not None:
        cfg.affcon.flow_init = args.flow_init
    if args.llm_model_fusion is not None:
        cfg.timeimm.llm_model_fusion = args.llm_model_fusion
        cfg.time_mmd.llm_model_fusion = args.llm_model_fusion
    run(cfg=cfg, args=args)
