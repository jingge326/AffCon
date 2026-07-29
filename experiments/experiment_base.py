
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.coms_lora import extract_lora_state_dict, has_lora_parameters, load_lora_state_dict
from models.model_sundial import SundialForecaster, load_backbones
from models.model_timesfm import TimesFMForecaster, load_timesfm_backbone
from models.model_timemoe import TimeMoEForecaster, load_timemoe_backbone
from experiments.training_env import (
    build_v5_optimizer,
    build_warmup_cosine_scheduler,
    current_lrs,
    get_train_profile,
    make_batch_metadata,
    next_fallback_micro_batch,
    resolve_batch_plan,
)

PROJ_PATH = Path(__file__).resolve().parent.parent
DATA_MODE = "affflow"

EPS = 1e-8
_GIT_METADATA_CACHE: Optional[Dict[str, Any]] = None
_TEXT_EMBEDDING_CONTROL_CACHE: Dict[Tuple[int, str, int, int], torch.Tensor] = {}




def format_run_id(
    run_mode: str,
    tsfm_model: Optional[str],
    active_dataset: str,
    timestamp: str,
    domain: str,
    run_stage: str,
    run_tag: str,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    variant_tag: Optional[str] = None,
) -> str:
    parts = [
        run_mode,
        tsfm_model or "base",
        DATA_MODE,
        run_stage,
        run_tag if not variant_tag else f"{run_tag}_{variant_tag}",
        active_dataset,
        domain or "all",
    ]
    if lookback is not None and horizon is not None:
        parts.append(f"L{lookback}_H{horizon}")
    parts.append(timestamp)
    return "_".join(parts)


def move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return moved
    return batch


def _stable_text_seed(base_seed: int, sample_key: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{sample_key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def _get_text_embedding_control_batch_info(
    batch: dict,
) -> Tuple[Optional[torch.Tensor], List[Tuple[int, str, torch.Tensor, int]], int]:
    text_embeddings = batch.get("text_embeddings")
    if not torch.is_tensor(text_embeddings):
        return None, [], 0

    sample_keys = batch.get("sample_keys")
    if not isinstance(sample_keys, (list, tuple)) or len(sample_keys) != text_embeddings.shape[0]:
        raise ValueError(
            "Deterministic text embedding control requires batch['sample_keys'] "
            "with one stable key per sample."
        )

    text_mask = batch.get("patch_text_mask")
    if text_mask is None:
        text_mask = batch.get("text_mask")
    if text_mask is None:
        text_mask = torch.ones(
            text_embeddings.shape[:2],
            device=text_embeddings.device,
            dtype=torch.bool,
        )
    else:
        text_mask = text_mask.to(device=text_embeddings.device, dtype=torch.bool)

    d_txt = text_embeddings.shape[2]
    row_info: List[Tuple[int, str, torch.Tensor, int]] = []
    for row_idx, sample_key in enumerate(sample_keys):
        valid_mask = text_mask[row_idx]
        num_valid = int(valid_mask.sum().item())
        row_info.append((row_idx, str(sample_key), valid_mask, num_valid))

    return text_embeddings, row_info, d_txt


def _prime_random_fixed_embeddings_for_batch(batch: dict, *, base_seed: int) -> None:
    text_embeddings, row_info, d_txt = _get_text_embedding_control_batch_info(batch)
    if text_embeddings is None:
        return

    for _, sample_key, _, num_valid in row_info:
        if num_valid == 0:
            continue
        cache_key = (base_seed, sample_key, num_valid, d_txt)
        cached = _TEXT_EMBEDDING_CONTROL_CACHE.get(cache_key)
        if cached is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(_stable_text_seed(base_seed, sample_key))
            cached = torch.randn((num_valid, d_txt), generator=gen, dtype=torch.float32)
            _TEXT_EMBEDDING_CONTROL_CACHE[cache_key] = cached


def apply_text_embedding_control(
    batch: dict,
    *,
    control: str,
    base_seed: int,
) -> dict:
    if control in ("", "none", None):
        return batch
    if control != "random_fixed":
        raise ValueError(f"Unsupported text_embedding_control: {control}")

    text_embeddings, row_info, d_txt = _get_text_embedding_control_batch_info(batch)
    if text_embeddings is None:
        return batch

    controlled = torch.zeros_like(text_embeddings)
    for row_idx, sample_key, valid_mask, num_valid in row_info:
        if num_valid == 0:
            continue
        cache_key = (base_seed, sample_key, num_valid, d_txt)
        rand_valid = _TEXT_EMBEDDING_CONTROL_CACHE.get(cache_key)
        if rand_valid is None:
            raise RuntimeError(
                "random_fixed embeddings were not primed before batch move "
                f"for sample_key={sample_key!r}."
            )
        controlled[row_idx, valid_mask] = rand_valid.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )

    batch["text_embeddings"] = controlled
    return batch


def setup_logger(
    run_id: str,
    log_dir: Path | str = PROJ_PATH / "log",
) -> logging.Logger:
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def collate_with_text(batch):
    xs, ys, texts = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    texts = list(texts)
    return x, y, texts


def filter_overrides(domain_window_overrides: Optional[dict], domain: str) -> Optional[dict]:
    if not domain_window_overrides:
        return None
    if domain in domain_window_overrides:
        return {domain: domain_window_overrides[domain]}
    return None


def effective_window(domain: str, overrides: Optional[dict], data_cfg) -> Tuple[int, int]:
    if overrides and domain in overrides:
        return overrides[domain]
    return data_cfg.lookback, data_cfg.horizon


def inverse_scale_tensor(arr: torch.Tensor, domain: str, dataset) -> torch.Tensor:
    if not getattr(dataset, "scale", False):
        return arr
    mean, std = getattr(dataset, "domain_scalers", {}).get(domain, (0.0, 1.0))
    mean_t = torch.tensor(mean, device=arr.device, dtype=arr.dtype)
    std_t = torch.tensor(std, device=arr.device, dtype=arr.dtype)
    return arr * std_t + mean_t


def unwrap_base_dataset(ds):
    base = ds
    seen = set()
    while id(base) not in seen:
        seen.add(id(base))
        if hasattr(base, "dataset"):
            base = base.dataset
            continue
        if hasattr(base, "base"):
            base = base.base
            continue
        break
    return base


def dataset_indices(ds) -> List[int]:
    if hasattr(ds, "indices"):
        return list(ds.indices)
    return list(range(len(ds)))


def fetch_sample(base_dataset, idx: int):
    if hasattr(base_dataset, "windows"):
        return base_dataset.windows[idx]
    if hasattr(base_dataset, "samples"):
        return base_dataset.samples[idx]
    return None


def select_target(arr: torch.Tensor, target_idx: int) -> torch.Tensor:
    if arr.dim() == 1:
        return arr
    if arr.dim() == 2:
        return arr[:, target_idx]
    return arr


def log_metrics(
    metrics_csv: Optional[Path],
    *,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    run_mode: str,
    tsfm_model: Optional[str] = None,
    active_dataset: Optional[str] = None,
    domain: str,
    lookback: int,
    horizon: int,
    test_metrics: dict,
    run_stage: str,
    run_tag: str,
    lookback_multiplier: Optional[int] = None,
    text_attention_mode: Optional[str] = None,
    text_position_mode: Optional[str] = None,
    text_injection_layout: Optional[str] = None,
    text_embedding_control: Optional[str] = None,
    affiliated_token_control: Optional[str] = None,
) -> None:
    if metrics_csv is None:
        return
    metrics_csv = Path(metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id or "",
        "seed": seed if seed is not None else "",
        "run_mode": run_mode,
        "tsfm_model": tsfm_model or "",
        "active_dataset": active_dataset or "",
        "domain": domain,
        "lookback": lookback,
        "horizon": horizon,
        "test_mse": test_metrics.get("mse", ""),
        "test_mae": test_metrics.get("mae", ""),
        "test_rmse": test_metrics.get("rmse", ""),
        "test_mape": test_metrics.get("mape", ""),
        "test_mspe": test_metrics.get("mspe", ""),
        "test_crps": test_metrics.get("crps", ""),
        "test_crps_norm": test_metrics.get("crps_norm", ""),
        "test_pit_cal_err": test_metrics.get("pit_cal_err", ""),
        "run_stage": run_stage,
        "run_tag": run_tag,
        "data_mode": DATA_MODE,
        "lookback_multiplier": lookback_multiplier if lookback_multiplier is not None else "",
        "text_attention_mode": text_attention_mode or "",
        "text_position_mode": text_position_mode or "",
        "text_injection_layout": text_injection_layout or "",
        "text_embedding_control": text_embedding_control or "",
        "affiliated_token_control": affiliated_token_control or "",
    }
    fieldnames = [
        "timestamp",
        "run_id",
        "seed",
        "run_mode",
        "tsfm_model",
        "active_dataset",
        "domain",
        "lookback",
        "horizon",
        "test_mse",
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_mspe",
        "test_crps",
        "test_crps_norm",
        "test_pit_cal_err",
        "run_stage",
        "run_tag",
        "data_mode",
        "lookback_multiplier",
        "text_attention_mode",
        "text_position_mode",
        "text_injection_layout",
        "text_embedding_control",
        "affiliated_token_control",
    ]
    existing_rows: list[dict] = []
    if metrics_csv.exists():
        with metrics_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    normalized_rows: list[dict] = []
    for existing in existing_rows:
        merged = {k: "" for k in fieldnames}
        merged.update(existing)
        normalized_rows.append(merged)

    identity_fields = [
        "run_mode",
        "seed",
        "tsfm_model",
        "active_dataset",
        "domain",
        "lookback",
        "horizon",
        "run_stage",
        "run_tag",
        "data_mode",
        "lookback_multiplier",
        "text_attention_mode",
        "text_position_mode",
        "text_injection_layout",
        "text_embedding_control",
        "affiliated_token_control",
    ]
    identity = {k: str(row.get(k, "")) for k in identity_fields}
    filtered_rows = [
        existing for existing in normalized_rows
        if {k: str(existing.get(k, "")) for k in identity_fields} != identity
    ]
    filtered_rows.append(row)

    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for output_row in filtered_rows:
            writer.writerow(output_row)


def alignment_ablation_fields(
    text_attention_mode: Optional[str] = None,
    text_position_mode: Optional[str] = None,
    text_injection_layout: Optional[str] = None,
    text_embedding_control: Optional[str] = None,
    affiliated_token_control: Optional[str] = None,
) -> Dict[str, str]:
    fields = {
        "text_attention_mode": text_attention_mode or "temporal_causal",
        "text_position_mode": text_position_mode or "token_aligned",
        "text_injection_layout": text_injection_layout or "distributed",
        "text_embedding_control": text_embedding_control or "none",
        "affiliated_token_control": affiliated_token_control or "none",
    }
    return fields


def build_alignment_variant_tag(
    text_attention_mode: Optional[str] = None,
    text_position_mode: Optional[str] = None,
    text_injection_layout: Optional[str] = None,
    text_embedding_control: Optional[str] = None,
    affiliated_token_control: Optional[str] = None,
) -> Optional[str]:
    fields = alignment_ablation_fields(
        text_attention_mode=text_attention_mode,
        text_position_mode=text_position_mode,
        text_injection_layout=text_injection_layout,
        text_embedding_control=text_embedding_control,
        affiliated_token_control=affiliated_token_control,
    )
    if not any(fields.values()):
        return None
    value_map = {
        "temporal_causal": "tc",
        "global_prefix": "gp",
        "token_aligned": "tok",
        "raw_timestamp": "raw",
        "zero": "zero",
        "distributed": "dist",
        "collapsed_global": "cglob",
        "none": "real",
        "random_fixed": "rand",
        "random": "randtok",
    }
    parts = []
    if fields["text_attention_mode"]:
        parts.append(
            f"ta{value_map.get(fields['text_attention_mode'], fields['text_attention_mode'])}"
        )
    if fields["text_position_mode"]:
        parts.append(
            f"tp{value_map.get(fields['text_position_mode'], fields['text_position_mode'])}"
        )
    if fields["text_injection_layout"]:
        parts.append(
            f"ti{value_map.get(fields['text_injection_layout'], fields['text_injection_layout'])}"
        )
    if fields["text_embedding_control"] not in ("", "none", None):
        parts.append(
            f"te{value_map.get(fields['text_embedding_control'], fields['text_embedding_control'])}"
        )
    if fields["affiliated_token_control"] not in ("", "none", None):
        parts.append(
            f"at{value_map.get(fields['affiliated_token_control'], fields['affiliated_token_control'])}"
        )
    if not parts:
        return None
    return "_".join(parts)


def max_horizon_from_overrides(
    domain: str,
    active_overrides: Optional[dict],
    fallback_horizon: int,
) -> int:
    if active_overrides and domain in active_overrides:
        return active_overrides[domain][1]
    return fallback_horizon


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return {k: _jsonify(v) for k, v in vars(value).items()}
    if dataclasses.is_dataclass(value):
        return _jsonify(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    return repr(value)


def get_git_metadata(repo_root: Path = PROJ_PATH) -> Dict[str, Any]:
    global _GIT_METADATA_CACHE
    if _GIT_METADATA_CACHE is not None:
        return _GIT_METADATA_CACHE

    def _run_git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        return proc.stdout.strip()

    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--short")
    _GIT_METADATA_CACHE = {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status) if status is not None else None,
        "status_short": status,
    }
    return _GIT_METADATA_CACHE


def build_model_manifest(model: Optional[nn.Module]) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    manifest = {
        "class_name": model.__class__.__name__,
        "repr": repr(model),
        "num_parameters": total_params,
        "num_trainable_parameters": trainable_params,
    }
    generator = getattr(model, "affflow", None)
    if isinstance(generator, nn.Module):
        manifest["generator"] = {
            "attribute": "affflow",
            "class_name": generator.__class__.__name__,
            "num_parameters": sum(p.numel() for p in generator.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in generator.parameters() if p.requires_grad
            ),
        }
    return manifest


def build_checkpoint_payload(
    state_dict: Dict[str, torch.Tensor],
    *,
    run_id: str,
    manifest_path: Optional[Path],
    args: Any,
    cfg: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "metadata": {
            "schema_version": 1,
            "run_id": run_id,
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "args": _jsonify(vars(args)),
            "resolved_config": _jsonify(cfg),
        },
    }
    if metadata:
        payload["metadata"].update(_jsonify(metadata))
    return payload


def extract_checkpoint_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def extract_checkpoint_lora_state_dict(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        lora_state = payload.get("lora_state_dict")
        if isinstance(lora_state, dict):
            return lora_state
    return None


def write_manifest(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def scheduler_context_from_env() -> Dict[str, Optional[str]]:
    backend = os.environ.get("SCHEDULER_BACKEND")
    job_id = os.environ.get("SCHEDULER_JOB_ID")
    job_name = os.environ.get("SCHEDULER_JOB_NAME")
    if not backend and os.environ.get("SLURM_JOB_ID"):
        backend = "slurm"
        job_id = job_id or os.environ.get("SLURM_JOB_ID")
    return {
        "scheduler": backend,
        "scheduler_job_id": job_id,
        "scheduler_job_name": job_name,
    }


def append_campaign_registry_row(
    registry_path: Optional[Path | str],
    payload: Dict[str, Any],
) -> None:
    if registry_path in (None, ""):
        return
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, sort_keys=True)
        f.write("\n")


def write_model_structure(path: Path, model: Optional[nn.Module]) -> Optional[Path]:
    if model is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{model}\n")
    return path


def snapshot_helper_scripts(
    helper_scripts: List[str],
    dest_dir: Path,
) -> List[str]:
    copied: List[str] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in helper_scripts:
        if not entry:
            continue
        src = Path(entry).expanduser()
        if not src.is_absolute():
            src = (PROJ_PATH / src).resolve()
        if not src.exists() or not src.is_file():
            continue
        target = dest_dir / src.name
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while target.exists():
            target = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src, target)
        copied.append(str(target))
    return copied




class ExperimentBase(ABC):

    def __init__(
        self,
        cfg,
        args,
        data_cfg,
        *,
        metrics_csv: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.args = args
        self.data_cfg = data_cfg
        self.device = torch.device(args.device)
        self.metrics_csv = metrics_csv or (
            PROJ_PATH / "results" / "metrics.csv")
        self.run_tag = args.run_tag
        self.run_stage = args.run_stage
        self.run_timestamp = getattr(
            args,
            "trace_timestamp",
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        )

        self.model_choice = args.tsfm_model

        self.use_patch_text_embeds = args.use_patch_text_embeds

        self.run_id: Optional[str] = None
        self.logger: Optional[logging.Logger] = None
        self.model = None
        default_trace_root = (
            PROJ_PATH
            / "results"
            / "run_traces"
            / f"{self.run_mode}_{self.model_choice}_{self.active_dataset}_{self.data_mode}_{self.run_stage}_{self.run_tag}_{self.run_timestamp}"
        )
        self.trace_root = Path(getattr(args, "trace_root", "")).expanduser() if getattr(args, "trace_root", None) else default_trace_root
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self._snapshot_launch_context()

    @property
    def active_dataset(self) -> str:
        return self.args.active_dataset

    @property
    def run_mode(self) -> str:
        return self.args.run_mode

    @property
    def data_mode(self) -> str:
        return DATA_MODE

    def current_alignment_fields(self) -> Dict[str, str]:
        return alignment_ablation_fields(
            text_attention_mode=getattr(self.cfg.affcon, "text_attention_mode", None),
            text_position_mode=getattr(self.cfg.affcon, "text_position_mode", None),
            text_injection_layout=getattr(self.cfg.affcon, "text_injection_layout", None),
            text_embedding_control=getattr(self.cfg.affcon, "text_embedding_control", None),
            affiliated_token_control=getattr(self.cfg.affcon, "affiliated_token_control", None),
        )

    def current_variant_tag(self) -> Optional[str]:
        return build_alignment_variant_tag(
            text_attention_mode=getattr(self.cfg.affcon, "text_attention_mode", None),
            text_position_mode=getattr(self.cfg.affcon, "text_position_mode", None),
            text_injection_layout=getattr(self.cfg.affcon, "text_injection_layout", None),
            text_embedding_control=getattr(self.cfg.affcon, "text_embedding_control", None),
            affiliated_token_control=getattr(self.cfg.affcon, "affiliated_token_control", None),
        )

    def _move_batch(self, batch):
        if not isinstance(batch, dict):
            return batch
        moved = move_batch_to_device(batch, self.device)
        return apply_text_embedding_control(
            moved,
            control=getattr(self.cfg.affcon, "text_embedding_control", "none"),
            base_seed=int(getattr(self.cfg.affcon, "text_embedding_random_seed", 0)),
        )

    def _prime_text_embedding_control_cache(self, *loaders: DataLoader) -> None:
        control = getattr(self.cfg.affcon, "text_embedding_control", "none")
        if control != "random_fixed":
            return
        base_seed = int(getattr(self.cfg.affcon, "text_embedding_random_seed", 0))
        for loader in loaders:
            if loader is None:
                continue
            for batch in loader:
                if isinstance(batch, dict):
                    _prime_random_fixed_embeddings_for_batch(
                        batch,
                        base_seed=base_seed,
                    )

    def _crop_synthetic_mix_batch(
        self,
        batch: Dict[str, Any],
        *,
        lookback: int,
        horizon: int,
    ) -> Dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError("Synthetic mix loader must emit dict batches.")
        lookback = int(lookback)
        horizon = int(horizon)
        if lookback <= 0 or horizon <= 0:
            raise ValueError(
                f"Synthetic mixing requires positive lookback/horizon, got {lookback}/{horizon}."
            )

        obs = batch["observed_data"]
        obs_mask = batch["observed_mask"]
        fut = batch["data_to_predict"]
        fut_mask = batch["mask_predicted_data"]
        B, _, C = obs.shape
        device = obs.device
        dtype = obs.dtype

        out_obs = torch.zeros(B, lookback, C, device=device, dtype=dtype)
        out_obs_mask = torch.zeros(B, lookback, C, device=device, dtype=obs_mask.dtype)
        take_hist = min(obs.shape[1], lookback)
        out_obs[:, -take_hist:, :] = obs[:, -take_hist:, :]
        out_obs_mask[:, -take_hist:, :] = obs_mask[:, -take_hist:, :]

        out_fut = torch.zeros(B, horizon, fut.shape[-1], device=fut.device, dtype=fut.dtype)
        out_fut_mask = torch.zeros(
            B, horizon, fut_mask.shape[-1], device=fut_mask.device, dtype=fut_mask.dtype
        )
        take_fut = min(fut.shape[1], horizon)
        out_fut[:, :take_fut, :] = fut[:, :take_fut, :]
        out_fut_mask[:, :take_fut, :] = fut_mask[:, :take_fut, :]

        batch = dict(batch)
        batch["observed_data"] = out_obs
        batch["observed_mask"] = out_obs_mask
        batch["observed_tp"] = torch.arange(
            lookback, device=device, dtype=torch.float32
        ).unsqueeze(0).expand(B, -1)
        batch["data_to_predict"] = out_fut
        batch["mask_predicted_data"] = out_fut_mask
        batch["tp_to_predict"] = torch.arange(
            lookback, lookback + horizon, device=device, dtype=torch.float32
        ).unsqueeze(0).expand(B, -1)
        if torch.is_tensor(batch.get("text_tp")):
            batch["text_tp"] = torch.full_like(
                batch["text_tp"].to(device=device, dtype=torch.float32),
                float(lookback - 1),
            )
        batch["target_idx"] = 0
        batch["synthetic_mix"] = True
        return batch

    def _build_synthetic_mix_loader(
        self,
        *,
        lookback: int,
        horizon: int,
        batch_size: int,
    ) -> DataLoader:
        from experiments.data_synthetic import build_synthetic_dataloaders

        data_root = Path(self.cfg.synthetic.data_path)
        jsonl_path = data_root / "synthetic_pairs_history_forecast_split.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Synthetic mixing requires forecast-style data at {jsonl_path}."
            )
        train_loader, _ = build_synthetic_dataloaders(
            jsonl_path=jsonl_path,
            cache_dir=data_root,
            batch_size=batch_size,
            train_ratio=self.cfg.synthetic.train_ratio,
            device=str(self.device),
            llm_model=getattr(self.data_cfg, "llm_model_fusion", "Qwen3_E4B"),
            llm_layers=getattr(self.data_cfg, "llm_layers_fusion", None),
            phase1_mode="forecast",
        )
        base_collate = train_loader.collate_fn

        def collate_and_crop(samples):
            synthetic_batch = base_collate(samples)
            return self._crop_synthetic_mix_batch(
                synthetic_batch,
                lookback=lookback,
                horizon=horizon,
            )

        train_loader.collate_fn = collate_and_crop
        return train_loader

    def _synthetic_mix_ratio_for_epoch(self, epoch: int) -> float:
        ratio = float(getattr(self.cfg.affcon, "synthetic_mix_ratio", 0.0))
        if ratio <= 0:
            return 0.0
        decay = float(getattr(self.cfg.affcon, "synthetic_mix_decay", 0.5))
        min_ratio = float(getattr(self.cfg.affcon, "synthetic_mix_min_ratio", 1e-3))
        ratio *= decay ** int(epoch)
        return ratio if ratio >= min_ratio else 0.0

    @staticmethod
    def _expected_synthetic_batches(real_batches: int, ratio: float) -> int:
        return int(math.floor(max(real_batches, 0) * max(ratio, 0.0) + 1e-9))

    def _set_generator_trainable(self, model: nn.Module, trainable: bool) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name in ("affflow", "affflow_adapter"):
            module = getattr(model, name, None)
            if module is None:
                continue
            module.requires_grad_(trainable)
            counts[name] = int(sum(p.numel() for p in module.parameters()))
        return counts

    def _log_trainable_parameter_summary(self, model: nn.Module, label: str) -> None:
        total = 0
        by_group: Dict[str, int] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            total += param.numel()
            group = name.split(".", 1)[0]
            by_group[group] = by_group.get(group, 0) + param.numel()
        self.logger.info(
            f"[Trainable] {label}: total={total} groups={dict(sorted(by_group.items()))}"
        )

    def setup_run_logger(
        self,
        domain: str,
        lookback: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> logging.Logger:
        self.run_id = format_run_id(
            run_mode=self.run_mode,
            tsfm_model=self.model_choice,
            active_dataset=self.active_dataset,
            timestamp=self.run_timestamp,
            domain=domain or "all",
            run_stage=self.run_stage,
            run_tag=self.run_tag,
            lookback=lookback,
            horizon=horizon,
            variant_tag=self.current_variant_tag(),
        )
        self.logger = setup_logger(self.run_id, log_dir=self.trace_root / "logs")
        return self.logger

    def manifest_path(self, run_id: Optional[str] = None) -> Path:
        resolved_run_id = run_id or self.run_id or "unknown_run"
        return self.trace_root / "manifests" / f"{resolved_run_id}.json"

    def model_structure_path(self, run_id: Optional[str] = None) -> Path:
        resolved_run_id = run_id or self.run_id or "unknown_run"
        return self.trace_root / "manifests" / f"{resolved_run_id}.model.txt"

    def forecast_dir(self) -> Path:
        path = self.trace_root / "forecasts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def checkpoint_dir(self) -> Path:
        path = self.trace_root / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _snapshot_launch_context(self) -> None:
        launch_dir = self.trace_root / "launch"
        launch_dir.mkdir(parents=True, exist_ok=True)
        (launch_dir / "argv.txt").write_text(
            shlex.join(sys.argv) + "\n",
            encoding="utf-8",
        )
        write_manifest(
            launch_dir / "args.json",
            {"args": _jsonify(vars(self.args))},
        )
        write_manifest(
            launch_dir / "git.json",
            {"git": get_git_metadata()},
        )
        helper_scripts = list(getattr(self.args, "helper_scripts", []) or [])
        copied = snapshot_helper_scripts(helper_scripts, launch_dir / "helper_scripts")
        write_manifest(
            launch_dir / "trace_bundle.json",
            {
                "trace_root": str(self.trace_root),
                "metrics_csv": str(self.metrics_csv),
                "helper_scripts": helper_scripts,
                "copied_helper_scripts": copied,
                "campaign_id": getattr(self.args, "campaign_id", None),
                "campaign_spec_path": getattr(self.args, "campaign_spec_path", None),
                "campaign_label": getattr(self.args, "campaign_label", None),
                "campaign_note": getattr(self.args, "campaign_note", None),
                "registry_path": str(getattr(self.args, "registry_path", "") or ""),
            },
        )

    def write_run_manifest(
        self,
        *,
        status: str,
        domain: str,
        lookback: Optional[int],
        horizon: Optional[int],
        model: Optional[nn.Module] = None,
        run_id: Optional[str] = None,
        forecast_csv: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        test_metrics: Optional[Dict[str, Any]] = None,
        lookback_multiplier: Optional[int] = None,
        rows_written: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        resolved_run_id = run_id or self.run_id
        if resolved_run_id is None:
            raise ValueError("run_id must be set before writing a manifest")
        model_structure_path = write_model_structure(
            self.model_structure_path(resolved_run_id),
            model,
        )

        dataset_settings = None
        settings_getter = getattr(self.data_cfg, "dataset_settings", None)
        if callable(settings_getter):
            try:
                dataset_settings = settings_getter(domain)
            except Exception:
                dataset_settings = None

        manifest = {
            "schema_version": 1,
            "status": status,
            "run_id": resolved_run_id,
            "run_timestamp": self.run_timestamp,
            "project_root": str(PROJ_PATH),
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git": get_git_metadata(),
            "run": {
                "run_mode": self.run_mode,
                "active_dataset": self.active_dataset,
                "campaign_id": getattr(self.args, "campaign_id", None),
                "campaign_spec_path": getattr(self.args, "campaign_spec_path", None),
                "campaign_label": getattr(self.args, "campaign_label", None),
                "campaign_note": getattr(self.args, "campaign_note", None),
                "domain": domain,
                "lookback": lookback,
                "horizon": horizon,
                "lookback_multiplier": lookback_multiplier,
                "tsfm_model": self.model_choice,
                "data_mode": self.data_mode,
                "run_stage": self.run_stage,
                "train_profile": getattr(self.args, "train_profile", None),
                "device": str(self.device),
                "run_tag": self.run_tag,
                **self.current_alignment_fields(),
                "loaded_pretrain_checkpoint": str(self.args.pretrain_checkpoint)
                if getattr(self.args, "pretrain_checkpoint", None) is not None
                else None,
            },
            "args": vars(self.args),
            "resolved_config": self.cfg,
            "dataset_settings": dataset_settings,
            "model": build_model_manifest(model),
            "metrics": {
                "val": val_metrics,
                "test": test_metrics,
            },
            "outputs": {
                "trace_root": str(self.trace_root),
                "log_file": str(self.trace_root / "logs" / f"{resolved_run_id}.log"),
                "manifest_path": str(self.manifest_path(resolved_run_id)),
                "model_structure_txt": str(model_structure_path) if model_structure_path is not None else None,
                "metrics_csv": str(self.metrics_csv) if self.metrics_csv is not None else None,
                "forecast_csv": str(forecast_csv) if forecast_csv is not None else None,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "forecast_rows": rows_written,
            },
        }
        if extra:
            manifest["extra"] = extra

        manifest_path = self.manifest_path(resolved_run_id)
        write_manifest(manifest_path, manifest)
        scheduler_ctx = scheduler_context_from_env()
        append_campaign_registry_row(
            getattr(self.args, "registry_path", None),
            {
                "event_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "campaign_id": getattr(self.args, "campaign_id", None),
                "campaign_spec_path": getattr(self.args, "campaign_spec_path", None),
                "campaign_label": getattr(self.args, "campaign_label", None),
                "campaign_note": getattr(self.args, "campaign_note", None),
                "run_id": resolved_run_id,
                "run_mode": self.run_mode,
                "active_dataset": self.active_dataset,
                "domain": domain,
                "tsfm_model": self.model_choice,
                "data_mode": self.data_mode,
                "run_stage": self.run_stage,
                "train_profile": getattr(self.args, "train_profile", None),
                "run_tag": self.run_tag,
                "trace_root": str(self.trace_root),
                "manifest_path": str(manifest_path),
                "git_commit": get_git_metadata().get("commit"),
                **scheduler_ctx,
            },
        )
        return manifest_path

    def resolve_run_window(
        self,
        domain: str,
        domain_overrides: Optional[dict] = None,
        fallback_horizon: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        if domain_overrides and domain in domain_overrides:
            lb, hz = domain_overrides[domain]
            return lb, fallback_horizon if fallback_horizon is not None else hz

        settings_getter = getattr(self.data_cfg, "dataset_settings", None)
        if callable(settings_getter):
            try:
                settings = settings_getter(domain)
            except Exception:
                settings = None
            if isinstance(settings, dict):
                lb = settings.get("lookback")
                hz = settings.get("horizon")
                return lb, fallback_horizon if fallback_horizon is not None else hz

        lb = getattr(self.data_cfg, "lookback", None)
        hz = getattr(self.data_cfg, "horizon", None)
        if fallback_horizon is not None:
            hz = fallback_horizon
        return lb, hz

    def build_model(
        self,
        horizon: int,
        num_channels: Optional[int] = None,
    ) -> nn.Module:
        use_text_embeddings = getattr(
            self.data_cfg, "use_text_embeddings", False)
        load_text_backbone = not use_text_embeddings

        use_lora = getattr(self.args, "use_lora", False)
        lora_r = getattr(self.args, "lora_r", 8)
        lora_alpha = getattr(self.args, "lora_alpha", 16)
        lora_dropout = getattr(self.args, "lora_dropout", 0.05)
        lora_modality_routed = getattr(
            self.args, "lora_modality_routed", False)

        if self.model_choice == "timesfm":
            timesfm_backbones = load_timesfm_backbone(
                timesfm_model_name=self.cfg.train.timesfm_model_name,
                qwen_model_name=self.cfg.train.qwen_model_name,
                device=self.device,
            )
            model = TimesFMForecaster(
                backbones=timesfm_backbones,
                horizon=horizon,
                cfg=self.cfg,
                run_mode=self.run_mode,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_modality_routed=lora_modality_routed,
            ).to(self.device)
        elif self.model_choice == "timemoe":
            tm_backbones = load_timemoe_backbone(
                timemoe_model_name=self.cfg.train.timemoe_model_name,
                qwen_model_name=self.cfg.train.qwen_model_name,
                device=self.device,
                use_flash_attn=self.cfg.train.use_flash_attn,
                freeze_text=self.cfg.train.freeze_text,
                load_text_backbone=load_text_backbone,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_modality_routed=lora_modality_routed,
            )
            model = TimeMoEForecaster(
                backbones=tm_backbones,
                horizon=horizon,
                cfg=self.cfg,
                num_channels=num_channels,
                run_mode=self.run_mode,
                use_lora=use_lora,
            ).to(self.device)
        else:
            backbones = load_backbones(
                sundial_model_name=self.cfg.train.sundial_model_name,
                qwen_model_name=self.cfg.train.qwen_model_name,
                device=self.device,
                use_flash_attn=self.cfg.train.use_flash_attn,
                freeze_sundial=self.cfg.train.freeze_sundial,
                freeze_text=self.cfg.train.freeze_text,
                load_text_backbone=load_text_backbone,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_modality_routed=lora_modality_routed,
            )
            model = SundialForecaster(
                backbones=backbones,
                horizon=horizon,
                cfg=self.cfg,
                num_channels=num_channels,
                run_mode=self.run_mode,
                use_lora=use_lora,
            ).to(self.device)

        pretrain_ckpt = getattr(self.args, "pretrain_checkpoint", None)
        loaded_lora_state = False
        if pretrain_ckpt is not None:
            ckpt_path = Path(pretrain_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Pretrain checkpoint not found: {ckpt_path}")
            payload = torch.load(
                ckpt_path, map_location=self.device, weights_only=True)
            state = extract_checkpoint_state_dict(payload)
            lora_state = extract_checkpoint_lora_state_dict(payload)
            if hasattr(model, 'affflow'):
                model.affflow.load_state_dict(state)
                print(f"[PretrainLoad] Loaded pre-trained {self.data_mode} from {ckpt_path}")
                if (self.run_mode == "zeroshot"
                        or self.cfg.train.freeze_affflow
                        or self.cfg.affcon.affflow_adapter):
                    model.affflow.requires_grad_(False)
                    if hasattr(model, "affflow_adapter"):
                        model.affflow_adapter.requires_grad_(False)
                    print("[PretrainLoad] affflow frozen (zeroshot=True, freeze_affflow=True, or affflow_adapter=True)")
                else:
                    print("[PretrainLoad] affflow will be fine-tuned on real data (freeze_affflow=False)")
            if use_lora and lora_state:
                backbone = getattr(model, "lora_backbone", None)
                if backbone is None or not has_lora_parameters(backbone):
                    raise RuntimeError(
                        f"Checkpoint {ckpt_path} includes LoRA weights, but the current model has no injected LoRA adapters."
                    )
                missing_keys, unexpected_keys = load_lora_state_dict(backbone, lora_state)
                if unexpected_keys:
                    raise RuntimeError(
                        f"Unexpected LoRA checkpoint keys for {ckpt_path}: {sorted(unexpected_keys)}"
                    )
                lora_missing = [key for key in missing_keys if "lora_" in key]
                if lora_missing:
                    raise RuntimeError(
                        f"Missing LoRA checkpoint keys for {ckpt_path}: {sorted(lora_missing)}"
                    )
                loaded_lora_state = True
                print(f"[PretrainLoad] Loaded LoRA adapter weights from {ckpt_path}")

        stage_a_ckpt = getattr(self.args, "stage_a_lora_checkpoint", None)
        if stage_a_ckpt is not None:
            ckpt_path = Path(stage_a_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Stage-A LoRA checkpoint not found: {ckpt_path}")
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            lora_state = extract_checkpoint_lora_state_dict(payload)
            if not lora_state:
                raise RuntimeError(f"Stage-A checkpoint has no lora_state_dict: {ckpt_path}")
            meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            meta_args = meta.get("args") or {}
            mismatches = []
            ckpt_model = meta.get("tsfm_model")
            if ckpt_model is not None and ckpt_model != self.model_choice:
                mismatches.append(f"tsfm_model {ckpt_model!r} != {self.model_choice!r}")
            for key in ("lora_r", "lora_alpha", "timeimm_datasets"):
                ours = getattr(self.args, key, None)
                theirs = meta_args.get(key)
                if theirs is not None and ours is not None and theirs != ours:
                    mismatches.append(f"{key}: checkpoint {theirs!r} != run {ours!r}")
            if mismatches:
                raise RuntimeError(
                    f"Stage-A checkpoint mismatch for {ckpt_path}: " + "; ".join(mismatches)
                )
            backbone = getattr(model, "lora_backbone", None)
            if backbone is None or not has_lora_parameters(backbone):
                raise RuntimeError(
                    "--stage-a-lora-checkpoint requires a backbone with injected "
                    "LoRA adapters (--use-lora)."
                )
            missing_keys, unexpected_keys = load_lora_state_dict(backbone, lora_state)
            if unexpected_keys:
                raise RuntimeError(
                    f"Unexpected Stage-A LoRA keys for {ckpt_path}: {sorted(unexpected_keys)}"
                )
            lora_missing = [key for key in missing_keys if "lora_" in key]
            if lora_missing:
                raise RuntimeError(
                    f"Missing Stage-A LoRA keys for {ckpt_path}: {sorted(lora_missing)}"
                )
            freeze_stage_a = True
            profile_name = getattr(self.args, "train_profile", None)
            if profile_name:
                try:
                    freeze_stage_a = get_train_profile(profile_name).freeze_lora
                except (KeyError, ValueError):
                    pass
            loaded_lora_state = True
            if freeze_stage_a:
                frozen_tensors = 0
                for name, param in backbone.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(False)
                        frozen_tensors += 1
                for mod_name, module in backbone.named_modules():
                    if "lora_dropout" in mod_name and isinstance(module, nn.Dropout):
                        module.p = 0.0
                print(
                    f"[StageALoad] Loaded Stage-A LoRA from {ckpt_path} "
                    f"({frozen_tensors} tensors frozen, LoRA dropout disabled)"
                )
            else:
                print(
                    f"[StageALoad] Loaded Stage-A LoRA from {ckpt_path} "
                    f"(warm start: LoRA stays trainable)"
                )

        if use_lora and self.run_mode == "zeroshot" and not loaded_lora_state:
            warning_msg = (
                "[LoRAWarn] --use-lora was set for zeroshot, but no LoRA adapter "
                "weights were loaded from the provided checkpoint. PEFT LoRA "
                "adapters remain at their zero-effect initialization, so the "
                "backbone behaves like a frozen backbone and results may be misleading."
            )
            if self.logger is not None:
                self.logger.warning(warning_msg)
            else:
                print(warning_msg)

        self.model = model
        return model

    @abstractmethod
    def build_dataloader(
        self,
        domain: str,
        domain_overrides: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @abstractmethod
    def save_forecasts(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        output_csv: Path,
        **kwargs,
    ) -> int:
        pass

    def train(
        self,
        domain: str,
        domain_overrides: Optional[dict] = None,
        model_horizon: Optional[int] = None,
        lookback_multiplier: Optional[int] = None,
    ) -> Tuple[nn.Module, dict, Path]:
        self.logger.info(
            f"[Run] Training domain: {domain}")

        profile = get_train_profile(self.args.train_profile)
        batch_plan = resolve_batch_plan(self.args)
        effective_batch = batch_plan.effective_batch_size
        initial_micro_batch = batch_plan.micro_batch_size
        min_batch = 1

        self.args.batch_size = initial_micro_batch

        train_loader, val_loader, test_loader = self.build_dataloader(
            domain=domain,
            domain_overrides=domain_overrides,
        )
        self._prime_text_embedding_control_cache(
            train_loader, val_loader, test_loader)

        self.logger.info(
            f"[Data] train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
        )
        num_channels = getattr(train_loader, "num_channels", None)

        self.cfg.affcon.affflow_text_concat = self.args.affflow_text_concat
        self.cfg.affcon.affflow_text_mode = self.args.affflow_text_mode
        self.cfg.affcon.affflow_prior_type = self.args.affflow_prior_type
        self.cfg.affcon.affflow_adapter = getattr(self.args, "affflow_adapter", False)
        self.cfg.affcon.affflow_dummy_token = getattr(self.args, "affflow_dummy_token", False)
        self.cfg.affcon.affflow_dummy_init = getattr(self.args, "affflow_dummy_init", "zero")
        self.cfg.affcon.text_gate = getattr(self.args, "text_gate", False)

        horizon = model_horizon or self.data_cfg.horizon
        model = self.build_model(horizon=horizon, num_channels=num_channels)
        lora_warm_start = (
            profile.lora_warm_start and bool(getattr(self.args, 'use_lora', False))
        )
        if lora_warm_start:
            frozen_counts = self._set_generator_trainable(model, False)
            self.logger.info(
                "[LoRAWarmStart] Epoch 0: LoRA-only, generator frozen "
                f"(frozen={frozen_counts})"
            )

        optimizer, trainable_params, optimizer_metadata = build_v5_optimizer(
            model,
            profile,
        )
        num_trainable = optimizer_metadata["total_trainable_parameters"]
        self.logger.info(
            f"[Train] Number of trainable parameters: {num_trainable / 1e6:.3f}M"
        )
        for group in optimizer_metadata["parameter_groups"]:
            self.logger.info(
                "[OptGroup] "
                f"name={group['name']} params={group['num_parameters']} "
                f"lr={group['lr']} wd={group['weight_decay']} betas={tuple(group['betas'])} "
                f"examples={group['representative_names']}"
            )
        for group_name in optimizer_metadata["absent_parameter_groups"]:
            self.logger.info(f"[OptGroup] absent={group_name}")
        criterion = nn.MSELoss()

        probe_steps = 4
        batch_size = initial_micro_batch
        grad_accum = batch_plan.grad_accum_steps
        oom_fallback_history: List[Dict[str, int]] = []
        while batch_size >= min_batch:
            try:
                num_probed = 0
                for probe_idx, probe_batch in enumerate(train_loader):
                    if probe_idx >= probe_steps:
                        break
                    num_probed += 1
                    optimizer.zero_grad(set_to_none=True)
                    self._train_step(
                        model,
                        probe_batch,
                        criterion,
                        trainable_params,
                        accum_scale=float(grad_accum),
                    )
                    optimizer.zero_grad(set_to_none=True)
                if num_probed == 0:
                    raise RuntimeError("Empty train loader during batch-size probe.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not batch_plan.auto_batch_fallback_enabled:
                    raise
                next_batch = next_fallback_micro_batch(batch_size, effective_batch)
                if next_batch is None or next_batch < min_batch:
                    raise RuntimeError(
                        f"OOM at minimum batch_size={min_batch}. "
                        "Use a smaller model or shorter sequences."
                    )
                self.logger.warning(
                    f"[Batch] OOM at batch_size={batch_size}, "
                    f"retrying with batch_size={next_batch} grad_accum={effective_batch // next_batch}"
                )
                oom_fallback_history.append(
                    {
                        "from_micro_batch_size": batch_size,
                        "to_micro_batch_size": next_batch,
                    }
                )
                batch_size = next_batch
                grad_accum = effective_batch // batch_size
                self.args.batch_size = batch_size
                train_loader, val_loader, test_loader = self.build_dataloader(
                    domain=domain,
                    domain_overrides=domain_overrides,
                )
                self._prime_text_embedding_control_cache(
                    train_loader, val_loader, test_loader)
        batch_metadata = make_batch_metadata(
            plan=batch_plan,
            micro_batch_size=batch_size,
            grad_accum_steps=grad_accum,
            oom_fallback_history=oom_fallback_history,
        )
        self.logger.info(
            f"[Batch] effective_batch={effective_batch}, micro_batch={batch_size}, "
            f"grad_accum={grad_accum}, auto_fallback={batch_plan.auto_batch_fallback_enabled}, "
            f"probe_batches={min(probe_steps, len(train_loader))}, "
            f"oom_fallbacks={oom_fallback_history}"
        )

        resolved_lookback, resolved_horizon = self.resolve_run_window(
            domain,
            domain_overrides=domain_overrides,
            fallback_horizon=model_horizon or self.data_cfg.horizon,
        )
        synthetic_mix_loader = None
        if (
            self.run_mode == "train"
            and self.active_dataset != "synthetic"
            and self._synthetic_mix_ratio_for_epoch(0) > 0.0
        ):
            synthetic_mix_loader = self._build_synthetic_mix_loader(
                lookback=resolved_lookback,
                horizon=resolved_horizon,
                batch_size=batch_size,
            )
            self._prime_text_embedding_control_cache(synthetic_mix_loader)
            mix_counts = [
                self._expected_synthetic_batches(
                    len(train_loader),
                    self._synthetic_mix_ratio_for_epoch(epoch),
                )
                for epoch in range(self.args.num_epochs)
            ]
            self.logger.info(
                "[SyntheticMix] enabled "
                f"r0={getattr(self.cfg.affcon, 'synthetic_mix_ratio', 0.0)} "
                f"decay={getattr(self.cfg.affcon, 'synthetic_mix_decay', 0.5)} "
                f"threshold={getattr(self.cfg.affcon, 'synthetic_mix_min_ratio', 1e-3)} "
                f"lookback={resolved_lookback} horizon={resolved_horizon} "
                f"synthetic_batches_by_epoch={mix_counts}"
            )

        expected_micro_steps = len(train_loader) * self.args.num_epochs
        if synthetic_mix_loader is not None:
            expected_micro_steps += sum(
                self._expected_synthetic_batches(
                    len(train_loader),
                    self._synthetic_mix_ratio_for_epoch(epoch),
                )
                for epoch in range(self.args.num_epochs)
            )
        total_steps = math.ceil(expected_micro_steps / grad_accum)
        scheduler, scheduler_metadata = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=profile.warmup_ratio,
        )
        self.logger.info(
            f"[Scheduler] type={scheduler_metadata['type']} "
            f"warmup_steps={scheduler_metadata['warmup_steps']} "
            f"total_steps={scheduler_metadata['total_steps']} initial_lrs={current_lrs(optimizer)}"
        )
        self.args.batch_size = effective_batch

        use_simple_text_collate = (
            not self.use_patch_text_embeds
            and self.model_choice != "sundial"
            and self.active_dataset not in {"time_imm", "time_mmd", "cgtsf"}
        )
        if use_simple_text_collate:
            train_loader.collate_fn = collate_with_text
            val_loader.collate_fn = collate_with_text
            test_loader.collate_fn = collate_with_text

        global_step = 0
        best_val_mse = float("inf")
        best_metrics = None
        best_state_dict = None
        patience_counter = 0

        if getattr(self.args, "stage_a_lora_checkpoint", None) is not None:
            val_metrics0 = self.evaluate(model, val_loader, criterion)
            test_metrics0 = self.evaluate(model, test_loader, criterion)
            best_val_mse = val_metrics0["mse"]
            best_metrics = {"val": val_metrics0, "test": test_metrics0, "epoch": 0}
            best_state_dict = {k: v.detach().cpu()
                               for k, v in model.state_dict().items()}
            self.logger.info(
                f"[Epoch0Floor] Pre-training eval registered as initial best | "
                f"Val MSE: {val_metrics0['mse']:.6f} | Test MSE: {test_metrics0['mse']:.6f} | "
                f"Test MAE: {test_metrics0['mae']:.6f}"
            )

        forecast_csv = self.forecast_dir() / f"{self.run_id}_forecasts.csv"
        manifest_path = self.write_run_manifest(
            status="running",
            domain=domain,
            lookback=resolved_lookback,
            horizon=resolved_horizon,
            model=model,
            forecast_csv=forecast_csv,
            lookback_multiplier=lookback_multiplier,
            extra={
                "training_env": {
                    "train_profile": profile.name,
                    "batch": batch_metadata,
                    "optimizer": optimizer_metadata,
                    "scheduler": {
                        **scheduler_metadata,
                        "current_lr": current_lrs(optimizer),
                    },
                }
            },
        )
        self.logger.info(f"[Manifest] Saved run manifest to {manifest_path}")

        synthetic_iter = iter(synthetic_mix_loader) if synthetic_mix_loader is not None else None

        def next_synthetic_batch():
            nonlocal synthetic_iter
            if synthetic_mix_loader is None:
                return None
            try:
                return next(synthetic_iter)
            except StopIteration:
                synthetic_iter = iter(synthetic_mix_loader)
                return next(synthetic_iter)

        for epoch in range(self.args.num_epochs):
            if lora_warm_start:
                if epoch == 0:
                    self.logger.info("[LoRAWarmStart] Epoch 0: LoRA-only, generator frozen")
                    self._log_trainable_parameter_summary(model, "epoch 0 start")
                elif epoch == 1:
                    optimizer.zero_grad(set_to_none=True)
                    unfrozen_counts = self._set_generator_trainable(model, True)
                    self.logger.info(
                        "[LoRAWarmStart] Epoch 1+: LoRA and generator jointly "
                        f"(unfrozen={unfrozen_counts})"
                    )
                    optimizer, trainable_params, optimizer_metadata = build_v5_optimizer(
                        model,
                        profile,
                    )
                    remaining_micro_steps = len(train_loader) * (self.args.num_epochs - epoch)
                    if synthetic_mix_loader is not None:
                        remaining_micro_steps += sum(
                            self._expected_synthetic_batches(
                                len(train_loader),
                                self._synthetic_mix_ratio_for_epoch(e),
                            )
                            for e in range(epoch, self.args.num_epochs)
                        )
                    scheduler, scheduler_metadata = build_warmup_cosine_scheduler(
                        optimizer,
                        total_steps=math.ceil(remaining_micro_steps / grad_accum),
                        warmup_ratio=profile.warmup_ratio,
                    )
                    for group in optimizer_metadata["parameter_groups"]:
                        self.logger.info(
                            "[OptGroup] "
                            f"name={group['name']} params={group['num_parameters']} "
                            f"lr={group['lr']} wd={group['weight_decay']} "
                            f"betas={tuple(group['betas'])} "
                            f"examples={group['representative_names']}"
                        )
                    self._log_trainable_parameter_summary(model, "epoch 1+ start")

            model.train()
            running_loss = 0.0
            running_affflow = 0.0
            n_micro = 0
            affflow_count = 0
            accum_count = 0
            synthetic_batches_epoch = 0
            mix_ratio = self._synthetic_mix_ratio_for_epoch(epoch)
            mix_accumulator = 0.0

            optimizer.zero_grad(set_to_none=True)

            def process_micro_batch(batch, source: str) -> None:
                nonlocal accum_count, global_step, running_loss, running_affflow
                nonlocal n_micro, affflow_count
                step_result = self._train_step(
                    model, batch, criterion, trainable_params,
                    accum_scale=float(grad_accum),
                )
                if step_result is None:
                    return

                accum_count += 1
                running_loss += step_result["loss"]
                n_micro += 1
                if step_result["affflow_loss"] is not None:
                    running_affflow += step_result["affflow_loss"]
                    affflow_count += 1

                if accum_count == grad_accum:
                    if self.cfg.train.max_grad_norm is not None and self.cfg.train.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, self.cfg.train.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_step += 1

                    if global_step % self.cfg.train.log_every == 0:
                        avg_loss = running_loss / max(n_micro, 1)
                        msg = (
                            f"[Train] Epoch {epoch+1} Step {global_step} | "
                            f"Loss: {avg_loss:.6f} | LR: {current_lrs(optimizer)}"
                        )
                        if source == "synthetic":
                            msg += " | last_batch=synthetic"
                        if affflow_count > 0:
                            avg_affflow = running_affflow / affflow_count
                            msg += f" | AffFlow: {avg_affflow:.6f}"
                        self.logger.info(msg)
                        running_loss = 0.0
                        running_affflow = 0.0
                        n_micro = 0
                        affflow_count = 0

            for batch_idx, batch in enumerate(train_loader):
                process_micro_batch(batch, source="real")
                if synthetic_mix_loader is not None and mix_ratio > 0.0:
                    mix_accumulator += mix_ratio
                    while mix_accumulator >= 1.0:
                        synth_batch = next_synthetic_batch()
                        if synth_batch is None:
                            break
                        process_micro_batch(synth_batch, source="synthetic")
                        synthetic_batches_epoch += 1
                        mix_accumulator -= 1.0

            if accum_count > 0:
                rescale = grad_accum / accum_count
                for p in trainable_params:
                    if p.grad is not None:
                        p.grad.mul_(rescale)
                if self.cfg.train.max_grad_norm is not None and self.cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0
            if synthetic_mix_loader is not None:
                self.logger.info(
                    f"[SyntheticMix] Epoch {epoch+1}: ratio={mix_ratio:.6f} "
                    f"synthetic_batches={synthetic_batches_epoch} "
                    f"real_batches={len(train_loader)}"
                )

            val_metrics = self.evaluate(model, val_loader, criterion)
            test_metrics = self.evaluate(model, test_loader, criterion)
            self.logger.info(
                f"[Val]   Epoch {epoch+1} | "
                f"MSE: {val_metrics['mse']:.6f} | MAE: {val_metrics['mae']:.6f} | "
                f"RMSE: {val_metrics['rmse']:.6f} | MAPE: {val_metrics['mape']:.6f}"
            )
            self.logger.info(f"[LR] Epoch {epoch+1} | {current_lrs(optimizer)}")
            gate_param = getattr(model, "text_gate", None)
            if gate_param is not None:
                self.logger.info(
                    f"[TextGate] Epoch {epoch+1} | b={float(gate_param.detach()):+.4f}"
                )
            self.logger.info(
                f"[Test]  Epoch {epoch+1} | "
                f"MSE: {test_metrics['mse']:.6f} | MAE: {test_metrics['mae']:.6f} | "
                f"RMSE: {test_metrics['rmse']:.6f} | MAPE: {test_metrics['mape']:.6f}"
            )

            min_delta_pct = float(getattr(self.args, "early_stop_min_delta_pct", 0.0) or 0.0)
            improvement_bar = best_val_mse * (1.0 - min_delta_pct / 100.0)
            if val_metrics["mse"] < improvement_bar:
                best_val_mse = val_metrics["mse"]
                best_metrics = {"val": val_metrics,
                                "test": test_metrics, "epoch": epoch + 1}
                patience_counter = 0
                best_state_dict = {k: v.detach().cpu()
                                   for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info(
                        f"[EarlyStop] No val improvement for {patience_counter} epochs, stopping."
                    )
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        if best_metrics is not None:
            self._augment_with_crps(
                model, train_loader, test_loader,
                best_metrics.get("test"), domain=domain,
            )

        if best_metrics is not None:
            self.logger.info(
                f"[Summary] Best @ epoch {best_metrics['epoch']}: "
                f"Val MSE {best_metrics['val']['mse']:.6f}, "
                f"Test MSE {best_metrics['test']['mse']:.6f}, "
                f"Test MAE {best_metrics['test']['mae']:.6f}, "
                f"Test RMSE {best_metrics['test']['rmse']:.6f}, "
                f"Test MAPE {best_metrics['test']['mape']:.6f}"
            )

        checkpoint_path = self._save_conditioning_checkpoint(model)
        if checkpoint_path is not None:
            self.logger.info(f"[Ckpt] Saved conditioning module checkpoint to {checkpoint_path}")

        rows = self.save_forecasts(
            model, test_loader, forecast_csv, domain=domain, run_id=self.run_id)
        self.logger.info(
            f"[Forecast] Saved {rows} forecasts to {forecast_csv}")
        manifest_path = self.write_run_manifest(
            status="completed",
            domain=domain,
            lookback=resolved_lookback,
            horizon=resolved_horizon,
            model=model,
            forecast_csv=forecast_csv,
            checkpoint_path=checkpoint_path,
            lookback_multiplier=lookback_multiplier,
            val_metrics=best_metrics.get("val") if best_metrics else None,
            test_metrics=best_metrics.get("test") if best_metrics else None,
            rows_written=rows,
            extra={
                "training_env": {
                    "train_profile": profile.name,
                    "batch": batch_metadata,
                    "optimizer": optimizer_metadata,
                    "scheduler": {
                        **scheduler_metadata,
                        "current_lr": current_lrs(optimizer),
                        "steps_completed": global_step,
                    },
                    "early_stopping": {
                        "patience": self.args.patience,
                        "patience_counter": patience_counter,
                        "best_epoch": best_metrics.get("epoch") if best_metrics else None,
                    },
                }
            },
        )
        self.logger.info(f"[Manifest] Saved run manifest to {manifest_path}")

        self.args.batch_size = effective_batch
        return model, best_metrics or {}

    def _save_conditioning_checkpoint(self, model: nn.Module) -> Optional[Path]:
        module_name = None
        save_module = None

        if hasattr(model, 'affflow'):
            module_name = getattr(
                self.cfg.affcon,
                "generator_type",
                self.data_mode,
            )
            save_module = model.affflow

        if module_name is None or self.run_id is None:
            return None

        ckpt_dir = self.checkpoint_dir()
        if module_name == "lora_ts_only":
            ckpt_path = ckpt_dir / f"{module_name}_{self.model_choice}_{self.run_id}.pt"
            base_state = {}
        else:
            ckpt_path = ckpt_dir / f"{module_name}_phase1_{self.model_choice}_{self.run_id}.pt"
            if save_module is None:
                return None
            base_state = save_module.state_dict()
        lora_state = None
        lora_backbone = getattr(model, "lora_backbone", None)
        if getattr(model, "use_lora", False) and lora_backbone is not None and has_lora_parameters(lora_backbone):
            lora_state = extract_lora_state_dict(lora_backbone)
        payload = build_checkpoint_payload(
            base_state,
            run_id=self.run_id,
            manifest_path=self.manifest_path(self.run_id),
            args=self.args,
            cfg=self.cfg,
            metadata={
                "module_name": module_name,
                "tsfm_model": self.model_choice,
                "data_mode": self.data_mode,
                "run_mode": self.run_mode,
                "use_lora": bool(getattr(model, "use_lora", False)),
                "lora_backbone_name": getattr(model, "lora_backbone_name", None),
            },
        )
        if lora_state:
            payload["lora_state_dict"] = lora_state
        torch.save(payload, ckpt_path)
        return ckpt_path

    def _train_step(
        self,
        model: nn.Module,
        batch,
        criterion: nn.Module,
        trainable_params: List[torch.nn.Parameter],
        accum_scale: float = 1.0,
    ) -> Optional[dict]:
        if isinstance(batch, dict):
            batch = self._move_batch(batch)
            y = batch.get("data_to_predict")

            outputs = model(batch)

            loss = outputs.get("loss")
            mask_y = outputs.get("mask_forecast_target")

            if loss is None:
                if not torch.is_tensor(mask_y):
                    mask_y = batch.get("mask_forecast_target")
                if not torch.is_tensor(mask_y):
                    mask_y = batch.get("mask_predicted_data")
                if torch.is_tensor(mask_y):
                    mask_y = mask_y.to(dtype=torch.float32)
                    denom = mask_y.sum().clamp_min(1.0)
                    diff = (outputs.get("y_hat") - y) * mask_y
                    loss = (diff * diff).sum() / denom
                else:
                    loss = criterion(outputs.get("y_hat"), y)

            (loss / accum_scale).backward()

            bl = outputs.get("affflow_loss")
            return {
                "loss": loss.item(),
                "affflow_loss": bl.item() if torch.is_tensor(bl) else None,
            }

        return self._train_step_tuple_batch(
            model, batch, criterion, trainable_params, accum_scale=accum_scale
        )

    def _train_step_tuple_batch(
        self,
        model: nn.Module,
        batch,
        criterion: nn.Module,
        trainable_params: List[torch.nn.Parameter],
        accum_scale: float = 1.0,
    ) -> Optional[dict]:
        if self.use_patch_text_embeds:
            x, y, patch_text_emb = batch
            x = x.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            patch_text_emb = patch_text_emb.to(
                device=self.device, dtype=torch.float32)
        elif isinstance(batch, (list, tuple)) and len(batch) == 5:
            x, y, mask_hist, mask_fut, texts_per_patch = batch
            x = x.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            mask_hist = mask_hist.to(device=self.device, dtype=torch.float32)
            mask_fut = mask_fut.to(device=self.device, dtype=torch.float32)
        else:
            x, y, texts = batch
            x = x.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)

        if self.use_patch_text_embeds:
            z = model({"observed_data": x, "patch_text_emb": patch_text_emb})[
                "z"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 5:
            B, L, C = x.shape
            z = model({"observed_data": x, "texts_per_patch": texts_per_patch})[
                "z"]
            H = y.shape[1]
            target_len = model.flow_loss.in_channels
            y_flat = y.permute(0, 2, 1).reshape(B * C, H)
            mask_fut_flat = mask_fut.permute(0, 2, 1).reshape(B * C, H)
            target = torch.zeros(
                B * C, target_len, device=y.device, dtype=y.dtype)
            target[:, :H] = y_flat
            mask_y = torch.zeros_like(target)
            mask_y[:, :H] = mask_fut_flat
            if mask_y.sum() == 0:
                return None
            loss = model.flow_loss(target, z, mask=None, mask_y=mask_y)["z"]
        else:
            if self.model_choice in ("timemoe", "timesfm"):
                outputs = model({"observed_data": x, "data_to_predict": y})
                loss = outputs.get("loss")
                y_hat = outputs.get("y_hat")
            else:
                z = model({"observed_data": x, "texts": texts})["z"]

        if not (isinstance(batch, (list, tuple)) and len(batch) == 5):
            if self.model_choice in ("timemoe", "timesfm"):
                if loss is None:
                    diff = y_hat - y
                    loss = (diff * diff).mean()
            else:
                Bsz = y.shape[0]
                H = y.shape[1]
                target_len = model.flow_loss.in_channels

                target = torch.zeros(
                    Bsz, target_len, device=y.device, dtype=y.dtype)
                target[:, :H] = y

                mask_y = torch.zeros(
                    Bsz, target_len, device=y.device, dtype=y.dtype)
                mask_y[:, :H] = 1.0

                loss = model.flow_loss(target, z, mask=None, mask_y=mask_y)

        (loss / accum_scale).backward()

        return {"loss": loss.item(), "affflow_loss": None}

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> dict:
        from experiments.data_utils import aggregate_macro

        model.eval()
        self._prime_text_embedding_control_cache(data_loader)
        total_loss = 0.0
        n_batches = 0
        per_sample = {"per_sample_mse": [], "per_sample_mae": [],
                      "per_sample_mape": [], "per_sample_mspe": []}

        for _, batch in enumerate(data_loader):
            metrics = self._evaluate_batch(model, batch, criterion)
            if metrics is None:
                continue
            total_loss += metrics["loss"]
            n_batches += 1
            for k in per_sample:
                per_sample[k].extend(metrics.get(k, []))

        if n_batches == 0 or len(per_sample["per_sample_mse"]) == 0:
            if self.logger:
                self.logger.warning(
                    "Evaluation had zero batches or samples; returning inf metrics."
                )
            return {
                "loss": float("inf"),
                "mse": float("inf"),
                "mae": float("inf"),
                "rmse": float("inf"),
                "mape": float("inf"),
                "mspe": float("inf"),
            }

        agg = aggregate_macro(per_sample)
        return {
            "loss": total_loss / n_batches,
            **agg,
        }

    @torch.no_grad()
    def evaluate_crps(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_samples: int = 100,
        sigma_y: Optional[float] = None,
        dump_path: Optional[str] = None,
    ) -> Optional[dict]:
        from experiments.crps import (
            crps_from_samples, crps_from_quantiles_pwl,
            pit_from_samples, pit_from_quantiles_pwl,
        )
        from experiments.data_utils import irregular_raw_alignment

        is_quantile = hasattr(model, "_quantile_forecast")
        is_sample = hasattr(model, "flow_loss")
        if not (is_quantile or is_sample):
            return None

        model.eval()
        self._prime_text_embedding_control_cache(data_loader)
        crps_chunks = []
        pit_chunks = []
        example_chunks: List[dict] = []

        for batch in data_loader:
            if not isinstance(batch, dict):
                continue
            batch = self._move_batch(batch)
            y = batch.get("data_to_predict")
            if not torch.is_tensor(y):
                continue
            y = y.to(dtype=torch.float32)
            batch["data_to_predict"] = y
            target_idx = batch.get("target_idx", 0)
            target_idx = int(target_idx.item()) if torch.is_tensor(target_idx) else int(target_idx)
            H = y.shape[1]

            taus = None
            if is_quantile:
                dist, taus = model._quantile_forecast(batch, H)
            else:
                outputs = model(batch, num_samples=num_samples)
                dist = outputs["samples"]

            if dump_path is not None:
                collected = self._collect_forecast_examples(
                    batch, dist, taus, target_idx, is_quantile)
                if collected is not None:
                    example_chunks.append(collected)

            raw_vals = batch.get("raw_data_to_predict")
            raw_mask = batch.get("raw_mask_predicted_data")
            raw_tp = batch.get("raw_tp_to_predict")
            grid_tp_raw = batch.get("tp_to_predict_raw")
            use_raw = (
                torch.is_tensor(raw_vals) and torch.is_tensor(raw_mask)
                and torch.is_tensor(raw_tp) and torch.is_tensor(grid_tp_raw)
                and raw_vals.numel() > 0
            )

            if use_raw:
                for b in range(y.shape[0]):
                    aligned = irregular_raw_alignment(
                        raw_vals, raw_mask, raw_tp, grid_tp_raw, target_idx, b)
                    if aligned is None:
                        continue
                    vals_raw, mask_raw, nn_idx = aligned
                    yb = vals_raw.unsqueeze(0)
                    mb = mask_raw.unsqueeze(0)
                    if is_quantile:
                        db = dist[b][nn_idx].unsqueeze(0)
                        crps_b = crps_from_quantiles_pwl(db, taus, yb, mb)
                        pit_b = pit_from_quantiles_pwl(db, taus, yb, mb)
                    else:
                        db = dist[b][:, nn_idx].unsqueeze(0)
                        crps_b = crps_from_samples(db, yb, mb)
                        pit_b = pit_from_samples(db, yb, mb)
                    crps_chunks.append(crps_b.detach().cpu())
                    pit_chunks.append(pit_b.detach().cpu())
            else:
                y_2d = y[..., target_idx] if y.dim() == 3 else y
                mask = batch.get("mask_predicted_data")
                mask_2d = None
                if torch.is_tensor(mask):
                    mask = mask.to(dtype=torch.float32)
                    mask_2d = mask[..., target_idx] if mask.dim() == 3 else mask
                if is_quantile:
                    crps = crps_from_quantiles_pwl(dist, taus, y_2d, mask_2d)
                    pit = pit_from_quantiles_pwl(dist, taus, y_2d, mask_2d)
                else:
                    crps = crps_from_samples(dist, y_2d, mask_2d)
                    pit = pit_from_samples(dist, y_2d, mask_2d)
                crps_chunks.append(crps.detach().cpu())
                pit_chunks.append(pit.detach().cpu())

        if not crps_chunks:
            return None
        crps_all = torch.cat(crps_chunks)
        crps_all = crps_all[~torch.isnan(crps_all)]
        if crps_all.numel() == 0:
            return None
        crps_mean = float(crps_all.mean().item())

        pit_all = torch.cat(pit_chunks) if pit_chunks else torch.empty(0)
        levels = torch.linspace(0.1, 0.9, 9)
        if pit_all.numel() > 0:
            lo = (1.0 - levels) / 2.0
            hi = (1.0 + levels) / 2.0
            inside = ((pit_all.unsqueeze(1) >= lo) & (pit_all.unsqueeze(1) <= hi))
            emp = inside.to(dtype=torch.float32).mean(dim=0)
            pit_cal_err = float((emp - levels).abs().mean().item())
        else:
            pit_cal_err = float("nan")

        denom = sigma_y if (sigma_y is not None and sigma_y > 1e-8) else 1.0
        result = {
            "crps": crps_mean,
            "crps_norm": crps_mean / denom,
            "pit_cal_err": pit_cal_err,
            "n_scored": int(crps_all.numel()),
        }

        if dump_path is not None and pit_all.numel() > 0:
            import numpy as np
            Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
            arrays = dict(
                pit=pit_all.numpy(),
                crps_per_sample=crps_all.numpy(),
                levels=levels.numpy(),
                sigma_y=np.float32(denom),
                crps=np.float32(crps_mean),
                crps_norm=np.float32(crps_mean / denom),
            )
            if example_chunks:
                for field in example_chunks[0]:
                    arrays[f"ex_{field}"] = np.concatenate(
                        [c[field] for c in example_chunks], axis=0)
            np.savez(dump_path, **arrays)
            result["dump_path"] = str(dump_path)
        return result

    @torch.no_grad()
    def _collect_forecast_examples(
        self, batch, dist, taus, target_idx, is_quantile,
    ) -> Optional[dict]:
        import numpy as np
        obs = batch.get("observed_data")
        y = batch.get("data_to_predict")
        if not (torch.is_tensor(obs) and torch.is_tensor(y)):
            return None
        hist = obs[:, :, target_idx] if obs.dim() == 3 else obs
        truth = y[:, :, target_idx] if y.dim() == 3 else y

        obs_mask = batch.get("observed_mask")
        pred_mask = batch.get("mask_predicted_data")

        def _chan(m):
            if not torch.is_tensor(m):
                return torch.ones_like(hist)
            return m[:, :, target_idx] if m.dim() == 3 else m

        hist_mask = _chan(obs_mask)
        truth_mask = _chan(pred_mask)

        if is_quantile:
            lo_i = int((taus - 0.1).abs().argmin())
            md_i = int((taus - 0.5).abs().argmin())
            hi_i = int((taus - 0.9).abs().argmin())
            lo, median, hi = dist[..., lo_i], dist[..., md_i], dist[..., hi_i]
        else:
            d = dist.to(dtype=torch.float32)
            median = d.median(dim=1).values
            lo = torch.quantile(d, 0.1, dim=1)
            hi = torch.quantile(d, 0.9, dim=1)

        keys = batch.get("sample_keys") or [""] * int(y.shape[0])
        return {
            "history": hist.detach().cpu().numpy(),
            "truth": truth.detach().cpu().numpy(),
            "hist_mask": hist_mask.detach().cpu().numpy(),
            "truth_mask": truth_mask.detach().cpu().numpy(),
            "median": median.detach().cpu().numpy(),
            "lo": lo.detach().cpu().numpy(),
            "hi": hi.detach().cpu().numpy(),
            "key": np.asarray([str(k) for k in keys], dtype=np.str_),
        }

    def _augment_with_crps(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        test_metrics: Optional[dict],
        domain: Optional[str] = None,
    ) -> Optional[dict]:
        if not getattr(self.cfg.affcon, "compute_crps", False):
            return None
        if not (hasattr(model, "_quantile_forecast") or hasattr(model, "flow_loss")):
            return None
        from experiments.crps import compute_target_train_std

        sigma_y = compute_target_train_std(train_loader)
        dump_path = None
        if getattr(self.cfg.affcon, "crps_dump", False) and self.metrics_csv is not None:
            tag = str(getattr(self, "run_id", None) or "run")
            if domain:
                tag = f"{tag}_{domain}"
            dump_path = str(Path(self.metrics_csv).parent / "crps_dumps" / f"{tag}.npz")
        num_samples = int(getattr(self.cfg.affcon, "crps_num_samples", 100))
        crps_metrics = self.evaluate_crps(
            model, test_loader,
            num_samples=num_samples, sigma_y=sigma_y, dump_path=dump_path,
        )
        if crps_metrics is None:
            return None
        if isinstance(test_metrics, dict):
            test_metrics["crps"] = crps_metrics["crps"]
            test_metrics["crps_norm"] = crps_metrics["crps_norm"]
            test_metrics["pit_cal_err"] = crps_metrics["pit_cal_err"]
        if self.logger:
            self.logger.info(
                f"[CRPS] crps={crps_metrics['crps']:.6f} "
                f"crps_norm={crps_metrics['crps_norm']:.6f} "
                f"pit_cal_err={crps_metrics['pit_cal_err']:.4f} "
                f"(sigma_y={sigma_y:.4f}, n={crps_metrics['n_scored']})"
            )
        return crps_metrics

    def _evaluate_batch(
        self,
        model: nn.Module,
        batch,
        criterion: nn.Module,
    ) -> Optional[dict]:
        if isinstance(batch, dict):
            return self._evaluate_dict_batch(model, batch, criterion)
        return self._evaluate_tuple_batch(model, batch, criterion)

    def _evaluate_dict_batch(
        self,
        model: nn.Module,
        batch: dict,
        criterion: nn.Module,
    ) -> Optional[dict]:
        from experiments.data_utils import evaluate_batch_persample

        batch = self._move_batch(batch)
        y = batch.get("data_to_predict")
        y = y.to(dtype=torch.float32)
        batch["data_to_predict"] = y

        if torch.is_tensor(batch.get("mask_predicted_data")):
            batch["mask_predicted_data"] = batch["mask_predicted_data"].to(
                dtype=torch.float32)

        with torch.no_grad():
            outputs = model(batch)

        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            y_hat = outputs.get("logits") or outputs.get(
                "predictions") or outputs.get("y_hat")
        else:
            loss = None
            y_hat = outputs

        mask_y = outputs.get("mask_forecast_target") if isinstance(
            outputs, dict) else None
        if not torch.is_tensor(mask_y):
            mask_y = batch.get("mask_forecast_target")
        if not torch.is_tensor(mask_y):
            mask_y = batch.get("mask_predicted_data")

        raw_vals = batch.get("raw_data_to_predict")
        raw_mask = batch.get("raw_mask_predicted_data")
        raw_tp = batch.get("raw_tp_to_predict")
        grid_tp_raw = batch.get("tp_to_predict_raw")
        target_idx = (
            outputs.get("target_idx")
            if isinstance(outputs, dict) and outputs.get("target_idx") is not None
            else batch.get("target_idx", 0)
        )
        if torch.is_tensor(target_idx):
            target_idx = int(target_idx.item())
        else:
            target_idx = int(target_idx)

        y_hat_2d = y_hat[..., target_idx] if y_hat.dim() == 3 else y_hat
        y_2d = y[..., target_idx] if y.dim() == 3 else y
        mask_2d = None
        if torch.is_tensor(mask_y):
            mask_2d = mask_y[..., target_idx] if mask_y.dim() == 3 else mask_y

        if loss is None:
            if torch.is_tensor(mask_2d):
                m = mask_2d.to(dtype=torch.float32)
                ms = m.sum().item()
                if ms > 0:
                    loss = ((y_hat_2d - y_2d) * m).pow(2).sum() / ms
                else:
                    loss = criterion(y_hat_2d, y_2d)
            else:
                loss = criterion(y_hat_2d, y_2d)

        per_sample = evaluate_batch_persample(
            y_hat=y_hat_2d,
            y=y_2d,
            mask=mask_2d,
            raw_vals=raw_vals,
            raw_mask=raw_mask,
            raw_tp=raw_tp,
            grid_tp_raw=grid_tp_raw,
            target_idx=target_idx,
        )
        if len(per_sample["per_sample_mse"]) == 0:
            return None
        return {
            "loss": float(loss.item()) if torch.is_tensor(loss) else float(loss),
            **per_sample,
        }

    def _evaluate_tuple_batch(
        self,
        model: nn.Module,
        batch,
        criterion: nn.Module,
    ) -> Optional[dict]:
        from experiments.data_utils import evaluate_batch_persample

        mask_fut = None
        if isinstance(batch, (list, tuple)) and len(batch) == 5:
            x, y, mask_hist, mask_fut, texts_per_patch = batch
            x = x.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            mask_fut = mask_fut.to(device=self.device, dtype=torch.float32)
            H = y.shape[1]
            preds = model.predict(
                {"observed_data": x, "texts_per_patch": texts_per_patch},
                horizon=H,
            )
            y_hat = preds["y_hat"]
            ms = mask_fut.sum().item()
            if ms == 0:
                return None
            loss = ((y_hat - y) * mask_fut).pow(2).sum() / ms
        elif isinstance(batch, (list, tuple)) and len(batch) == 3 and not isinstance(batch[2], str):
            x, y, emb = batch
            x = x.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            emb = emb.to(device=self.device, dtype=torch.float32)
            with torch.no_grad():
                H = y.shape[1]
                preds = model.predict(
                    {"observed_data": x, "patch_text_emb": emb},
                    horizon=H,
                )
                y_hat = preds["y_hat"]
            loss = criterion(y_hat, y)
        else:
            x, y, texts = batch
            x = x.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            with torch.no_grad():
                H = y.shape[1]
                if hasattr(model, "timemoe") or hasattr(model, "timesfm_module"):
                    preds = model.predict(
                        {"observed_data": x, "data_to_predict": y},
                        horizon=H,
                    )
                else:
                    preds = model.predict(
                        {"observed_data": x, "texts": texts},
                        horizon=H,
                    )
                y_hat = preds["y_hat"]
            loss = criterion(y_hat, y)

        y_hat_2d = y_hat[..., 0] if y_hat.dim() == 3 else y_hat
        y_2d = y[..., 0] if y.dim() == 3 else y
        mask_2d = None
        if torch.is_tensor(mask_fut):
            mask_2d = mask_fut[..., 0] if mask_fut.dim() == 3 else mask_fut

        per_sample = evaluate_batch_persample(
            y_hat=y_hat_2d, y=y_2d, mask=mask_2d,
        )
        if len(per_sample["per_sample_mse"]) == 0:
            return None
        return {
            "loss": float(loss.item()) if torch.is_tensor(loss) else float(loss),
            **per_sample,
        }

    @abstractmethod
    def run(self) -> None:
        pass
