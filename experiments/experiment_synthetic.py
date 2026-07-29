
from __future__ import annotations

import logging
import math
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.model_sundial import SundialForecaster, load_backbones
from models.model_timemoe import TimeMoEForecaster, load_timemoe_backbone
from models.model_timesfm import TimesFMForecaster, load_timesfm_backbone
from experiments.experiment_base import (
    DATA_MODE,
    _jsonify,
    append_campaign_registry_row,
    PROJ_PATH,
    build_checkpoint_payload,
    build_model_manifest,
    extract_checkpoint_state_dict,
    get_git_metadata,
    log_metrics,
    move_batch_to_device,
    scheduler_context_from_env,
    setup_logger,
    snapshot_helper_scripts,
    write_model_structure,
    write_manifest,
)
from experiments.data_synthetic import build_synthetic_dataloaders
from experiments.training_env import (
    build_v5_optimizer,
    build_warmup_cosine_scheduler,
    current_lrs,
    get_train_profile,
    make_batch_metadata,
    next_fallback_micro_batch,
    resolve_batch_plan,
)


class SyntheticExperiment:

    def __init__(self, cfg, args, *, metrics_csv: Optional[Path] = None):
        self.cfg = cfg
        self.args = args
        self.device = torch.device(args.device)
        self.model_choice = args.tsfm_model
        self.data_mode = DATA_MODE
        self.cfg.affcon.affflow_text_concat = args.affflow_text_concat
        self.cfg.affcon.affflow_text_mode = args.affflow_text_mode
        self.cfg.affcon.affflow_prior_type = args.affflow_prior_type
        self.cfg.affcon.affflow_dummy_token = getattr(args, "affflow_dummy_token", False)
        self.cfg.affcon.affflow_dummy_init = getattr(args, "affflow_dummy_init", "zero")
        self.run_timestamp = getattr(
            args,
            "trace_timestamp",
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        )

        self.data_cfg = cfg.synthetic
        self.phase1_mode = getattr(args, "synthetic_phase1_mode", "full_series")
        self.jsonl_path = self._resolve_jsonl_path()
        self.emb_cache_dir = Path(self.data_cfg.data_path)
        self.metrics_csv = metrics_csv or (PROJ_PATH / "results" / "metrics.csv")
        self.trace_root = Path(getattr(args, "trace_root", "")).expanduser() if getattr(args, "trace_root", None) else (
            PROJ_PATH
            / "results"
            / "run_traces"
            / f"train_{self.model_choice}_synthetic_{self.data_mode}_{self.args.run_stage}_{self.args.run_tag}_{self.run_timestamp}"
        )
        self.trace_root.mkdir(parents=True, exist_ok=True)

        self.logger: Optional[logging.Logger] = None
        self.model = None
        self._snapshot_launch_context()

    def _resolve_jsonl_path(self) -> Path:
        data_root = Path(self.data_cfg.data_path)
        if self.phase1_mode == "full_series":
            jsonl_path = data_root / "synthetic_pairs.jsonl"
        elif self.phase1_mode == "forecast":
            jsonl_path = data_root / "synthetic_pairs_history_forecast_split.jsonl"
        else:
            raise ValueError(
                f"Unknown synthetic phase1 mode '{self.phase1_mode}'. "
                "Expected 'full_series' or 'forecast'."
            )
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Synthetic Phase 1 file not found: {jsonl_path}. "
                f"Generate it first for mode '{self.phase1_mode}'."
            )
        return jsonl_path

    def manifest_path(self, run_id: str) -> Path:
        return self.trace_root / "manifests" / f"{run_id}.json"

    def model_structure_path(self, run_id: str) -> Path:
        return self.trace_root / "manifests" / f"{run_id}.model.txt"

    def _snapshot_launch_context(self) -> None:
        launch_dir = self.trace_root / "launch"
        launch_dir.mkdir(parents=True, exist_ok=True)
        helper_scripts = list(getattr(self.args, "helper_scripts", []) or [])
        copied = snapshot_helper_scripts(helper_scripts, launch_dir / "helper_scripts")
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
        run_id: str,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[Path] = None,
        val_metrics: Optional[dict[str, float]] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        model_structure_path = write_model_structure(
            self.model_structure_path(run_id),
            model,
        )
        manifest = {
            "schema_version": 1,
            "status": status,
            "run_id": run_id,
            "run_timestamp": self.run_timestamp,
            "project_root": str(PROJ_PATH),
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git": get_git_metadata(),
            "run": {
                "run_mode": self.args.run_mode,
                "active_dataset": self.args.active_dataset,
                "campaign_id": getattr(self.args, "campaign_id", None),
                "campaign_spec_path": getattr(self.args, "campaign_spec_path", None),
                "campaign_label": getattr(self.args, "campaign_label", None),
                "campaign_note": getattr(self.args, "campaign_note", None),
                "domain": "synthetic",
                "lookback": None,
                "horizon": None if self.phase1_mode == "forecast" else 256,
                "tsfm_model": self.model_choice,
                "data_mode": self.data_mode,
                "run_stage": self.args.run_stage,
                "train_profile": getattr(self.args, "train_profile", None),
                "device": str(self.device),
                "run_tag": self.args.run_tag,
                "loaded_pretrain_checkpoint": None,
                "synthetic_phase1_mode": self.phase1_mode,
            },
            "args": vars(self.args),
            "resolved_config": self.cfg,
            "dataset_settings": {
                "data_path": self.data_cfg.data_path,
                "jsonl_path": str(self.jsonl_path),
                "train_ratio": self.data_cfg.train_ratio,
                "phase1_mode": self.phase1_mode,
            },
            "model": build_model_manifest(model),
            "metrics": {
                "val": val_metrics,
                "test": None,
            },
            "outputs": {
                "trace_root": str(self.trace_root),
                "log_file": str(self.trace_root / "logs" / f"{run_id}.log"),
                "manifest_path": str(self.manifest_path(run_id)),
                "model_structure_txt": str(model_structure_path) if model_structure_path is not None else None,
                "metrics_csv": str(self.metrics_csv),
                "forecast_csv": None,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "forecast_rows": None,
            },
        }
        if extra:
            manifest["extra"] = extra
        manifest_path = self.manifest_path(run_id)
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
                "run_id": run_id,
                "run_mode": self.args.run_mode,
                "active_dataset": self.args.active_dataset,
                "domain": "synthetic",
                "tsfm_model": self.model_choice,
                "data_mode": self.data_mode,
                "run_stage": self.args.run_stage,
                "train_profile": getattr(self.args, "train_profile", None),
                "run_tag": self.args.run_tag,
                "trace_root": str(self.trace_root),
                "manifest_path": str(manifest_path),
                "git_commit": get_git_metadata().get("commit"),
                **scheduler_ctx,
            },
        )
        return manifest_path


    def _build_model(self, horizon: Optional[int]) -> nn.Module:
        if self.model_choice == "timemoe":
            backbones = load_timemoe_backbone(
                timemoe_model_name=self.cfg.train.timemoe_model_name,
                qwen_model_name=self.cfg.train.qwen_model_name,
                device=self.device,
                use_flash_attn=self.cfg.train.use_flash_attn,
                freeze_text=self.cfg.train.freeze_text,
                load_text_backbone=False,
            )
            model = TimeMoEForecaster(
                backbones=backbones,
                horizon=horizon,
                cfg=self.cfg,
                run_mode="train",
            ).to(self.device)
        elif self.model_choice == "timesfm":
            backbones = load_timesfm_backbone(
                timesfm_model_name=self.cfg.train.timesfm_model_name,
                qwen_model_name=self.cfg.train.qwen_model_name,
                device=self.device,
            )
            model = TimesFMForecaster(
                backbones=backbones,
                horizon=horizon,
                cfg=self.cfg,
                run_mode="train",
            ).to(self.device)
        else:
            backbones = load_backbones(
                sundial_model_name=self.cfg.train.sundial_model_name,
                qwen_model_name=self.cfg.train.qwen_model_name,
                device=self.device,
                use_flash_attn=self.cfg.train.use_flash_attn,
                freeze_sundial=self.cfg.train.freeze_sundial,
                freeze_text=self.cfg.train.freeze_text,
                load_text_backbone=False,
            )
            model = SundialForecaster(
                backbones=backbones,
                horizon=horizon,
                cfg=self.cfg,
                run_mode="train",
            ).to(self.device)

        self.model = model
        return model


    def train_phase1(self) -> Path:
        parts = ["pretrain", self.model_choice, self.data_mode, self.args.run_stage, self.args.run_tag]
        parts.extend([f"synthetic_{self.phase1_mode}", self.run_timestamp])
        run_id = "_".join(parts)
        self.logger = setup_logger(run_id, log_dir=self.trace_root / "logs")
        self.logger.info("[Phase1] Starting synthetic pre-training")

        profile = get_train_profile(self.args.train_profile)
        batch_plan = resolve_batch_plan(self.args)
        effective_batch = batch_plan.effective_batch_size
        initial_micro_batch = batch_plan.micro_batch_size
        min_batch = 1
        self.args.batch_size = initial_micro_batch

        train_loader, val_loader = build_synthetic_dataloaders(
            jsonl_path=self.jsonl_path,
            cache_dir=self.emb_cache_dir,
            batch_size=self.args.batch_size,
            train_ratio=self.data_cfg.train_ratio,
            device=str(self.device),
            phase1_mode=self.phase1_mode,
        )
        self.logger.info(
            f"[Data] train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, d_txt={train_loader.d_txt}"
        )

        model = self._build_model(horizon=None)
        model.train()

        module_name = self.data_mode
        optimizer, trainable_params, optimizer_metadata = build_v5_optimizer(
            model,
            profile,
        )
        num_trainable = optimizer_metadata["total_trainable_parameters"]
        self.logger.info(
            f"[Train] Trainable parameters ({module_name} only): {num_trainable / 1e6:.3f}M"
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

        probe_steps = 4
        batch_size = initial_micro_batch
        grad_accum = batch_plan.grad_accum_steps
        oom_fallback_history: list[dict[str, int]] = []
        while batch_size >= min_batch:
            try:
                num_probed = 0
                for probe_idx, probe_batch in enumerate(train_loader):
                    if probe_idx >= probe_steps:
                        break
                    num_probed += 1
                    probe_batch = move_batch_to_device(probe_batch, self.device)
                    optimizer.zero_grad(set_to_none=True)
                    probe_out = self._forward_phase1(
                        model,
                        probe_batch,
                    )
                    (probe_out["loss"] / grad_accum).backward()
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
                train_loader, val_loader = build_synthetic_dataloaders(
                    jsonl_path=self.jsonl_path,
                    cache_dir=self.emb_cache_dir,
                    batch_size=self.args.batch_size,
                    train_ratio=self.data_cfg.train_ratio,
                    device=str(self.device),
                    phase1_mode=self.phase1_mode,
                )
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

        total_steps = math.ceil(len(train_loader) / grad_accum) * self.args.num_epochs
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

        best_val_mse = float("inf")
        patience_counter = 0
        global_step = 0

        ckpt_dir = PROJ_PATH / "results" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_tag_suffix = f"_{self.args.run_tag}" if self.args.run_tag else ""
        ckpt_path = (
            ckpt_dir
            / f"{module_name}_phase1_{self.model_choice}{run_tag_suffix}_{self.run_timestamp}.pt"
        )
        manifest_path = self.write_run_manifest(
            status="running",
            run_id=run_id,
            model=model,
            checkpoint_path=ckpt_path,
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

        for epoch in range(self.args.num_epochs):
            model.train()
            running_loss = 0.0
            n_micro = 0
            accum_count = 0

            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(train_loader):
                is_last_batch = (batch_idx == len(train_loader) - 1)
                accum_count += 1
                is_update_step = (accum_count == grad_accum) or is_last_batch

                batch = move_batch_to_device(batch, self.device)

                outputs = self._forward_phase1(model, batch)
                loss = outputs["loss"]

                (loss / grad_accum).backward()

                running_loss += loss.item()
                n_micro += 1

                if is_update_step:
                    if accum_count < grad_accum:
                        rescale = grad_accum / accum_count
                        for p in trainable_params:
                            if p.grad is not None:
                                p.grad.mul_(rescale)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.cfg.train.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_step += 1

                    if global_step % self.cfg.train.log_every == 0:
                        avg = running_loss / max(n_micro, 1)
                        self.logger.info(
                            f"[Train] Epoch {epoch+1} Step {global_step} | "
                            f"Loss: {avg:.6f} | LR: {current_lrs(optimizer)}"
                        )
                        running_loss = 0.0
                        n_micro = 0

            val_metrics = self._evaluate(model, val_loader)
            val_mse = val_metrics["mse"]
            val_mae = val_metrics["mae"]
            self.logger.info(
                f"[Val]   Epoch {epoch+1} | MSE: {val_mse:.6f} | MAE: {val_mae:.6f}"
            )
            self.logger.info(f"[LR] Epoch {epoch+1} | {current_lrs(optimizer)}")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                save_module = model.affflow
                payload = build_checkpoint_payload(
                    save_module.state_dict(),
                    run_id=run_id,
                    manifest_path=self.manifest_path(run_id),
                    args=self.args,
                    cfg=self.cfg,
                    metadata={
                        "module_name": module_name,
                        "tsfm_model": self.model_choice,
                        "data_mode": self.data_mode,
                        "run_mode": self.args.run_mode,
                        "best_epoch": epoch + 1,
                        "best_val_mse": val_mse,
                    },
                )
                torch.save(payload, ckpt_path)
                self.logger.info(
                    f"[Ckpt]  Saved best {module_name} (MSE={val_mse:.6f}) → {ckpt_path}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info(
                        f"[EarlyStop] No val improvement for "
                        f"{patience_counter} epochs, stopping."
                    )
                    break

        self.logger.info(
            f"[Phase1] Best val MSE: {best_val_mse:.6f}  "
            f"Checkpoint: {ckpt_path}"
        )
        manifest_path = self.write_run_manifest(
            status="completed",
            run_id=run_id,
            model=model,
            checkpoint_path=ckpt_path,
            val_metrics={"mse": best_val_mse},
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
                    },
                }
            },
        )
        self.logger.info(f"[Manifest] Saved run manifest to {manifest_path}")
        self.args.batch_size = effective_batch
        return ckpt_path


    @torch.no_grad()
    def _evaluate(self, model: nn.Module, data_loader) -> dict[str, float]:
        model.eval()

        mse_sum = 0.0
        mae_sum = 0.0
        count = 0

        for batch in data_loader:
            batch = move_batch_to_device(batch, self.device)
            outputs = self._forward_phase1(model, batch)

            if self.phase1_mode == "full_series":
                pred = outputs["y_hat"]
                gt = batch["gt_series"]
                diff = pred - gt
                sq = diff * diff
                if "lengths" in batch:
                    lengths = batch["lengths"]
                    max_len = gt.shape[1]
                    mask = torch.arange(max_len, device=gt.device).unsqueeze(0) < lengths.unsqueeze(1)
                    mse_sum += (sq * mask).sum().item()
                    mae_sum += (diff.abs() * mask).sum().item()
                    count += mask.sum().item()
                else:
                    mse_sum += sq.sum().item()
                    mae_sum += diff.abs().sum().item()
                    count += gt.numel()
            else:
                target_idx = int(outputs.get("target_idx", 0))
                pred = outputs["y_hat"][:, :, target_idx]
                gt = batch["data_to_predict"][:, :, target_idx]
                mask = batch["mask_predicted_data"][:, :, target_idx]
                diff = (pred - gt) * mask
                mse_sum += (diff * diff).sum().item()
                mae_sum += diff.abs().sum().item()
                count += mask.sum().item()

        n = max(count, 1)
        return {"mse": mse_sum / n, "mae": mae_sum / n}


    def run(self) -> None:
        if self.args.run_mode == "zeroshot":
            self.eval_phase1()
            return
        if self.args.run_mode != "train":
            raise ValueError(
                "Synthetic dataset only supports --run-mode train or zeroshot"
            )
        ckpt_path = self.train_phase1()
        print(f"Phase 1 complete. Checkpoint: {ckpt_path}")

    @torch.no_grad()
    def eval_phase1(self) -> dict[str, float]:
        run_id = "_".join([
            "eval", self.model_choice, self.data_mode, self.args.run_stage,
            self.args.run_tag, f"synthetic_{self.phase1_mode}", self.run_timestamp,
        ])
        self.logger = setup_logger(run_id, log_dir=self.trace_root / "logs")

        ckpt = getattr(self.args, "pretrain_checkpoint", None)
        if ckpt is None:
            raise ValueError(
                "Synthetic zeroshot requires --pretrain-checkpoint "
                "(the Phase-1 generator to evaluate)."
            )
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt_path}")

        nfe = int(getattr(self.cfg.affcon, "affflow_nfe", 50))
        self.logger.info(
            f"[Eval] checkpoint={ckpt_path.name} nfe={nfe} mode={self.phase1_mode}"
        )

        _, val_loader = build_synthetic_dataloaders(
            jsonl_path=self.jsonl_path,
            cache_dir=self.emb_cache_dir,
            batch_size=self.args.batch_size,
            train_ratio=self.data_cfg.train_ratio,
            device=str(self.device),
            phase1_mode=self.phase1_mode,
        )
        self.logger.info(f"[Data] val={len(val_loader.dataset)}, d_txt={val_loader.d_txt}")

        model = self._build_model(horizon=None)
        payload = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        state = extract_checkpoint_state_dict(payload)
        model.affflow.load_state_dict(state)
        model.affflow.requires_grad_(False)
        model.eval()

        metrics = self._evaluate(model, val_loader)
        self.logger.info(
            f"[Eval] nfe={nfe} mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}"
        )

        log_metrics(
            self.metrics_csv,
            run_id=run_id,
            run_mode="zeroshot",
            tsfm_model=self.model_choice,
            active_dataset="synthetic",
            domain=f"synthetic_{self.phase1_mode}",
            lookback=0,
            horizon=0,
            test_metrics=metrics,
            run_stage=self.args.run_stage,
            run_tag=self.args.run_tag,
        )
        self.write_run_manifest(
            status="completed",
            run_id=run_id,
            model=model,
            checkpoint_path=ckpt_path,
            val_metrics={"mse": metrics["mse"], "mae": metrics["mae"]},
            extra={"eval": {"nfe": nfe, "pretrain_checkpoint": str(ckpt_path)}},
        )
        print(f"[Eval] nfe={nfe} test_mse={metrics['mse']:.6f} → {self.metrics_csv}")
        return metrics

    def _forward_phase1(
        self,
        model: nn.Module,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        if self.phase1_mode == "forecast":
            return model(batch)

        return model.forward_text_only(batch)
