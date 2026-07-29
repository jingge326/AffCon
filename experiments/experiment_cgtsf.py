
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.data_cgtsf import (
    build_cgtsf_dataloaders,
    build_cgtsf_dataloaders_timestep,
)
from experiments.experiment_base import (
    ExperimentBase,
    PROJ_PATH,
    format_run_id,

    filter_overrides,
    effective_window,
    inverse_scale_tensor,
    unwrap_base_dataset,
    dataset_indices,
    fetch_sample,
    select_target,
    log_metrics,
    max_horizon_from_overrides,
)


def prepare_cgtsf_windows(
    cgtsf_cfg,
    domains: List[str],
    lookback_multiplier: Optional[int] = None,
) -> dict:
    overrides = {}

    for domain in domains:
        preset = cgtsf_cfg.dataset_settings(domain)
        if not preset:
            continue
        horizon = preset.get("horizon", cgtsf_cfg.horizon)
        allowed_mults = preset.get(
            "allowed_lookback_multipliers", [2, 3, 4, 5])
        configured_lookback = preset.get("lookback")

        allowed_lookbacks = [horizon * m for m in allowed_mults]

        if lookback_multiplier is not None:
            desired_lookback = horizon * lookback_multiplier
        elif configured_lookback is not None:
            desired_lookback = configured_lookback
        else:
            desired_lookback = max(allowed_lookbacks)

        overrides[domain] = (desired_lookback, horizon)

    return overrides


def get_horizon_map(cgtsf_cfg) -> dict:
    return {
        domain: preset["horizon"]
        for domain, preset in cgtsf_cfg.dataset_presets.items()
    }


def get_allowed_lookback_multipliers(cgtsf_cfg, domain: Optional[str] = None) -> List[int]:
    if domain:
        preset = cgtsf_cfg.dataset_settings(domain)
        if preset:
            return preset.get("allowed_lookback_multipliers", [2, 3, 4, 5])
    first_preset = next(iter(cgtsf_cfg.dataset_presets.values()), {})
    return first_preset.get("allowed_lookback_multipliers", [2, 3, 4, 5])


class CGTSFExperiment(ExperimentBase):

    def __init__(self, cfg, args, *, metrics_csv: Optional[Path] = None):
        super().__init__(
            cfg,
            args,
            data_cfg=cfg.cgtsf,
            metrics_csv=metrics_csv,
        )
        self.lookback_multiplier = getattr(args, "lookback_multiplier", None)
        self.run_all_lookback_multipliers = getattr(
            args, "run_all_lookback_multipliers", False)
        self.text_emb_dir = PROJ_PATH / "data/CGTSF/text_embeddings"

    def build_dataloader(
        self,
        domain: str,
        domain_overrides: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        data_path = Path(self.data_cfg.data_path)

        lookback_overrides = {}
        if domain_overrides and domain in domain_overrides:
            lookback_overrides[domain] = domain_overrides[domain][0]

        if self.model_choice in ("sundial", "timemoe", "timesfm"):
            emb_cache_dir = self.text_emb_dir
            return build_cgtsf_dataloaders_timestep(
                data_path=str(data_path),
                horizon_map=get_horizon_map(self.data_cfg),
                domains=[domain],
                lookback_overrides=lookback_overrides,
                allowed_lookback_multipliers=get_allowed_lookback_multipliers(
                    self.data_cfg),
                batch_size=self.args.batch_size,
                train_ratio=self.data_cfg.train_ratio,
                val_ratio=self.data_cfg.val_ratio,
                data_usage_ratio=self.data_cfg.data_usage_ratio,
                scale=self.data_cfg.scale,
                emb_cache_dir=emb_cache_dir,
                device=str(self.device),
            )

        return build_cgtsf_dataloaders(
            data_path=data_path,
            horizon_map=get_horizon_map(self.data_cfg),
            domains=[domain],
            lookback_overrides=lookback_overrides,
            allowed_lookback_multipliers=get_allowed_lookback_multipliers(
                self.data_cfg),
            batch_size=self.args.batch_size,
            train_ratio=self.data_cfg.train_ratio,
            val_ratio=self.data_cfg.val_ratio,
            data_usage_ratio=self.data_cfg.data_usage_ratio,
            scale=self.data_cfg.scale,
        )

    def _move_batch(self, batch):
        return super()._move_batch(batch)

    @torch.no_grad()
    def save_forecasts(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        output_csv: Path,
        **kwargs,
    ) -> int:
        model.eval()
        base_dataset = unwrap_base_dataset(data_loader.dataset)
        indices = dataset_indices(data_loader.dataset)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "run_id",
            "domain",
            "sample_idx",
            "order",
            "date",
            "horizon",
            "mse",
            "mae",
            "y_true",
            "y_pred",
        ]
        rows_written = 0
        idx_ptr = 0
        run_id = kwargs.get("run_id", "")

        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for batch in data_loader:
                outputs = None
                y = None
                target_idx = 0
                mask_predicted = None

                if isinstance(batch, dict):
                    batch = self._move_batch(batch)
                    y = batch.get("data_to_predict")
                    if y is None:
                        continue
                    y = y.to(dtype=torch.float32)
                    batch["data_to_predict"] = y
                    mask_predicted = batch.get("mask_predicted_data")
                    if torch.is_tensor(mask_predicted):
                        mask_predicted = mask_predicted.to(dtype=torch.float32)
                        batch["mask_predicted_data"] = mask_predicted
                    outputs = model(batch)
                    target_idx = (
                        outputs.get("target_idx")
                        if isinstance(outputs, dict) and outputs.get("target_idx") is not None
                        else batch.get("target_idx", 0)
                    )
                elif isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        x, y = batch[0], batch[1]
                        extra = batch[2] if len(batch) > 2 else None

                        x = x.to(device=self.device, dtype=torch.float32)
                        y = y.to(device=self.device, dtype=torch.float32)

                        if torch.is_tensor(extra):
                            outputs = model.predict(
                                {"observed_data": x, "patch_text_emb": extra.to(
                                    device=self.device, dtype=torch.float32)},
                                horizon=y.shape[1],
                            )
                        elif isinstance(extra, (list, tuple)):
                            outputs = model.predict(
                                {"observed_data": x, "texts": extra}, horizon=y.shape[1]
                            )
                        else:
                            outputs = model.predict(
                                {"observed_data": x, "texts": extra}, horizon=y.shape[1]
                            )
                    else:
                        continue
                else:
                    continue

                if isinstance(outputs, dict):
                    y_hat = None
                    for key in ("y_hat", "predictions", "logits"):
                        val = outputs.get(key)
                        if val is not None:
                            y_hat = val
                            break
                    if outputs.get("mask_forecast_target") is not None:
                        mask_predicted = outputs.get("mask_forecast_target")
                else:
                    y_hat = outputs
                if y_hat is None:
                    continue

                y_hat = y_hat.to(dtype=torch.float32)

                bsz = y.shape[0]
                batch_indices = indices[idx_ptr: idx_ptr + bsz]
                idx_ptr += bsz

                for j, base_idx in enumerate(batch_indices):
                    sample = fetch_sample(base_dataset, base_idx)
                    domain = getattr(sample, "domain", "")
                    order = getattr(sample, "order", "")
                    date = getattr(sample, "date", "")

                    y_true_j = select_target(y[j], target_idx).detach().cpu()
                    y_pred_j = select_target(
                        y_hat[j], target_idx).detach().cpu()
                    if sample is not None:
                        y_true_j = inverse_scale_tensor(
                            y_true_j, domain, base_dataset)
                        y_pred_j = inverse_scale_tensor(
                            y_pred_j, domain, base_dataset)

                    diff = y_pred_j - y_true_j
                    mse = (diff * diff).mean().item()
                    mae = diff.abs().mean().item()

                    row = {
                        "run_id": run_id,
                        "domain": domain,
                        "sample_idx": base_idx,
                        "order": order,
                        "date": str(date) if date else "",
                        "horizon": int(y_true_j.shape[0]),
                        "mse": mse,
                        "mae": mae,
                        "y_true": json.dumps(y_true_j.tolist()),
                        "y_pred": json.dumps(y_pred_j.tolist()),
                    }
                    writer.writerow(row)
                    rows_written += 1

        return rows_written

    def _get_lookback_multiplier_values(self) -> List[Optional[int]]:
        if self.run_all_lookback_multipliers:
            return get_allowed_lookback_multipliers(self.data_cfg)
        elif self.lookback_multiplier is not None:
            return [self.lookback_multiplier]
        else:
            return [None]

    @torch.no_grad()
    def zero_shot_cgtsf(
        self,
        domain: str,
        domain_window_overrides: Optional[dict],
        run_tag: Optional[str] = None,
        current_lookback_multiplier: Optional[int] = None,
    ) -> dict:
        data_path = Path(self.data_cfg.data_path)

        overrides = filter_overrides(domain_window_overrides, domain)
        lb_hz = overrides.get(domain) if overrides else (
            self.data_cfg.lookback, self.data_cfg.horizon
        )

        self.logger.info(f"Dataset path: {data_path}")
        self.logger.info(
            f"Device={self.device} | Lookback={lb_hz[0]} | Horizon={lb_hz[1]}"
        )

        try:
            train_loader, val_loader, test_loader = self.build_dataloader(
                domain=domain,
                domain_overrides=overrides,
            )
        except ValueError as e:
            if "num_samples should be a positive integer value" in str(e):
                self.logger.warning(
                    f"[Zeroshot] Domain '{domain}' has no usable windows; skipping."
                )
                return {}
            raise

        try:
            self.logger.info(
                f"[Data] train={len(train_loader.dataset)}, "
                f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
            )
        except Exception:
            pass

        num_channels = getattr(train_loader, "num_channels", None)
        horizon = lb_hz[1]

        self.cfg.affcon.affflow_text_concat = self.args.affflow_text_concat
        self.cfg.affcon.affflow_text_mode = self.args.affflow_text_mode
        self.cfg.affcon.affflow_prior_type = self.args.affflow_prior_type
        self.cfg.affcon.affflow_adapter = getattr(
            self.args, "affflow_adapter", False)

        model = self.build_model(horizon=horizon, num_channels=num_channels)
        model.eval()

        forecast_csv = self.forecast_dir() / f"{run_tag}_forecasts.csv"
        manifest_path = self.write_run_manifest(
            status="running",
            domain=domain,
            lookback=lb_hz[0],
            horizon=horizon,
            model=model,
            run_id=run_tag,
            forecast_csv=forecast_csv,
            lookback_multiplier=current_lookback_multiplier,
        )
        self.logger.info(f"[Manifest] Saved run manifest to {manifest_path}")

        criterion = nn.MSELoss()
        val_metrics = self.evaluate(model, val_loader, criterion)
        test_metrics = self.evaluate(model, test_loader, criterion)

        self.logger.info(
            f"[Val] MSE: {val_metrics['mse']:.6f} | MAE: {val_metrics['mae']:.6f} | "
            f"RMSE: {val_metrics['rmse']:.6f} | MAPE: {val_metrics['mape']:.6f}"
        )
        self.logger.info(
            f"[Test] MSE: {test_metrics['mse']:.6f} | MAE: {test_metrics['mae']:.6f} | "
            f"RMSE: {test_metrics['rmse']:.6f} | MAPE: {test_metrics['mape']:.6f}"
        )

        self._augment_with_crps(
            model, train_loader, test_loader, test_metrics, domain=domain,
        )

        rows = self.save_forecasts(
            model, test_loader, forecast_csv, domain=domain, run_id=run_tag
        )
        self.logger.info(
            f"[Forecast] Saved {rows} forecasts to {forecast_csv}"
        )
        manifest_path = self.write_run_manifest(
            status="completed",
            domain=domain,
            lookback=lb_hz[0],
            horizon=horizon,
            model=model,
            run_id=run_tag,
            forecast_csv=forecast_csv,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            lookback_multiplier=current_lookback_multiplier,
            rows_written=rows,
        )
        self.logger.info(f"[Manifest] Saved run manifest to {manifest_path}")

        return {"val": val_metrics, "test": test_metrics}

    def run(self) -> None:
        data_path = Path(self.data_cfg.data_path)
        domains = list(self.data_cfg.dataset_presets.keys())

        horizon_map = get_horizon_map(self.data_cfg)

        for current_lookback_multiplier in self._get_lookback_multiplier_values():
            for domain in domains:
                domain_window_overrides = prepare_cgtsf_windows(
                    self.data_cfg, [
                        domain], lookback_multiplier=current_lookback_multiplier
                )

                if domain_window_overrides:
                    self.data_cfg.lookback = max(
                        lb for lb, _ in domain_window_overrides.values()
                    )
                    self.data_cfg.horizon = max(
                        hz for _, hz in domain_window_overrides.values()
                    )

                active_overrides = filter_overrides(
                    domain_window_overrides, domain)
                lb, hz = effective_window(
                    domain, active_overrides, self.data_cfg)

                self.setup_run_logger(domain, lookback=lb, horizon=hz)

                allowed_mults = get_allowed_lookback_multipliers(
                    self.data_cfg, domain)
                allowed_lookbacks = [
                    horizon_map.get(domain, hz) * m
                    for m in allowed_mults
                ]
                self.logger.info(f"Dataset root: {data_path}")
                self.logger.info(f"Domain: {domain}")
                self.logger.info(
                    f"CGTSF window for {domain}: lookback={lb}, horizon={hz}; "
                    f"allowed lookbacks {allowed_lookbacks}"
                )
                if current_lookback_multiplier is not None:
                    self.logger.info(
                        f"Using lookback_multiplier={current_lookback_multiplier}")
                self.logger.info(
                    f"Lookback={lb}, Horizon={hz}, Device={self.device}")

                if self.run_mode == "train":
                    self._run_train(
                        domain=domain,
                        active_overrides=active_overrides,
                        current_lookback_multiplier=current_lookback_multiplier,
                        lb=lb,
                        hz=hz,
                    )
                else:
                    self._run_zero_shot(
                        domain=domain,
                        active_overrides=active_overrides,
                        current_lookback_multiplier=current_lookback_multiplier,
                        lb=lb,
                        hz=hz,
                    )

    def _run_train(
        self,
        domain: str,
        active_overrides: Optional[dict],
        current_lookback_multiplier: Optional[int],
        lb: int,
        hz: int,
    ) -> None:
        model, best_metrics = self.train(
            domain=domain,
            domain_overrides=active_overrides,
            model_horizon=max_horizon_from_overrides(
                domain, active_overrides, self.data_cfg.horizon
            ),
            lookback_multiplier=current_lookback_multiplier,
        )
        if best_metrics:
            log_metrics(
                self.metrics_csv,
                run_id=self.run_id,
                run_mode="train",
                tsfm_model=self.model_choice,
                active_dataset=self.active_dataset,
                domain=domain,
                lookback=lb,
                horizon=hz,
                test_metrics=best_metrics.get("test", {}),
                run_stage=self.run_stage,
                run_tag=self.run_tag,
                lookback_multiplier=current_lookback_multiplier,
                **self.current_alignment_fields(),
            )

    def _run_zero_shot(
        self,
        domain: str,
        active_overrides: Optional[dict],
        current_lookback_multiplier: Optional[int],
        lb: int,
        hz: int,
    ) -> None:
        self.logger.info(
            f"[Zeroshot] Domain '{domain}' | Window (lookback={lb}, horizon={hz})"
        )

        run_id = format_run_id(
            run_mode=self.run_mode,
            tsfm_model=self.model_choice,
            active_dataset=self.active_dataset,
            timestamp=self.run_timestamp,
            domain=domain,
            run_stage=self.run_stage,
            run_tag=self.run_tag,
            lookback=lb,
            horizon=hz,
            variant_tag=self.current_variant_tag(),
        )
        zs_result = self.zero_shot_cgtsf(
            domain=domain,
            domain_window_overrides=active_overrides,
            run_tag=run_id,
            current_lookback_multiplier=current_lookback_multiplier,
        )
        test_metrics = zs_result.get("test", {}) if zs_result else {}

        if test_metrics:
            log_metrics(
                self.metrics_csv,
                run_id=self.run_id,
                run_mode="zeroshot",
                tsfm_model=self.model_choice,
                active_dataset=self.active_dataset,
                domain=domain,
                lookback=lb,
                horizon=hz,
                test_metrics=test_metrics,
                run_stage=self.run_stage,
                run_tag=self.run_tag,
                lookback_multiplier=current_lookback_multiplier,
                **self.current_alignment_fields(),
            )
