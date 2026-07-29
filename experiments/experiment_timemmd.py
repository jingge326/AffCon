
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.data_timemmd import (
    build_time_mmd_dataloaders_timestep,
)
from experiments.experiment_base import (
    ExperimentBase,
    filter_overrides,
    effective_window,
    inverse_scale_tensor,
    unwrap_base_dataset,
    dataset_indices,
    fetch_sample,
    select_target,
    log_metrics,
    max_horizon_from_overrides,
    format_run_id,
)


def apply_window_presets(data_cfg) -> Optional[dict]:
    domain_window_overrides = None
    if data_cfg.use_domain_freq_windows:
        domain_window_overrides = {}
        for domain, preset in data_cfg.dataset_presets.items():
            lb = preset["lookback"]
            hz = preset["horizons"][0] if preset.get("horizons") else lb
            domain_window_overrides[domain] = (lb, hz)
    return domain_window_overrides


class TimeMMDExperiment(ExperimentBase):

    def __init__(self, cfg, args, *, metrics_csv: Optional[Path] = None):
        super().__init__(
            cfg,
            args,
            data_cfg=cfg.time_mmd,
            metrics_csv=metrics_csv,
        )
        data_path = Path(self.data_cfg.data_path)
        self.text_emb_dir = data_path / "text_embeddings"

    def build_dataloader(
        self,
        domain: str,
        domain_overrides: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        data_path = Path(self.data_cfg.data_path)

        emb_cache_dir = self.text_emb_dir
        return build_time_mmd_dataloaders_timestep(
            data_path=str(data_path),
            lookback=self.data_cfg.lookback,
            horizon=self.data_cfg.horizon,
            batch_size=self.args.batch_size,
            emb_cache_dir=emb_cache_dir,
            train_ratio=self.data_cfg.train_ratio,
            val_ratio=self.data_cfg.val_ratio,
            use_report=self.data_cfg.use_report,
            use_search=self.data_cfg.use_search,
            domains=[domain],
            scale=self.data_cfg.scale,
            domain_window_overrides=domain_overrides,
            tsfm_model=self.model_choice or "sundial",
            llm_model=self.data_cfg.llm_model_fusion,
            llm_layers=self.data_cfg.llm_layers_fusion,
            input_token_len=self.data_cfg.input_token_len,
            device=str(self.device),
            use_patch_text_embeds=self.use_patch_text_embeds,
        )

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
            "start_idx",
            "end_idx",
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
                    if len(batch) == 5:
                        x, y, _, mask_fut, texts_per_patch = batch
                        x = x.to(device=self.device, dtype=torch.float32)
                        y = y.to(device=self.device, dtype=torch.float32)
                        mask_predicted = mask_fut.to(
                            device=self.device, dtype=torch.float32)
                        outputs = model.predict(
                            {"observed_data": x, "texts_per_patch": texts_per_patch},
                            horizon=y.shape[1],
                        )
                    elif len(batch) == 3:
                        x, y, third = batch
                        x = x.to(device=self.device, dtype=torch.float32)
                        y = y.to(device=self.device, dtype=torch.float32)
                        if torch.is_tensor(third):
                            outputs = model.predict(
                                {"observed_data": x, "patch_text_emb": third.to(
                                    device=self.device, dtype=torch.float32)},
                                horizon=y.shape[1],
                            )
                        else:
                            outputs = model.predict(
                                {"observed_data": x, "texts": third},
                                horizon=y.shape[1],
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
                if y_hat is None or y is None:
                    continue

                y_hat = y_hat.to(dtype=torch.float32)
                target_idx = int(target_idx) if target_idx is not None else 0

                bsz = y.shape[0]
                batch_indices = indices[idx_ptr: idx_ptr + bsz]
                idx_ptr += bsz

                for j, base_idx in enumerate(batch_indices):
                    sample = fetch_sample(base_dataset, base_idx)
                    domain = getattr(sample, "domain", "")
                    start_idx = getattr(sample, "start_idx", "")
                    end_idx = getattr(sample, "end_idx", "")

                    y_true_j = select_target(y[j], target_idx).detach().cpu()
                    y_pred_j = select_target(
                        y_hat[j], target_idx).detach().cpu()
                    if sample is not None:
                        y_true_j = inverse_scale_tensor(
                            y_true_j, domain, base_dataset)
                        y_pred_j = inverse_scale_tensor(
                            y_pred_j, domain, base_dataset)

                    mask_j = None
                    if torch.is_tensor(mask_predicted):
                        mask_j = select_target(
                            mask_predicted[j], target_idx).detach().cpu()

                    if torch.is_tensor(mask_j):
                        mask_sum = mask_j.sum().item()
                        if mask_sum > 0:
                            diff = (y_pred_j - y_true_j) * mask_j
                            mse = (diff * diff).sum().item() / mask_sum
                            mae = diff.abs().sum().item() / mask_sum
                        else:
                            diff = y_pred_j - y_true_j
                            mse = (diff * diff).mean().item()
                            mae = diff.abs().mean().item()
                    else:
                        diff = y_pred_j - y_true_j
                        mse = (diff * diff).mean().item()
                        mae = diff.abs().mean().item()

                    row = {
                        "run_id": run_id,
                        "domain": domain,
                        "sample_idx": base_idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "horizon": int(y_true_j.shape[0]),
                        "mse": mse,
                        "mae": mae,
                        "y_true": json.dumps(y_true_j.tolist()),
                        "y_pred": json.dumps(y_pred_j.tolist()),
                    }
                    writer.writerow(row)
                    rows_written += 1

        return rows_written

    def _move_batch(self, batch):
        return super()._move_batch(batch)

    @torch.no_grad()
    def zero_shot_timemmd(
        self,
        domain: str,
        domain_window_overrides: Optional[dict],
        run_tag: Optional[str] = None,
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
            rows_written=rows,
        )
        self.logger.info(f"[Manifest] Saved run manifest to {manifest_path}")

        return {"val": val_metrics, "test": test_metrics}

    def run(self) -> None:
        data_path = Path(self.data_cfg.data_path)
        domains = list(self.data_cfg.dataset_presets.keys())

        domain_window_overrides = apply_window_presets(self.data_cfg)

        for domain in domains:
            active_overrides = filter_overrides(
                domain_window_overrides, domain)
            lb, hz = effective_window(
                domain, active_overrides, self.data_cfg)

            self.setup_run_logger(domain, lookback=lb, horizon=hz)

            self.logger.info(f"Dataset root: {data_path}")
            self.logger.info(f"Domain: {domain}")
            self.logger.info(
                f"Lookback={lb}, Horizon={hz}, Device={self.device}")

            if self.run_mode == "train":
                model, best_metrics = self.train(
                    domain=domain,
                    domain_overrides=active_overrides,
                    model_horizon=max_horizon_from_overrides(
                        domain, active_overrides, self.data_cfg.horizon
                    ),
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
                        **self.current_alignment_fields(),
                    )
            else:
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
                zs_result = self.zero_shot_timemmd(
                    domain=domain,
                    domain_window_overrides=active_overrides,
                    run_tag=run_id,
                )
                test_metrics = zs_result.get(
                    "test", {}) if zs_result else {}

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
                        **self.current_alignment_fields(),
                    )
