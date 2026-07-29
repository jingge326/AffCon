
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.data_timeimm import build_timeimm_dataloaders
from experiments.experiment_base import (
    ExperimentBase,
    format_run_id,
    dataset_indices,
    unwrap_base_dataset,
    fetch_sample,
    select_target,
    log_metrics,
)


class TimeIMMExperiment(ExperimentBase):

    def __init__(self, cfg, args, *, metrics_csv: Optional[Path] = None):
        super().__init__(
            cfg,
            args,
            data_cfg=cfg.timeimm,
            metrics_csv=metrics_csv,
        )
        self.use_covariates = False
        self.use_text = True
        self.use_text_embeddings = self.data_cfg.use_text_embeddings if self.use_text else False

    def build_dataloader(
        self,
        domain: str,
        domain_overrides: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset_name = domain
        settings = self.data_cfg.dataset_settings(dataset_name)

        return build_timeimm_dataloaders(
            data_path=self.data_cfg.data_path,
            dataset=dataset_name,
            lookback=settings["lookback"],
            horizon=settings["horizon"],
            stride=settings["stride"],
            batch_size=self.args.batch_size,
            split_method=self.data_cfg.split_method,
            time_unit=settings["time_unit"],
            enable_text=self.use_text,
            use_text_embeddings=self.use_text_embeddings,
            llm_model_fusion=self.data_cfg.llm_model_fusion,
            llm_layers_fusion=self.data_cfg.llm_layers_fusion,
            max_length=self.data_cfg.max_length,
            train_ratio=self.data_cfg.train_ratio,
            val_ratio=self.data_cfg.val_ratio,
            target_column=self.data_cfg.target_column or settings.get("OT"),
            model=self.args.tsfm_model,
            use_time_rescale=self.data_cfg.use_time_rescale,
            resample_interval_sec=settings.get("resample_interval_sec"),
            use_covariates=self.use_covariates,
            timeimm_irregularity=getattr(
                self.args, "timeimm_irregularity", "resampled"),
            tsfm_model=self.args.tsfm_model,
            device=self.args.device,
            selected_chunk_ids=getattr(
                self.args, "timeimm_selected_chunk_ids", None),
            chunk_anchor_date=getattr(self.args, "chunk_anchor_date", None),
            use_patch_text_embeds=self.use_patch_text_embeds,
            fill_policy=getattr(self.args, "fill_policy", "global_mean"),
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
        indices = dataset_indices(data_loader.dataset)
        base_dataset = unwrap_base_dataset(data_loader.dataset)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        dataset_name = kwargs.get("domain")
        if not dataset_name:
            domains = kwargs.get("domains", [])
            dataset_name = domains[0] if domains else "unknown"
        settings = self.data_cfg.dataset_settings(dataset_name) or {}
        time_unit = settings.get("time_unit")
        resample_interval_sec = settings.get("resample_interval_sec")

        fieldnames = [
            "run_id",
            "dataset",
            "sample_idx",
            "chunk_id",
            "target_idx",
            "target_name",
            "horizon",
            "time_unit",
            "resample_interval_sec",
            "tp_future_raw",
            "tp_future",
            "mask_predicted",
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
                if not isinstance(batch, dict):
                    continue

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
                if isinstance(outputs, dict):
                    y_hat = None
                    for key in ("y_hat", "predictions", "logits"):
                        val = outputs.get(key)
                        if val is not None:
                            y_hat = val
                            break
                else:
                    y_hat = outputs
                if y_hat is None:
                    continue
                y_hat = y_hat.to(dtype=torch.float32)

                target_idx = (
                    outputs.get("target_idx")
                    if isinstance(outputs, dict) and outputs.get("target_idx") is not None
                    else batch.get("target_idx", 0)
                )
                target_name = batch.get("target_name", "")
                target_idx = int(target_idx) if target_idx is not None else 0

                bsz = y.shape[0]
                batch_indices = indices[idx_ptr: idx_ptr + bsz]
                idx_ptr += bsz

                tp_future = batch.get("tp_to_predict")
                tp_future_raw = batch.get(
                    "raw_tp_to_predict", batch.get("tp_to_predict_raw"))
                raw_vals_b = batch.get("raw_data_to_predict")
                raw_mask_b = batch.get("raw_mask_predicted_data")
                raw_tp_b = batch.get("raw_tp_to_predict")
                grid_tp_raw_b = batch.get("tp_to_predict_raw")
                use_irregular = (
                    torch.is_tensor(raw_vals_b) and torch.is_tensor(raw_mask_b)
                    and torch.is_tensor(raw_tp_b) and torch.is_tensor(grid_tp_raw_b)
                    and raw_vals_b.numel() > 0
                )

                for j, base_idx in enumerate(batch_indices):
                    sample = fetch_sample(base_dataset, int(base_idx))
                    y_true_j = select_target(y[j], target_idx).detach().cpu()
                    y_pred_j = select_target(
                        y_hat[j], target_idx).detach().cpu()
                    horizon = int(y_true_j.shape[0])

                    mask_j = None
                    if torch.is_tensor(mask_predicted):
                        mask_j = select_target(
                            mask_predicted[j], target_idx).detach().cpu()

                    if torch.is_tensor(tp_future_raw):
                        t_raw_j = tp_future_raw[j]
                        if t_raw_j.numel() > horizon:
                            t_raw_j = t_raw_j[:horizon]
                        t_raw = json.dumps(t_raw_j.detach().cpu().tolist())
                    else:
                        t_raw = ""
                    if torch.is_tensor(tp_future):
                        t_future_j = tp_future[j]
                        if t_future_j.numel() > horizon:
                            t_future_j = t_future_j[:horizon]
                        t_future_val = json.dumps(
                            t_future_j.detach().cpu().tolist())
                    else:
                        t_future_val = ""

                    if use_irregular:
                        time_mask_j = raw_mask_b[j][:, target_idx] > 0
                        if time_mask_j.any():
                            tt_raw_j = raw_tp_b[j][time_mask_j]
                            vals_raw_j = raw_vals_b[j][time_mask_j][:, target_idx].detach().cpu()
                            mask_raw_j = raw_mask_b[j][time_mask_j][:, target_idx].to(dtype=torch.float32).detach().cpu()
                            ms = mask_raw_j.sum().item()
                            if ms > 0:
                                diff_time = torch.abs(
                                    grid_tp_raw_b[j].detach().cpu().view(1, -1)
                                    - tt_raw_j.detach().cpu().view(-1, 1))
                                nn_idx = diff_time.argmin(dim=1)
                                preds_at_raw = y_pred_j[nn_idx]
                                d = (preds_at_raw - vals_raw_j) * mask_raw_j
                                mse = (d * d).sum().item() / ms
                                mae = d.abs().sum().item() / ms
                            else:
                                mse = mae = float("nan")
                        else:
                            mse = mae = float("nan")
                    elif torch.is_tensor(mask_j):
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
                        "dataset": dataset_name,
                        "sample_idx": base_idx,
                        "chunk_id": getattr(sample, "chunk_id", "") if sample is not None else "",
                        "target_idx": target_idx,
                        "target_name": target_name or "",
                        "horizon": horizon,
                        "time_unit": time_unit or "",
                        "resample_interval_sec": resample_interval_sec if resample_interval_sec is not None else "",
                        "tp_future_raw": t_raw,
                        "tp_future": t_future_val,
                        "mask_predicted": json.dumps(mask_j.tolist()) if torch.is_tensor(mask_j) else "",
                        "mse": mse,
                        "mae": mae,
                        "y_true": json.dumps(y_true_j.tolist()),
                        "y_pred": json.dumps(y_pred_j.tolist()),
                    }
                    writer.writerow(row)
                    rows_written += 1

        return rows_written

    @torch.no_grad()
    def zero_shot_timeimm(
        self,
        dataset_name: str,
        run_tag: Optional[str] = None,
    ) -> dict:
        data_path = Path(self.data_cfg.data_path)
        settings = self.data_cfg.dataset_settings(dataset_name)

        self.logger.info(f"Dataset path: {data_path}")
        self.logger.info(
            f"Device={self.device} | Lookback={settings['lookback']} | "
            f"Horizon={settings['horizon']}"
        )

        train_loader, val_loader, test_loader = self.build_dataloader(
            domain=dataset_name,
            domain_overrides=None,
        )

        self.logger.info(
            f"[Data] train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
        )

        num_channels = getattr(train_loader, "num_channels", None)
        horizon = settings["horizon"]

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
            domain=dataset_name,
            lookback=settings["lookback"],
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
            model, train_loader, test_loader, test_metrics, domain=dataset_name,
        )

        rows = self.save_forecasts(
            model, test_loader, forecast_csv, domain=dataset_name, run_id=run_tag
        )
        self.logger.info(
            f"[Forecast] Saved {rows} forecasts to {forecast_csv}"
        )
        manifest_path = self.write_run_manifest(
            status="completed",
            domain=dataset_name,
            lookback=settings["lookback"],
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
        dataset_names = self.data_cfg.datasets or [self.data_cfg.dataset]

        for dataset_name in dataset_names:
            settings = self.data_cfg.dataset_settings(dataset_name)
            lookback = settings["lookback"]
            horizon = settings["horizon"]

            self.setup_run_logger(
                dataset_name, lookback=lookback, horizon=horizon)

            self.logger.info(f"IMM-TSF dataset: {dataset_name}")
            self.logger.info(f"Dataset path: {self.data_cfg.data_path}")
            self.logger.info(
                f"Lookback={lookback}, Horizon={horizon}, "
                f"Stride={settings['stride']}, Time unit={settings['time_unit']}"
            )

            if self.run_mode == "train":
                model, best_metrics = self.train(
                    domain=dataset_name,
                    domain_overrides=None,
                    model_horizon=horizon,
                )
                if best_metrics:
                    log_metrics(
                        self.metrics_csv,
                        run_id=self.run_id,
                        seed=self.args.seed,
                        run_mode="train",
                        tsfm_model=self.model_choice,
                        active_dataset=self.active_dataset,
                        domain=dataset_name,
                        lookback=lookback,
                        horizon=horizon,
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
                    domain=dataset_name,
                    run_stage=self.run_stage,
                    run_tag=self.run_tag,
                    lookback=lookback,
                    horizon=horizon,
                    variant_tag=self.current_variant_tag(),
                )

                zs_metrics = self.zero_shot_timeimm(
                    dataset_name=dataset_name,
                    run_tag=run_id,
                )
                log_metrics(
                    self.metrics_csv,
                    run_id=run_id,
                    seed=self.args.seed,
                    run_mode="zeroshot",
                    tsfm_model=self.model_choice,
                    active_dataset=self.active_dataset,
                    domain=dataset_name,
                    lookback=lookback,
                    horizon=horizon,
                    test_metrics=zs_metrics.get("test", {}),
                    run_stage=self.run_stage,
                    run_tag=self.run_tag,
                    **self.current_alignment_fields(),
                )
