
from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence




def _left_pad_sequence(seq_list, padding_value: float = 0.0, device: torch.device | None = None):
    if len(seq_list) == 0:
        return None
    max_len = max(t.size(0) for t in seq_list)
    first = seq_list[0]
    if device is None:
        device = first.device
    out_shape = (len(seq_list), max_len, *first.shape[1:])
    out = first.new_full(out_shape, padding_value, device=device)
    for i, t in enumerate(seq_list):
        L = t.size(0)
        out[i, max_len - L: max_len] = t.to(device)
    return out


def _maybe_get_step_units(args, unit_seconds_map: dict):
    if not getattr(args, "use_time_rescale", False):
        return None
    resample_interval_sec = getattr(args, "resample_interval_sec", None)
    if resample_interval_sec is None:
        raise ValueError(
            "use_time_rescale is True but resample_interval_sec is not set."
        )
    time_unit = getattr(args, "time_unit", None)
    if time_unit == "custom":
        unit_seconds = getattr(args, "unit_scale", None)
        if unit_seconds is None:
            raise ValueError("unit_scale must be set when time_unit='custom'.")
        unit_seconds = float(unit_seconds)
    else:
        unit_seconds = unit_seconds_map.get(time_unit)
    if unit_seconds is None:
        raise ValueError(f"Unknown time_unit '{time_unit}' for time rescale.")
    return resample_interval_sec / unit_seconds


def normalize_masked_tp(data, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8
    if (scale != 0.).all():
        data_norm = (data - att_min) / scale
    return data_norm




def collate_varlen_rightpad(batch, args, time_max=None, unit_seconds_map=None):
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    unit_seconds_map = unit_seconds_map or {}
    step_units = _maybe_get_step_units(args, unit_seconds_map)
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )
    for _, tt, vals, mask in batch:
        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])
    observed_tp = pad_sequence(
        observed_tp, batch_first=True, padding_value=0.0)
    observed_data = pad_sequence(
        observed_data, batch_first=True, padding_value=0.0)
    observed_mask = pad_sequence(
        observed_mask, batch_first=True, padding_value=0.0)
    predicted_tp = pad_sequence(
        predicted_tp, batch_first=True, padding_value=0.0)
    predicted_data = pad_sequence(
        predicted_data, batch_first=True, padding_value=0.0)
    predicted_mask = pad_sequence(
        predicted_mask, batch_first=True, padding_value=0.0)

    if step_units is not None:
        observed_tp = observed_tp / step_units
        predicted_tp = predicted_tp / step_units
    else:
        observed_tp = normalize_masked_tp(
            observed_tp, att_min=0.0, att_max=chunk_time_max
        )
        predicted_tp = normalize_masked_tp(
            predicted_tp, att_min=0.0, att_max=chunk_time_max
        )
    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }


def collate_varlen_leftpad(batch, args, time_max=None, unit_seconds_map=None):
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    unit_seconds_map = unit_seconds_map or {}
    step_units = _maybe_get_step_units(args, unit_seconds_map)
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )
    for _, tt, vals, mask in batch:
        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])

    observed_tp = _left_pad_sequence(
        observed_tp, padding_value=0.0, device=args.device)
    observed_data = _left_pad_sequence(
        observed_data, padding_value=0.0, device=args.device)
    observed_mask = _left_pad_sequence(
        observed_mask, padding_value=0.0, device=args.device)
    predicted_tp = _left_pad_sequence(
        predicted_tp, padding_value=0.0, device=args.device)
    predicted_data = _left_pad_sequence(
        predicted_data, padding_value=0.0, device=args.device)
    predicted_mask = _left_pad_sequence(
        predicted_mask, padding_value=0.0, device=args.device)

    if step_units is not None:
        observed_tp = observed_tp / step_units
        predicted_tp = predicted_tp / step_units
    else:
        observed_tp = normalize_masked_tp(
            observed_tp, att_min=0.0, att_max=chunk_time_max
        )
        predicted_tp = normalize_masked_tp(
            predicted_tp, att_min=0.0, att_max=chunk_time_max
        )
    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }



EVAL_EPS = 1e-8


def _persample_grid(y_hat, y, mask):
    B = y_hat.shape[0]
    out = {"per_sample_mse": [], "per_sample_mae": [],
           "per_sample_mape": [], "per_sample_mspe": []}
    for b in range(B):
        diff = y_hat[b] - y[b]
        denom = y[b].abs() + EVAL_EPS
        if mask is not None:
            m = mask[b].to(dtype=torch.float32)
            ms = m.sum().item()
            if ms == 0:
                continue
            d = diff * m
            out["per_sample_mse"].append((d * d).sum().item() / ms)
            out["per_sample_mae"].append(d.abs().sum().item() / ms)
            out["per_sample_mape"].append((d.abs() / denom * m).sum().item() / ms)
            out["per_sample_mspe"].append(((d / denom) ** 2 * m).sum().item() / ms)
        else:
            n = float(diff.numel())
            if n == 0:
                continue
            out["per_sample_mse"].append((diff * diff).sum().item() / n)
            out["per_sample_mae"].append(diff.abs().sum().item() / n)
            out["per_sample_mape"].append((diff.abs() / denom).sum().item() / n)
            out["per_sample_mspe"].append(((diff / denom) ** 2).sum().item() / n)
    return out


def irregular_raw_alignment(raw_vals, raw_mask, raw_tp, grid_tp_raw, target_idx, b):
    time_mask = raw_mask[b][:, target_idx] > 0
    if not time_mask.any():
        return None
    tt_raw = raw_tp[b][time_mask]
    vals_raw = raw_vals[b][time_mask][:, target_idx]
    mask_raw = raw_mask[b][time_mask][:, target_idx].to(dtype=torch.float32)
    if mask_raw.sum().item() == 0:
        return None
    grid_tp_b = grid_tp_raw[b]
    diff_time = torch.abs(grid_tp_b.view(1, -1) - tt_raw.view(-1, 1))
    nn_idx = diff_time.argmin(dim=1)
    return vals_raw, mask_raw, nn_idx


def _persample_irregular(y_hat, raw_vals, raw_mask, raw_tp, grid_tp_raw, target_idx):
    B = y_hat.shape[0]
    out = {"per_sample_mse": [], "per_sample_mae": [],
           "per_sample_mape": [], "per_sample_mspe": []}
    for b in range(B):
        aligned = irregular_raw_alignment(
            raw_vals, raw_mask, raw_tp, grid_tp_raw, target_idx, b)
        if aligned is None:
            continue
        vals_raw, mask_raw, nn_idx = aligned
        ms = mask_raw.sum().item()
        preds_at_raw = y_hat[b, nn_idx]
        diff = (preds_at_raw - vals_raw) * mask_raw
        denom = vals_raw.abs() + EVAL_EPS
        out["per_sample_mse"].append((diff * diff).sum().item() / ms)
        out["per_sample_mae"].append(diff.abs().sum().item() / ms)
        out["per_sample_mape"].append((diff.abs() / denom * mask_raw).sum().item() / ms)
        out["per_sample_mspe"].append(((diff / denom) ** 2 * mask_raw).sum().item() / ms)
    return out


def evaluate_batch_persample(
    *,
    y_hat,
    y,
    mask=None,
    raw_vals=None,
    raw_mask=None,
    raw_tp=None,
    grid_tp_raw=None,
    target_idx: int = 0,
):
    if (
        torch.is_tensor(raw_vals) and torch.is_tensor(raw_mask)
        and torch.is_tensor(raw_tp) and torch.is_tensor(grid_tp_raw)
        and raw_vals.numel() > 0
    ):
        return _persample_irregular(y_hat, raw_vals, raw_mask, raw_tp, grid_tp_raw, target_idx)
    return _persample_grid(y_hat, y, mask)


def aggregate_macro(per_sample):
    n = len(per_sample.get("per_sample_mse", []))
    if n == 0:
        return {
            "mse": float("inf"), "mae": float("inf"),
            "rmse": float("inf"), "mape": float("inf"),
            "mspe": float("inf"),
        }
    mse = sum(per_sample["per_sample_mse"]) / n
    mae = sum(per_sample["per_sample_mae"]) / n
    mape = sum(per_sample["per_sample_mape"]) / n
    mspe = sum(per_sample["per_sample_mspe"]) / n
    return {"mse": mse, "mae": mae, "rmse": mse ** 0.5, "mape": mape, "mspe": mspe}


def collate_varlen_autoreg(batch, args, time_max=None, unit_seconds_map=None):
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    unit_seconds_map = unit_seconds_map or {}
    step_units = _maybe_get_step_units(args, unit_seconds_map)
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )
    for _, tt, vals, mask in batch:
        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])

    observed_tp = _left_pad_sequence(
        observed_tp, padding_value=0.0, device=args.device)
    observed_data = _left_pad_sequence(
        observed_data, padding_value=0.0, device=args.device)
    observed_mask = _left_pad_sequence(
        observed_mask, padding_value=0.0, device=args.device)

    predicted_tp = pad_sequence(
        predicted_tp, batch_first=True, padding_value=0.0)
    predicted_data = pad_sequence(
        predicted_data, batch_first=True, padding_value=0.0)
    predicted_mask = pad_sequence(
        predicted_mask, batch_first=True, padding_value=0.0)

    if step_units is not None:
        observed_tp = observed_tp / step_units
        predicted_tp = predicted_tp / step_units
    else:
        observed_tp = normalize_masked_tp(
            observed_tp, att_min=0.0, att_max=chunk_time_max
        )
        predicted_tp = normalize_masked_tp(
            predicted_tp, att_min=0.0, att_max=chunk_time_max
        )
    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }
