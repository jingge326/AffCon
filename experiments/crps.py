from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader


def _reduce_over_horizon(
    crps_per_step: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is not None:
        mask = mask.to(dtype=crps_per_step.dtype)
        mask_sum = mask.sum(dim=-1)
        denom = mask_sum.clamp(min=1.0)
        result = (crps_per_step * mask).sum(dim=-1) / denom
        nan_fill = torch.full_like(result, float("nan"))
        return torch.where(mask_sum > 0, result, nan_fill)
    return crps_per_step.mean(dim=-1)


def crps_from_samples(
    samples: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if samples.dim() != 3:
        raise ValueError(f"samples must be [B, S, H], got {tuple(samples.shape)}")
    B, S, H = samples.shape
    if y.shape != (B, H):
        raise ValueError(f"y must be [B, H]=({B},{H}), got {tuple(y.shape)}")

    samples = samples.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    if S == 1:
        crps_per_step = (samples[:, 0, :] - y).abs()
        return _reduce_over_horizon(crps_per_step, mask)

    term1 = (samples - y.unsqueeze(1)).abs().mean(dim=1)

    samples_sorted, _ = torch.sort(samples, dim=1)
    i = torch.arange(1, S + 1, device=samples.device, dtype=torch.float32)
    w = (2.0 * i - S - 1.0) / (S * (S - 1))
    term2 = (samples_sorted * w.view(1, S, 1)).sum(dim=1)

    crps_per_step = term1 - term2
    return _reduce_over_horizon(crps_per_step, mask)


def crps_from_quantiles_pwl(
    q: torch.Tensor,
    taus: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q.dim() != 3:
        raise ValueError(f"q must be [B, H, K], got {tuple(q.shape)}")
    B, H, K = q.shape
    if taus.shape != (K,):
        raise ValueError(f"taus must be [K]=({K},), got {tuple(taus.shape)}")
    if y.shape != (B, H):
        raise ValueError(f"y must be [B, H]=({B},{H}), got {tuple(y.shape)}")
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")

    q = q.to(dtype=torch.float32)
    taus = taus.to(dtype=torch.float32, device=q.device)
    y = y.to(dtype=torch.float32, device=q.device)

    q_sorted, _ = torch.sort(q, dim=-1)

    eps = 1e-12
    slope_left = (q_sorted[..., 1] - q_sorted[..., 0]) / (taus[1] - taus[0]).clamp(min=eps)
    slope_right = (q_sorted[..., -1] - q_sorted[..., -2]) / (taus[-1] - taus[-2]).clamp(min=eps)
    q0 = q_sorted[..., 0:1] - slope_left.unsqueeze(-1) * taus[0]
    qK1 = q_sorted[..., -1:] + slope_right.unsqueeze(-1) * (1.0 - taus[-1])

    taus_full = torch.cat([
        torch.zeros(1, device=q.device, dtype=torch.float32),
        taus,
        torch.ones(1, device=q.device, dtype=torch.float32),
    ])
    q_full = torch.cat([q0, q_sorted, qK1], dim=-1)

    y_b = y.unsqueeze(-1)
    tau_l = taus_full[:-1].view(1, 1, -1)
    tau_r = taus_full[1:].view(1, 1, -1)
    dtau = (tau_r - tau_l)
    q_l = q_full[..., :-1]
    q_r = q_full[..., 1:]
    delta_q = q_r - q_l
    safe_dq = delta_q.clamp(min=eps)

    term_a_out = dtau * (torch.abs(q_l - y_b) + torch.abs(q_r - y_b)) / 2.0
    term_a_in = dtau * ((y_b - q_l).pow(2) + (q_r - y_b).pow(2)) / (2.0 * safe_dq)
    y_in_seg = (y_b >= q_l) & (y_b <= q_r) & (delta_q > eps)
    term_a = torch.where(y_in_seg, term_a_in, term_a_out)

    slope = delta_q / dtau.clamp(min=eps)
    a = 2.0 * tau_l - 1.0
    term_b = (
        q_l * (dtau.pow(2) + a * dtau)
        + slope * ((2.0 / 3.0) * dtau.pow(3) + (a / 2.0) * dtau.pow(2))
    )

    crps_per_step = (term_a - term_b).sum(dim=-1)
    return _reduce_over_horizon(crps_per_step, mask)


def _flatten_valid(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        keep = mask.to(dtype=torch.bool, device=values.device)
        return values[keep]
    return values.reshape(-1)


def pit_from_samples(
    samples: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if samples.dim() != 3:
        raise ValueError(f"samples must be [B, S, H], got {tuple(samples.shape)}")
    samples = samples.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    pit = (samples <= y.unsqueeze(1)).to(dtype=torch.float32).mean(dim=1)
    return _flatten_valid(pit, mask)


def pit_from_quantiles_pwl(
    q: torch.Tensor,
    taus: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q.dim() != 3:
        raise ValueError(f"q must be [B, H, K], got {tuple(q.shape)}")
    B, H, K = q.shape
    if taus.shape != (K,):
        raise ValueError(f"taus must be [K]=({K},), got {tuple(taus.shape)}")
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")

    q = q.to(dtype=torch.float32)
    taus = taus.to(dtype=torch.float32, device=q.device)
    y = y.to(dtype=torch.float32, device=q.device)

    q_sorted, _ = torch.sort(q, dim=-1)
    eps = 1e-12
    slope_left = (q_sorted[..., 1] - q_sorted[..., 0]) / (taus[1] - taus[0]).clamp(min=eps)
    slope_right = (q_sorted[..., -1] - q_sorted[..., -2]) / (taus[-1] - taus[-2]).clamp(min=eps)
    q0 = q_sorted[..., 0:1] - slope_left.unsqueeze(-1) * taus[0]
    qK1 = q_sorted[..., -1:] + slope_right.unsqueeze(-1) * (1.0 - taus[-1])

    taus_full = torch.cat([
        torch.zeros(1, device=q.device, dtype=torch.float32),
        taus,
        torch.ones(1, device=q.device, dtype=torch.float32),
    ])
    q_full = torch.cat([q0, q_sorted, qK1], dim=-1)
    M = q_full.shape[-1]

    y_b = y.unsqueeze(-1)
    counts = (q_full <= y_b).sum(dim=-1)
    idx_l = (counts - 1).clamp(min=0, max=M - 2)
    idx_r = idx_l + 1

    q_l = torch.gather(q_full, -1, idx_l.unsqueeze(-1)).squeeze(-1)
    q_r = torch.gather(q_full, -1, idx_r.unsqueeze(-1)).squeeze(-1)
    t_l = taus_full[idx_l]
    t_r = taus_full[idx_r]

    frac = (y - q_l) / (q_r - q_l).clamp(min=eps)
    pit = t_l + frac * (t_r - t_l)

    pit = torch.where(counts <= 0, torch.zeros_like(pit), pit)
    pit = torch.where(counts >= M, torch.ones_like(pit), pit)
    pit = pit.clamp(0.0, 1.0)
    return _flatten_valid(pit, mask)


@torch.no_grad()
def compute_target_train_std(
    train_loader: DataLoader,
    *,
    max_batches: Optional[int] = None,
) -> float:
    total = 0.0
    sq_total = 0.0
    count = 0.0
    for i, batch in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
        if not isinstance(batch, dict):
            continue
        target_idx = batch.get("target_idx", 0)
        if torch.is_tensor(target_idx):
            target_idx = int(target_idx.item()) if target_idx.numel() == 1 else int(target_idx.flatten()[0].item())
        else:
            target_idx = int(target_idx)
        y = batch.get("data_to_predict")
        if not torch.is_tensor(y):
            continue
        if y.dim() == 3:
            y = y[..., target_idx]
        y = y.to(dtype=torch.float64)

        mask = batch.get("mask_predicted_data")
        if torch.is_tensor(mask):
            if mask.dim() == 3:
                mask = mask[..., target_idx]
            mask = mask.to(dtype=torch.float64)
            total += (y * mask).sum().item()
            sq_total += ((y * y) * mask).sum().item()
            count += mask.sum().item()
        else:
            total += y.sum().item()
            sq_total += (y * y).sum().item()
            count += float(y.numel())

    if count <= 1.0:
        return 1.0
    mean = total / count
    var = max(sq_total / count - mean * mean, 0.0)
    std = var ** 0.5
    return std if std > 1e-8 else 1.0


__all__ = [
    "crps_from_samples",
    "crps_from_quantiles_pwl",
    "pit_from_samples",
    "pit_from_quantiles_pwl",
    "compute_target_train_std",
]
