
from __future__ import annotations

import torch
from typing import Any, Tuple


def crop_kv_cache_to_length(past_key_values: Any, keep_tokens: int) -> Any:
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop"):
        past_key_values.crop(keep_tokens)
        return past_key_values

    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        for idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[idx] = past_key_values.key_cache[idx][
                ..., :keep_tokens, :
            ]
            past_key_values.value_cache[idx] = past_key_values.value_cache[idx][
                ..., :keep_tokens, :
            ]
        if hasattr(past_key_values, "_seen_tokens"):
            past_key_values._seen_tokens = min(
                getattr(past_key_values, "_seen_tokens", keep_tokens),
                keep_tokens,
            )
        return past_key_values

    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(
                t[..., :keep_tokens, :]
                if torch.is_tensor(t) and t.dim() >= 3
                else t
                for t in layer
            )
            for layer in past_key_values
        )
    if isinstance(past_key_values, list):
        return [
            tuple(
                t[..., :keep_tokens, :]
                if torch.is_tensor(t) and t.dim() >= 3
                else t
                for t in layer
            )
            for layer in past_key_values
        ]

    raise TypeError(f"Unsupported KV cache type: {type(past_key_values)!r}")



def collapse_text_slots(
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    text_tp: torch.Tensor | None,
    obs_tp: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    text_mask_bool = text_mask.to(device=text_tokens.device, dtype=torch.bool)
    weights = text_mask_bool.unsqueeze(-1).to(dtype=text_tokens.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    collapsed_tokens = (text_tokens * weights).sum(dim=1, keepdim=True) / denom
    collapsed_mask = text_mask_bool.any(dim=1, keepdim=True)

    collapsed_tp = None
    if text_tp is not None:
        if obs_tp is not None:
            collapsed_tp = obs_tp[:, :1].to(device=text_tokens.device, dtype=text_tp.dtype)
        else:
            inf = torch.full_like(text_tp, float("inf"))
            earliest = torch.where(text_mask_bool, text_tp, inf).min(dim=1).values
            earliest = torch.where(
                collapsed_mask.squeeze(1),
                earliest,
                torch.zeros_like(earliest),
            )
            collapsed_tp = earliest.unsqueeze(1)
    elif obs_tp is not None:
        collapsed_tp = obs_tp[:, :1].to(device=text_tokens.device)

    return collapsed_tokens, collapsed_mask, collapsed_tp





def build_temporal_causal_4d_mask(
    text_positions: torch.Tensor,
    ts_positions: torch.Tensor,
    text_validity: torch.Tensor,
    ts_validity: torch.Tensor,
    dtype: torch.dtype,
    causal: bool = True,
    temporal_text_restriction: bool = True,
) -> torch.Tensor:
    B = text_positions.shape[0]
    N_txt = text_positions.shape[1]
    N_ts = ts_positions.shape[1]
    seq_len = N_txt + N_ts
    device = text_positions.device

    full_validity = torch.cat(
        [text_validity, ts_validity], dim=1)
    key_padding = ~full_validity.bool()

    if causal:
        causal_tri = torch.triu(
            torch.full((seq_len, seq_len), float(
                '-inf'), device=device, dtype=dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        mask_4d = causal_tri.expand(B, 1, -1, -1).clone()

        if temporal_text_restriction:
            ts_pos_q = ts_positions.unsqueeze(2)
            txt_pos_k = text_positions.unsqueeze(1)
            temporal_block = (txt_pos_k > ts_pos_q)
            mask_4d[:, 0, N_txt:, :N_txt].masked_fill_(
                temporal_block, float('-inf'))
    else:
        mask_4d = torch.zeros(B, 1, seq_len, seq_len,
                              device=device, dtype=dtype)

    mask_4d.masked_fill_(key_padding[:, None, None, :], float('-inf'))

    query_padding = key_padding[:, None, :, None]
    mask_4d.masked_fill_(query_padding, 0.0)

    return mask_4d



def build_affiliated_token_inputs(
    text_tokens: torch.Tensor,
    text_tp: torch.Tensor,
    text_mask: torch.Tensor,
    obs_tokens: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_mask: torch.Tensor,
    text_temporal_positions: torch.Tensor | None = None,
    obs_temporal_positions: torch.Tensor | None = None,
    causal: bool = True,
    invert_mask: bool = False,
    temporal_text_restriction: bool = True,
    text_gate: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    text_tokens = text_tokens * text_mask.unsqueeze(-1).float()

    inputs_embeds = torch.cat([text_tokens, obs_tokens], dim=1)
    position_ids = torch.cat([text_tp, obs_tp], dim=1)
    if text_temporal_positions is None:
        text_temporal_positions = text_tp
    if obs_temporal_positions is None:
        obs_temporal_positions = obs_tp

    attn_mask_4d = build_temporal_causal_4d_mask(
        text_positions=text_temporal_positions,
        ts_positions=obs_temporal_positions,
        text_validity=text_mask.float(),
        ts_validity=obs_mask.float(),
        dtype=obs_tokens.dtype,
        causal=causal,
        temporal_text_restriction=temporal_text_restriction,
    )

    N_txt_gate = text_tokens.shape[1]
    if text_gate is not None and N_txt_gate > 0:
        gate_cols = torch.zeros_like(attn_mask_4d)
        gate_cols[:, :, N_txt_gate:, :N_txt_gate] = 1.0
        attn_mask_4d = attn_mask_4d + gate_cols * text_gate.to(attn_mask_4d.dtype)

    attn_mask_2d = torch.cat([
        text_mask.to(dtype=torch.long),
        obs_mask.to(dtype=torch.long),
    ], dim=1)
    if invert_mask:
        attn_mask_2d = ~attn_mask_2d.bool()

    n_text_tokens = text_tokens.shape[1]
    return inputs_embeds, position_ids, attn_mask_4d, attn_mask_2d, n_text_tokens



def find_anchor_indices(
    text_tp: torch.Tensor,
    text_mask: torch.Tensor,
    ref_tp: torch.Tensor,
    ref_valid: torch.Tensor,
) -> torch.Tensor:
    B, N_txt = text_tp.shape
    N_ref = ref_tp.shape[1]
    device = text_tp.device

    text_tp_exp = text_tp.unsqueeze(-1)
    ref_tp_exp = ref_tp.unsqueeze(1)
    ref_valid_exp = ref_valid.bool().unsqueeze(1)

    valid_assignment = (ref_tp_exp >= text_tp_exp) & ref_valid_exp

    large_val = N_ref + 1
    indices = (
        torch.arange(N_ref, device=device)
        .view(1, 1, N_ref)
        .expand(B, N_txt, N_ref)
    )
    masked_indices = torch.where(valid_assignment, indices, large_val)
    first_valid_idx = masked_indices.argmin(dim=-1)

    any_valid = valid_assignment.any(dim=-1)
    ref_indices = torch.arange(N_ref, device=device).view(
        1, N_ref).expand(B, N_ref)
    last_valid_ref = (
        torch.where(ref_valid.bool(), ref_indices, -1)
        .max(dim=-1)
        .values
        .clamp(min=0)
    )
    last_valid_ref = last_valid_ref.unsqueeze(1).expand(B, N_txt)

    first_valid_idx = torch.where(any_valid, first_valid_idx, last_valid_ref)

    return first_valid_idx
