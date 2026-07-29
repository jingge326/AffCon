
from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        cos_sel = cos.unsqueeze(unsqueeze_dim)
        sin_sel = sin.unsqueeze(unsqueeze_dim)
    else:
        cos_sel = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin_sel = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_sel) + (rotate_half(q) * sin_sel)
    k_embed = (k * cos_sel) + (rotate_half(k) * sin_sel)
    return q_embed, k_embed


class RotaryEmbeddingExt(nn.Module):

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 10000,
        base: int = 10000,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim,
                          2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            if seq_len is not None and seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(
                    seq_len=seq_len, device=x.device, dtype=x.dtype)
            cos = self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype)
            sin = self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype)
            return cos, sin

        pos = position_ids
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        if pos.numel() == 0:
            zero = torch.zeros_like(pos, device=x.device, dtype=x.dtype)
            return zero, zero

        if not torch.is_floating_point(pos):
            max_pos = int(pos.max().item()) + 1
            needed_len = max_pos if seq_len is None else max(max_pos, seq_len)
            if needed_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(
                    seq_len=needed_len, device=x.device, dtype=x.dtype)
            cos = self.cos_cached.to(device=x.device, dtype=x.dtype)[pos]
            sin = self.sin_cached.to(device=x.device, dtype=x.dtype)[pos]
            return cos, sin

        pos = pos.to(device=x.device, dtype=x.dtype)
        inv_freq = self.inv_freq.to(device=x.device, dtype=x.dtype)
        freqs = torch.einsum("...i,j->...ij", pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
