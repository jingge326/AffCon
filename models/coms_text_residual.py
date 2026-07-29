
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TextResidualBlock(nn.Module):

    def __init__(self, d_h: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            d_h,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_h, mlp_ratio * d_h),
            nn.SiLU(),
            nn.Linear(mlp_ratio * d_h, d_h),
        )

    def forward(
        self,
        x: torch.Tensor,
        text_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() == 3:
            text_hidden = text_hidden.unsqueeze(1).expand_as(x)
        residual = self.mlp(
            torch.cat(
                [self.norm(x), self.norm(text_hidden)],
                dim=-1,
            )
        )
        return x + residual


class ResidualMLPNet(nn.Module):

    def __init__(
        self,
        d_ts: int,
        d_txt: int,
        d_h: Optional[int] = None,
        num_blocks: int = 2,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        if d_ts <= 0 or d_txt <= 0:
            raise ValueError(f"d_ts and d_txt must be positive, got {d_ts}, {d_txt}")
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if mlp_ratio < 1:
            raise ValueError(f"mlp_ratio must be >= 1, got {mlp_ratio}")

        d_h = d_h or d_ts
        if d_h <= 0:
            raise ValueError(f"d_h must be positive, got {d_h}")

        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_ts, elementwise_affine=False),
            nn.Linear(d_ts, d_h),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(d_txt, elementwise_affine=False),
            nn.Linear(d_txt, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )
        self.blocks = nn.ModuleList(
            [
                TextResidualBlock(d_h=d_h, mlp_ratio=mlp_ratio)
                for _ in range(num_blocks)
            ]
        )
        self.output_proj = nn.Linear(d_h, d_ts)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x0: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_proj(x0)
        text_hidden = self.text_proj(text_emb)
        for block in self.blocks:
            hidden = block(hidden, text_hidden)
        return self.output_proj(hidden)


class TextResidualMLP(nn.Module):

    def __init__(
        self,
        d_ts: int,
        d_txt: int,
        d_h: Optional[int] = None,
        num_blocks: int = 2,
        mlp_ratio: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        self.net = ResidualMLPNet(
            d_ts=d_ts,
            d_txt=d_txt,
            d_h=d_h,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
        )

    def sample(
        self,
        text_emb: torch.Tensor,
        x0: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        if x0 is None:
            raise ValueError("TextResidualMLP.sample requires an anchor x0")
        if x0.shape[0] != text_emb.shape[0]:
            raise ValueError(
                "x0 and text_emb batch sizes must match; got "
                f"{x0.shape[0]} and {text_emb.shape[0]}"
            )
        x0 = x0.to(device=text_emb.device)
        return x0 + self.net(x0, text_emb)
