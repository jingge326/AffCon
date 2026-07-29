
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedder(nn.Module):

    def __init__(self, d_h: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)




class AffFlowResBlock(nn.Module):

    def __init__(self, d_h: int, mlp_ratio: int = 4, text_mode: str = "concatext"):
        super().__init__()
        self.text_mode = text_mode
        self.norm = nn.LayerNorm(d_h, elementwise_affine=False, eps=1e-6)
        if text_mode in ("concatext", "affconcat"):
            self.mlp = nn.Sequential(
                nn.Linear(2 * d_h, d_h * mlp_ratio),
                nn.SiLU(),
                nn.Linear(d_h * mlp_ratio, d_h),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_h, d_h * mlp_ratio),
                nn.SiLU(),
                nn.Linear(d_h * mlp_ratio, d_h),
            )
            if text_mode == "adaln":
                self.text_mod = nn.Sequential(
                    nn.SiLU(), nn.Linear(d_h, 3 * d_h))
            elif text_mode == "filmtext":
                self.text_mod = nn.Sequential(
                    nn.SiLU(), nn.Linear(d_h, 2 * d_h))
        time_mod_dim = 2 * d_h if text_mode == "affconcat" else 3 * d_h
        self.adaLN_time = nn.Sequential(nn.SiLU(), nn.Linear(d_h, time_mod_dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        if self.text_mode == "affconcat":
            shift_t, scale_t = self.adaLN_time(t_emb).chunk(2, dim=-1)
            if x.dim() == 3:
                shift_t = shift_t.unsqueeze(1)
                scale_t = scale_t.unsqueeze(1)
                c = c_emb.unsqueeze(1).expand_as(x)
            else:
                c = c_emb
            h = self.mlp(torch.cat([self.norm(x), self.norm(c)], dim=-1))
            h = h * (1 + scale_t) + shift_t
            return x + h
        shift_t, scale_t, gate_t = self.adaLN_time(t_emb).chunk(3, dim=-1)
        if x.dim() == 3:
            shift_t = shift_t.unsqueeze(1)
            scale_t = scale_t.unsqueeze(1)
            gate_t = gate_t.unsqueeze(1)
            c_emb_3d = c_emb.unsqueeze(1).expand_as(x)
        else:
            c_emb_3d = c_emb
        h = self.norm(x) * (1 + scale_t) + shift_t
        if self.text_mode == "concatext":
            h = self.mlp(torch.cat([h, c_emb_3d], dim=-1))
        elif self.text_mode == "adaln":
            shift_c, scale_c, gate_c = self.text_mod(c_emb).chunk(3, dim=-1)
            if x.dim() == 3:
                shift_c = shift_c.unsqueeze(1)
                scale_c = scale_c.unsqueeze(1)
                gate_c = gate_c.unsqueeze(1)
            h = h * (1 + scale_c) + shift_c
            h = self.mlp(h) * gate_c
        else:
            gamma, beta = self.text_mod(c_emb).chunk(2, dim=-1)
            if x.dim() == 3:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            h = h * (1 + gamma) + beta
            h = self.mlp(h)
        return x + gate_t * h


class VelocityNet(nn.Module):

    def __init__(
        self,
        d_ts: int,
        d_txt: int,
        d_h: Optional[int] = None,
        num_blocks: int = 6,
        mlp_ratio: int = 4,
        text_mode: str = "concatext",
    ):
        super().__init__()
        d_h = d_h or d_ts
        self.d_ts = d_ts
        self.d_h = d_h
        self.text_mode = text_mode

        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_ts, elementwise_affine=False),
            nn.Linear(d_ts, d_h),
        )

        self.time_embed = SinusoidalTimestepEmbedder(d_h)

        self.text_proj = nn.Sequential(
            nn.LayerNorm(d_txt, elementwise_affine=False),
            nn.Linear(d_txt, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )
        if text_mode == "filmtext":
            self.input_text_proj = nn.Linear(d_h, d_h)

        self.blocks = nn.ModuleList([
            AffFlowResBlock(d_h, mlp_ratio=mlp_ratio, text_mode=text_mode)
            for _ in range(num_blocks)
        ])

        self.output_proj = nn.Linear(d_h, d_ts)

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_time[-1].weight, 0)
            nn.init.constant_(block.adaLN_time[-1].bias, 0)

        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x_t)
        t_emb = self.time_embed(t)
        c_emb = self.text_proj(text_emb)
        if self.text_mode == "filmtext":
            c_in = self.input_text_proj(c_emb)
            if h.dim() == 3:
                c_in = c_in.unsqueeze(1)
            h = h + c_in

        for block in self.blocks:
            h = block(h, t_emb, c_emb)

        return self.output_proj(h)


class AffFlow(nn.Module):

    def __init__(
        self,
        d_ts: int,
        d_txt: int,
        d_h: Optional[int] = None,
        num_blocks: int = 2,
        mlp_ratio: int = 2,
        num_sampling_steps: int = 50,
        train_steps_min: int = 8,
        train_steps_max: int = 64,
        text_mode: Optional[str] = None,
        prior_type: str = "standard",
        text_concat: Optional[bool] = None,
    ):
        super().__init__()
        if train_steps_min < 1 or train_steps_max < train_steps_min:
            raise ValueError(
                "Require 1 <= train_steps_min <= train_steps_max; got "
                f"{train_steps_min}, {train_steps_max}"
            )
        self.d_ts = d_ts
        self.num_sampling_steps = num_sampling_steps
        self.train_steps_min = train_steps_min
        self.train_steps_max = train_steps_max
        if text_mode not in {"concatext", "adaln", "filmtext", "affconcat"}:
            raise ValueError(f"Unsupported AffFlow text mode: {text_mode}")
        if prior_type not in {"standard", "textprior"}:
            raise ValueError(f"Unsupported AffFlow prior type: {prior_type}")
        self.text_mode = text_mode
        self.prior_type = prior_type

        self.velocity_net = VelocityNet(
            d_ts=d_ts,
            d_txt=d_txt,
            d_h=d_h,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            text_mode=text_mode,
        )
        if prior_type == "textprior":
            prior_hidden = d_h or d_ts
            self.prior_mu = nn.Sequential(
                nn.Linear(d_txt, prior_hidden),
                nn.LayerNorm(prior_hidden),
                nn.SiLU(),
                nn.Linear(prior_hidden, d_ts),
            )
            self.prior_log_sigma = nn.Sequential(
                nn.Linear(d_txt, prior_hidden),
                nn.LayerNorm(prior_hidden),
                nn.SiLU(),
                nn.Linear(prior_hidden, d_ts),
            )

    def _sample_x0(
        self,
        text_emb: torch.Tensor,
        *,
        ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.prior_type == "standard":
            if ref is None:
                return torch.randn(text_emb.shape[0], self.d_ts, device=text_emb.device)
            return torch.randn_like(ref)

        mu = self.prior_mu(text_emb)
        log_sigma = self.prior_log_sigma(text_emb)
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-6
        if ref is None:
            eps = torch.randn_like(mu)
            return mu + sigma * eps
        if ref.dim() == 3:
            mu = mu.unsqueeze(1).expand(
                ref.shape[0], ref.shape[1], mu.shape[-1])
            sigma = sigma.unsqueeze(1).expand_as(mu)
        eps = torch.randn_like(ref)
        return mu + sigma * eps

    def forward(
        self,
        x1: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = x1.shape[0]
        device = x1.device

        x0 = self._sample_x0(text_emb, ref=x1)
        t = torch.rand(B, device=device)

        t_expand = t.unsqueeze(-1)
        if x1.dim() == 3:
            t_expand = t_expand.unsqueeze(-1)
        x_t = (1 - t_expand) * x0 + t_expand * x1

        v_target = x1 - x0

        v_pred = self.velocity_net(x_t, t, text_emb)

        return nn.functional.mse_loss(v_pred, v_target)

    def _integration_grid(self, num_steps: int, random_grid: bool) -> list[float]:
        if not random_grid:
            return [i / num_steps for i in range(num_steps + 1)]
        n = int(torch.randint(
            self.train_steps_min, self.train_steps_max + 1, (1,)).item())
        if n == 1:
            return [0.0, 1.0]
        interior = torch.rand(n - 1).sort().values.tolist()
        return [0.0] + interior + [1.0]

    def sample(
        self,
        text_emb: torch.Tensor,
        num_steps: Optional[int] = None,
        x0: Optional[torch.Tensor] = None,
        random_grid: bool = False,
    ) -> torch.Tensor:
        num_steps = num_steps or self.num_sampling_steps
        if x0 is None or self.prior_type == "textprior":
            x = self._sample_x0(text_emb)
        else:
            x = x0.to(text_emb.device)

        grid = self._integration_grid(num_steps, random_grid)
        B = text_emb.shape[0]
        for j in range(len(grid) - 1):
            t = torch.full((B,), grid[j], device=text_emb.device)
            v = self.velocity_net(x, t, text_emb)
            x = x + v * (grid[j + 1] - grid[j])

        return x
