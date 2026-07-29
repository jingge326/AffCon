from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.coms_lora import LoRASettings, apply_lora_inplace, set_backbone_eval_keep_lora_train
from models.coms_text import (
    build_affiliated_token_inputs,
    collapse_text_slots,
    crop_kv_cache_to_length,
    find_anchor_indices,
)
from models.coms_affflow import AffFlow
from models.coms_text_residual import TextResidualMLP
from models.coms_sundial import model_observed_mask, standardize_w_mask
from models.coms_timemoe import TimeMoeForPrediction

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)


@dataclass
class BackboneHandles:
    timemoe_model: TimeMoeForPrediction
    qwen_model: Optional[AutoModel]
    qwen_tokenizer: Optional[AutoTokenizer]
    d_ts: int
    d_txt: int
    input_size: int


def load_timemoe_backbone(
    timemoe_model_name: str,
    qwen_model_name: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    use_flash_attn: bool = False,
    freeze_text: bool = True,
    load_text_backbone: bool = False,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modality_routed: bool = False,
) -> BackboneHandles:
    model: TimeMoeForPrediction = TimeMoeForPrediction.from_pretrained(
        timemoe_model_name,
        trust_remote_code=True,
    )
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    if use_lora:
        apply_lora_inplace(
            model,
            backbone_name="timemoe",
            settings=LoRASettings(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                modality_routed=lora_modality_routed,
            ),
        )

    d_ts = model.config.hidden_size

    tokenizer_kwargs = {"padding_side": "left"}
    model_kwargs: Dict[str, Any] = {}
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    qwen_tokenizer: Optional[AutoTokenizer] = None
    qwen_model: Optional[AutoModel] = None
    if load_text_backbone:
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            **tokenizer_kwargs,
        )
        qwen_model = AutoModel.from_pretrained(
            qwen_model_name,
            **model_kwargs,
        )
        qwen_model.to(device)
        qwen_model.eval()
        if freeze_text:
            for p in qwen_model.parameters():
                p.requires_grad_(False)
        d_txt = qwen_model.config.hidden_size
    else:
        qwen_cfg = AutoConfig.from_pretrained(qwen_model_name)
        d_txt = qwen_cfg.hidden_size

    return BackboneHandles(
        timemoe_model=model,
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
        d_ts=d_ts,
        d_txt=d_txt,
        input_size=model.config.input_size,
    )


class TimeMoEForecaster(nn.Module):

    def __init__(
        self,
        backbones: BackboneHandles,
        horizon: Optional[int],
        cfg,
        num_channels: Optional[int] = None,
        *,
        run_mode: str = "train",
        use_lora: bool = False,
    ) -> None:
        super().__init__()
        self.timemoe = backbones.timemoe_model
        self.qwen_model = backbones.qwen_model
        self.d_ts = backbones.d_ts
        self.d_txt = backbones.d_txt
        self.hidden_size = backbones.d_ts
        self.input_size = backbones.input_size
        self.cfg = cfg
        self.run_mode = run_mode
        self.use_lora = use_lora
        self.generator_type = getattr(cfg.affcon, "generator_type", "affflow")
        if self.generator_type not in ("affflow", "residual_mlp"):
            raise ValueError(f"Unsupported generator_type: {self.generator_type}")
        self.affflow_dummy_token = getattr(
            cfg.affcon, "affflow_dummy_token", False)
        self.affflow_dummy_init = getattr(
            cfg.affcon, "affflow_dummy_init", "zero")
        if self.affflow_dummy_init not in ("zero", "randn"):
            raise ValueError(
                f"Unsupported affflow_dummy_init: {self.affflow_dummy_init}"
            )
        self.lora_backbone = self.timemoe
        self.lora_backbone_name = "timemoe"
        self.text_gate = (
            nn.Parameter(torch.zeros(()))
            if getattr(cfg.affcon, "text_gate", False) else None
        )
        self.use_conditioned_zeroshot = (
            self.run_mode == 'zeroshot'
        )

        if self.run_mode == 'train' or self.use_conditioned_zeroshot:
            if (
                self.generator_type == "residual_mlp"
                and getattr(cfg.affcon, "flow_init", "anchor") != "anchor"
            ):
                raise ValueError(
                    "residual_mlp requires flow_init='anchor' for the matched "
                    "anchor-conditioned ablation."
                )
            if (
                self.generator_type == "affflow"
                and getattr(cfg.affcon, "affflow_prior_type", "standard") == "textprior"
            ):
                raise ValueError(
                    "affflow is incompatible with affflow_prior_type='textprior': "
                    "AffFlow.sample() ignores externally-supplied x0 under "
                    "textprior, which breaks the dummy / aligned-TS x0 design."
                )
            if not getattr(self, "affflow_dummy_token", False):
                print("[affflow] forcing affflow_dummy_token=True (override)")
            self.affflow_dummy_token = True
            if getattr(cfg.train, "affflow_loss_weight", 0.0) > 0:
                print(
                    "[affflow] affflow_loss_weight is ignored; "
                    "AffFlow velocity regularization is force-skipped."
                )
            if self.generator_type == "residual_mlp":
                self.affflow = TextResidualMLP(
                    d_ts=self.d_ts,
                    d_txt=self.d_txt,
                    num_blocks=getattr(cfg.affcon, "affflow_num_blocks", 2),
                    mlp_ratio=getattr(cfg.affcon, "affflow_mlp_ratio", 2),
                )
            else:
                self.affflow = AffFlow(
                    d_ts=self.d_ts,
                    d_txt=self.d_txt,
                    num_blocks=getattr(cfg.affcon, "affflow_num_blocks", 2),
                    mlp_ratio=getattr(cfg.affcon, "affflow_mlp_ratio", 2),
                    num_sampling_steps=getattr(
                        cfg.affcon, "affflow_nfe", 50),
                    train_steps_min=getattr(cfg.affcon, "affflow_train_steps_min", 8),
                    train_steps_max=getattr(cfg.affcon, "affflow_train_steps_max", 64),
                    text_mode=getattr(cfg.affcon, "affflow_text_mode", None),
                    prior_type=getattr(cfg.affcon, "affflow_prior_type", "standard"),
                    text_concat=getattr(cfg.affcon, "affflow_text_concat", True),
                )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        observed_data = batch.get("observed_data")
        observed_mask = batch.get("observed_mask")
        data_to_predict = batch.get("data_to_predict")
        observed_tp = batch.get("observed_tp")
        tp_to_predict = batch.get("tp_to_predict")
        target_idx = batch.get("target_idx", 0)

        if observed_mask is None:
            observed_mask = torch.ones_like(observed_data)

        output_channels = data_to_predict.size(-1)
        horizon = data_to_predict.shape[1]

        x = observed_data[:, :, target_idx]
        x_anchor_mask = observed_mask[:, :, target_idx]
        missing_policy = getattr(
            self.cfg.affcon, "observed_missing_policy", "mask_aware"
        )
        x_model_mask = model_observed_mask(x_anchor_mask, missing_policy)
        x, means, stdev = standardize_w_mask(x, x_anchor_mask)
        bsz, time_steps = x.shape

        seq_vals = x.unsqueeze(-1)
        seq_mask = x_model_mask.to(dtype=torch.long)
        seq_anchor_mask = x_anchor_mask.to(dtype=torch.long)
        observed_len = time_steps

        raw_base_tp = observed_tp.to(device=seq_vals.device, dtype=torch.float32)
        base_pos = torch.arange(
            observed_len,
            device=seq_vals.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)
        future_pos = (
            base_pos[:, -1:].expand(-1, tp_to_predict.shape[1])
            + torch.arange(
                1,
                tp_to_predict.shape[1] + 1,
                device=seq_vals.device,
                dtype=torch.long,
            ).unsqueeze(0)
        )
        full_pos = torch.cat([base_pos, future_pos], dim=1)

        obs_vals = seq_vals[:, :observed_len]
        obs_mask = seq_mask[:, :observed_len]
        obs_anchor_mask = seq_anchor_mask[:, :observed_len]
        obs_pos_ids = base_pos[:, :observed_len]
        raw_obs_tp = raw_base_tp[:, :observed_len]
        text_attention_mode = getattr(
            self.cfg.affcon, "text_attention_mode", "temporal_causal"
        )
        temporal_text_restriction = (text_attention_mode == "temporal_causal")
        text_position_mode = getattr(
            self.cfg.affcon, "text_position_mode", "token_aligned"
        )
        text_injection_layout = getattr(
            self.cfg.affcon, "text_injection_layout", "distributed"
        )
        if text_attention_mode not in ("temporal_causal", "global_prefix"):
            raise ValueError(f"Unsupported text_attention_mode: {text_attention_mode}")
        if text_position_mode not in ("token_aligned", "raw_timestamp", "zero"):
            raise ValueError(f"Unsupported text_position_mode: {text_position_mode}")
        if text_injection_layout not in ("distributed", "collapsed_global"):
            raise ValueError(f"Unsupported text_injection_layout: {text_injection_layout}")
        ts_observed_position_ids = (
            raw_base_tp[:, :observed_len]
            if text_position_mode == "raw_timestamp"
            else base_pos[:, :observed_len]
        )
        ts_future_position_ids = (
            tp_to_predict.to(device=seq_vals.device, dtype=torch.float32)
            if text_position_mode == "raw_timestamp"
            else future_pos
        )
        full_pos = torch.cat([ts_observed_position_ids, ts_future_position_ids], dim=1)

        obs_embeds_raw = self.timemoe.model.embed_layer(obs_vals)

        def align_text_positions_to_ts(text_tp_raw: torch.Tensor | None,
                                       text_mask_raw: torch.Tensor | None) -> torch.Tensor | None:
            if text_tp_raw is None:
                return None
            if text_mask_raw is None:
                text_mask_bool = torch.ones(
                    text_tp_raw.shape,
                    device=text_tp_raw.device,
                    dtype=torch.bool,
                )
            else:
                text_mask_bool = text_mask_raw.to(device=text_tp_raw.device, dtype=torch.bool)
            text_ref_idx = find_anchor_indices(
                text_tp=text_tp_raw.to(device=seq_vals.device, dtype=torch.float32),
                text_mask=text_mask_bool.to(device=seq_vals.device),
                ref_tp=raw_obs_tp,
                ref_valid=obs_anchor_mask.bool(),
            )
            return ts_observed_position_ids.gather(1, text_ref_idx)

        def select_text_positions(
            text_tp_raw: torch.Tensor | None,
            text_mask_raw: torch.Tensor | None,
        ) -> torch.Tensor | None:
            if text_mask_raw is None and text_tp_raw is None:
                return None
            if text_mask_raw is None:
                text_mask_raw = torch.ones(
                    text_tp_raw.shape,
                    device=text_tp_raw.device,
                    dtype=torch.bool,
                )
            if text_position_mode == "token_aligned":
                return align_text_positions_to_ts(text_tp_raw, text_mask_raw)
            if text_position_mode == "raw_timestamp":
                if text_tp_raw is None:
                    return None
                return text_tp_raw.to(device=seq_vals.device, dtype=torch.float32)
            shape = text_mask_raw.shape
            return torch.zeros(
                shape,
                device=seq_vals.device,
                dtype=ts_observed_position_ids.dtype,
            )

        def select_text_temporal_positions(
            text_tp_raw: torch.Tensor | None,
            text_mask_raw: torch.Tensor | None,
        ) -> torch.Tensor | None:
            if text_position_mode == "zero":
                return align_text_positions_to_ts(text_tp_raw, text_mask_raw)
            return select_text_positions(text_tp_raw, text_mask_raw)

        if (self.run_mode == 'zeroshot' and (not self.use_conditioned_zeroshot)):
            obs_inputs_embeds = obs_embeds_raw
            obs_attn_mask = obs_mask.to(dtype=torch.long)
            obs_position_ids = ts_observed_position_ids
        text_aligned = batch.get("text_embeddings")
        text_tp = batch.get("text_tp")
        text_mask_raw = batch.get("text_mask")
        if text_aligned is None:
            raise ValueError("affflow mode requires text_embeddings in batch")
        if text_tp is None:
            raise ValueError("affflow mode requires text_tp for aligned x0")

        if text_mask_raw is None:
            text_mask_raw = torch.ones(
                text_aligned.shape[:2], device=text_aligned.device)
        text_pos_ids = select_text_positions(text_tp, text_mask_raw)
        text_temporal_pos_ids = select_text_temporal_positions(
            text_tp, text_mask_raw)

        affiliated_token_control = getattr(
            self.cfg.affcon, "affiliated_token_control", "none")
        B_txt, N_txt, d_txt_dim = text_aligned.shape
        d_ts = obs_embeds_raw.shape[-1]
        if affiliated_token_control == "random":
            generator = None
            if not self.training:
                generator = torch.Generator(
                    device=obs_embeds_raw.device).manual_seed(0)
            gen_tokens = torch.randn(
                (B_txt, N_txt, d_ts),
                device=obs_embeds_raw.device,
                dtype=obs_embeds_raw.dtype,
                generator=generator,
            )
        elif affiliated_token_control == "none":
            AffFlow = self.affflow
            freeze = self.cfg.train.freeze_affflow
            text_flat = text_aligned.reshape(B_txt * N_txt, d_txt_dim)
            ref_idx = find_anchor_indices(
                text_tp=text_tp.to(device=seq_vals.device, dtype=torch.float32),
                text_mask=text_mask_raw.to(device=seq_vals.device, dtype=torch.bool),
                ref_tp=raw_obs_tp,
                ref_valid=obs_anchor_mask.bool(),
            )
            x0 = obs_embeds_raw.gather(
                1, ref_idx.unsqueeze(-1).expand(-1, -1, d_ts)
            ).reshape(B_txt * N_txt, d_ts)
            if getattr(self.cfg.affcon, "flow_init", "anchor") == "noise":
                x0 = torch.randn_like(x0)
            train_affflow = self.training and not freeze
            grad_ctx = torch.enable_grad() if train_affflow else torch.no_grad()
            with grad_ctx:
                gen_tokens_flat = AffFlow.sample(
                    text_flat, x0=x0, random_grid=(train_affflow and getattr(self.cfg.affcon, "affflow_train_random_grid", True)))
            gen_tokens = gen_tokens_flat.reshape(B_txt, N_txt, -1)
        else:
            raise ValueError(
                f"Unsupported affiliated_token_control: {affiliated_token_control}"
            )

        if text_injection_layout == "collapsed_global":
            orig_gen_tokens = gen_tokens
            orig_text_mask_raw = text_mask_raw
            gen_tokens, text_mask_raw, text_pos_ids = collapse_text_slots(
                orig_gen_tokens,
                orig_text_mask_raw,
                text_pos_ids,
                obs_tp=ts_observed_position_ids,
            )
            _, _, text_temporal_pos_ids = collapse_text_slots(
                orig_gen_tokens,
                orig_text_mask_raw,
                text_temporal_pos_ids,
                obs_tp=ts_observed_position_ids,
            )
        obs_inputs_embeds, obs_position_ids, obs_attn_mask_4d, obs_attn_mask, n_text_tokens = build_affiliated_token_inputs(
            text_tokens=gen_tokens,
            text_tp=text_pos_ids,
            text_mask=text_mask_raw,
            obs_tokens=obs_embeds_raw,
            obs_tp=ts_observed_position_ids,
            obs_mask=obs_mask,
            text_temporal_positions=text_temporal_pos_ids,
            obs_temporal_positions=ts_observed_position_ids,
            temporal_text_restriction=temporal_text_restriction,
            text_gate=self.text_gate,
        )

        def build_embeds_and_mask(values, mask):
            seq_len = values.size(1)
            if seq_len <= full_pos.size(1):
                pos_ids_vals = full_pos[:, :seq_len]
            else:
                extra = seq_len - full_pos.size(1)
                last_pos = full_pos[:, -1:].to(values.device)
                extra_pos = last_pos + torch.arange(
                    1, extra + 1, device=values.device, dtype=full_pos.dtype
                ).unsqueeze(0)
                pos_ids_vals = torch.cat([full_pos, extra_pos], dim=1)
            pred_embeds = self.timemoe.model.embed_layer(
                values[:, observed_len:, :]
            )
            pred_attn_mask = mask[:, observed_len:].to(dtype=torch.long)
            pred_position_ids = pos_ids_vals[:, observed_len:]

            inputs_embeds = torch.cat(
                (obs_inputs_embeds, pred_embeds), dim=1)
            attn_mask = torch.cat((obs_attn_mask, pred_attn_mask), dim=1)
            position_ids = torch.cat(
                (obs_position_ids, pred_position_ids), dim=1)

            return inputs_embeds, attn_mask, position_ids

        preds_chunks = []
        remaining = horizon
        supported = sorted(self.timemoe.config.horizon_lengths)
        grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
        past_key_values = None
        cached_tokens = 0

        while remaining > 0:
            prev_cached = cached_tokens
            h_sel = max((h for h in supported if h <= remaining),
                        default=supported[0])
            inputs_embeds, attn_mask, position_ids = build_embeds_and_mask(
                seq_vals, seq_mask)
            chunk_inputs = inputs_embeds[:, cached_tokens:, :]
            chunk_position_ids = position_ids[:, cached_tokens:]
            if chunk_inputs.size(1) == 0:
                raise RuntimeError(
                    "No new tokens to process but remaining > 0.")

            if prev_cached == 0:
                attn_mask = obs_attn_mask_4d

            step_n_text_tokens = n_text_tokens if prev_cached == 0 else 0

            with grad_ctx:
                outputs = self.timemoe(
                    input_ids=None,
                    inputs_embeds=chunk_inputs,
                    attention_mask=attn_mask,
                    position_ids=chunk_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    max_horizon_length=remaining,
                    n_text_tokens=step_n_text_tokens,
                )

            preds = outputs.logits
            past_key_values = outputs.past_key_values
            cached_tokens = inputs_embeds.size(1)
            if prev_cached == 0:
                step_ids = torch.arange(
                    seq_mask.size(1), device=seq_mask.device, dtype=seq_mask.dtype
                ).unsqueeze(0).expand_as(seq_mask)
                last_steps = torch.where(
                    seq_mask > 0, step_ids, torch.full_like(step_ids, -1)
                ).max(dim=1).values.clamp(min=0)
                token_indices = last_steps + n_text_tokens
                local_indices = (token_indices - prev_cached).clamp(min=0)
                gather_idx = local_indices.view(-1, 1, 1).expand(
                    preds.size(0), 1, preds.size(-1)
                )
                preds_last = preds.gather(1, gather_idx).squeeze(1).reshape(
                    preds.shape[0], h_sel, self.input_size
                )
            else:
                preds_last = preds[:, -1, :].reshape(
                    preds.shape[0], h_sel, self.input_size
                )
            preds_chunks.append(preds_last)

            seq_vals = torch.cat([seq_vals, preds_last.detach()], dim=1)
            seq_mask = torch.cat(
                [
                    seq_mask,
                    torch.ones(
                        (seq_mask.shape[0], h_sel),
                        device=seq_mask.device,
                        dtype=seq_mask.dtype,
                    ),
                ],
                dim=1,
            )
            remaining -= h_sel

        y_hat = torch.cat(preds_chunks, dim=1)[:, :horizon, 0]

        y_hat = y_hat * stdev + means

        full_pred = torch.zeros(
            y_hat.shape[0],
            horizon,
            output_channels,
            device=y_hat.device,
            dtype=y_hat.dtype,
        )
        full_pred[:, :, target_idx] = y_hat

        mask_pred = None
        mask_forecast_target = None
        if batch.get("mask_predicted_data") is not None:
            mask_pred_all = batch.get("mask_predicted_data")
            mask_pred = mask_pred_all[:, :, target_idx]
            mask_forecast_target = torch.zeros_like(mask_pred_all)
            mask_forecast_target[:, :, target_idx] = mask_pred
        target = None
        if torch.is_tensor(data_to_predict):
            target = data_to_predict[:, :, target_idx]

        loss = None
        if target is not None:
            if mask_pred is not None:
                denom = mask_pred.sum().clamp_min(1.0)
                diff = (y_hat - target) * mask_pred
                loss = (diff * diff).sum() / denom
            else:
                diff = y_hat - target
                loss = (diff * diff).mean()

        affflow_loss = None

        return {
            "y_hat": full_pred,
            "loss": loss,
            "affflow_loss": affflow_loss,
            "target_idx": target_idx,
            "mask_forecast_target": mask_forecast_target,
        }

    def forward_text_only(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = batch["text_embeddings"]
        gt_series = batch["gt_series"]
        B = text_embeddings.shape[0]
        N_txt = text_embeddings.shape[1]
        target_len = gt_series.shape[1]

        dummy_token = None
        use_token_noise_x0 = (
            getattr(self.cfg.affcon, "flow_init", "anchor") == "noise"
        )
        if self.affflow_dummy_token:
            if self.affflow_dummy_init == "randn":
                generator = None
                if not self.training:
                    generator = torch.Generator(
                        device=text_embeddings.device).manual_seed(0)
                dummy_vals = torch.randn(
                    B,
                    1,
                    self.input_size,
                    device=text_embeddings.device,
                    dtype=gt_series.dtype,
                    generator=generator,
                )
            else:
                dummy_vals = torch.zeros(
                    B,
                    1,
                    self.input_size,
                    device=text_embeddings.device,
                    dtype=gt_series.dtype,
                )
            dummy_token = self.timemoe.model.embed_layer(dummy_vals)

        affiliated_token_control = getattr(
            self.cfg.affcon, "affiliated_token_control", "none")
        if dummy_token is None:
            raise RuntimeError("affflow requires a dummy token for x0")
        d_ts = dummy_token.shape[-1]
        if affiliated_token_control == "random":
            generator = None
            if not self.training:
                generator = torch.Generator(
                    device=text_embeddings.device).manual_seed(0)
            text_proj = torch.randn(
                (B, N_txt, d_ts),
                device=dummy_token.device,
                dtype=dummy_token.dtype,
                generator=generator,
            )
        elif affiliated_token_control == "none":
            x0 = (
                dummy_token.expand(B, N_txt, d_ts)
                .reshape(B * N_txt, d_ts)
                .to(dtype=text_embeddings.dtype)
            )
            if use_token_noise_x0:
                generator = None
                if not self.training:
                    generator = torch.Generator(
                        device=text_embeddings.device).manual_seed(0)
                x0 = torch.randn(
                    x0.shape,
                    device=x0.device,
                    dtype=x0.dtype,
                    generator=generator,
                )
            text_flat = text_embeddings.reshape(B * N_txt, -1)
            freeze = getattr(self.cfg.train, "freeze_affflow", False)
            train_affflow = self.training and not freeze
            grad_ctx = torch.enable_grad() if train_affflow else torch.no_grad()
            with grad_ctx:
                text_proj = self.affflow.sample(
                    text_flat, x0=x0, random_grid=(train_affflow and getattr(self.cfg.affcon, "affflow_train_random_grid", True))
                ).reshape(B, N_txt, d_ts)
        else:
            raise ValueError(
                f"Unsupported affiliated_token_control: {affiliated_token_control}"
            )

        prompt_embeds = text_proj
        if dummy_token is not None:
            prompt_embeds = torch.cat(
                [text_proj, dummy_token.to(dtype=text_proj.dtype)], dim=1)

        supported = sorted(self.timemoe.config.horizon_lengths)
        remaining = target_len
        past_key_values = None
        preds_chunks = []

        current_embeds = prompt_embeds
        pos_counter = 0
        total_tokens = 0
        dummy_prefill_pending = True
        prefill_done = False
        grad_ctx = torch.enable_grad() if self.training else torch.no_grad()

        while remaining > 0:
            h_sel = max(
                (h for h in supported if h <= remaining),
                default=supported[0],
            )
            chunk_len = current_embeds.shape[1]
            if dummy_prefill_pending:
                position_ids = torch.zeros(
                    B,
                    chunk_len,
                    device=text_proj.device,
                    dtype=torch.long,
                )
            else:
                position_ids = torch.arange(
                    pos_counter, pos_counter + chunk_len,
                    device=text_proj.device, dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)

            total_tokens += chunk_len
            attn_mask = torch.ones(
                B, total_tokens,
                device=text_proj.device, dtype=torch.long,
            )

            step_n_text_tokens = N_txt if not prefill_done else 0

            with grad_ctx:
                outputs = self.timemoe(
                    input_ids=None,
                    inputs_embeds=current_embeds,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    max_horizon_length=remaining,
                    n_text_tokens=step_n_text_tokens,
                )
            prefill_done = True

            past_key_values = outputs.past_key_values
            preds = outputs.logits
            preds_last = preds[:, -1, :].reshape(B, h_sel, self.input_size)
            preds_chunks.append(preds_last)

            if dummy_prefill_pending:
                keep_tokens = text_proj.shape[1]
                past_key_values = crop_kv_cache_to_length(
                    past_key_values,
                    keep_tokens,
                )
                total_tokens = keep_tokens
                pos_counter = keep_tokens
                dummy_prefill_pending = False
            else:
                pos_counter += chunk_len

            current_embeds = self.timemoe.model.embed_layer(preds_last.detach())
            remaining -= h_sel

        y_hat = torch.cat(preds_chunks, dim=1)[:, :target_len, 0]

        if "lengths" in batch:
            lengths = batch["lengths"].to(gt_series.device)
            mask = (torch.arange(target_len, device=gt_series.device).unsqueeze(0)
                    < lengths.unsqueeze(1)).to(gt_series.dtype)
        else:
            mask = torch.ones_like(gt_series)
        gt_z, _, _ = standardize_w_mask(gt_series, mask)
        diff = (y_hat - gt_z) * mask
        loss = (diff * diff).sum() / mask.sum().clamp_min(1.0)

        return {"y_hat": y_hat, "loss": loss}

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, Any],
        horizon: Optional[int] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        return self.forward(batch, horizon=horizon)

    def train(self, mode: bool = True):
        super().train(mode)
        set_backbone_eval_keep_lora_train(self.timemoe, mode)
        if self.qwen_model is not None:
            self.qwen_model.eval()
        return self
