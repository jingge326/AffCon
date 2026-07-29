
from __future__ import annotations

from dataclasses import dataclass
import os
import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timesfm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from timesfm.torch import util as timesfm_util
from transformers import AutoConfig, Cache, DynamicCache

from models.coms_lora import LoRASettings, apply_lora_inplace
from models.coms_affflow import AffFlow
from models.coms_text import (
    build_affiliated_token_inputs,
    collapse_text_slots,
    crop_kv_cache_to_length,
    find_anchor_indices,
)
from models.coms_sundial import model_observed_mask, standardize_w_mask
from models.coms_timesfm import Transformer as PatchedTransformer


@dataclass
class BackboneHandles:
    inner_module: nn.Module
    d_ts: int
    d_txt: int


@dataclass
class RevinEncoded:
    sequence_embeddings: torch.Tensor
    patch_level_mask: torch.Tensor
    running_stats: Dict[str, torch.Tensor]


def load_timesfm_backbone(
    timesfm_model_name: str,
    qwen_model_name: str,
    device: torch.device,
) -> BackboneHandles:
    model_cls = timesfm.TimesFM_2p5_200M_torch
    if os.path.isdir(timesfm_model_name):
        weights_path = os.path.join(
            timesfm_model_name, model_cls.WEIGHTS_FILENAME)
    else:
        weights_path = hf_hub_download(
            repo_id=timesfm_model_name,
            filename=model_cls.WEIGHTS_FILENAME,
        )
    model = model_cls(torch_compile=False)
    state_dict = load_file(weights_path, device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    inner = model.model
    _install_patched_timesfm_layers(inner)
    inner.device = device
    inner.to(device)
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)

    qwen_cfg = AutoConfig.from_pretrained(qwen_model_name)

    return BackboneHandles(
        inner_module=inner,
        d_ts=inner.md,
        d_txt=qwen_cfg.hidden_size,
    )


def _install_patched_timesfm_layers(inner: nn.Module) -> None:
    patched_layers = []
    for layer in inner.stacked_xf:
        if isinstance(layer, PatchedTransformer):
            patched = layer
        else:
            patched = PatchedTransformer(layer.config)
            patched.load_state_dict(layer.state_dict(), strict=True)
        patched_layers.append(patched)
    inner.stacked_xf = nn.ModuleList(patched_layers)

    forward_params = inspect.signature(inner.stacked_xf[0].forward).parameters
    required = {"position_ids", "attn_mask_4d", "layer_idx", "n_text_tokens"}
    missing = required.difference(forward_params)
    if missing:
        raise TypeError(
            "TimesFM AffCon layer patch did not install required forward kwargs: "
            f"{sorted(missing)}"
        )


class TimesFMForecaster(nn.Module):

    def __init__(
        self,
        backbones: BackboneHandles,
        horizon: Optional[int],
        cfg,
        run_mode: str,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modality_routed: bool = False,
    ) -> None:
        super().__init__()
        self.timesfm_module = backbones.inner_module
        self.d_ts = backbones.d_ts
        self.d_txt = backbones.d_txt
        self.cfg = cfg
        self.run_mode = run_mode
        self.patch_position_mode = getattr(
            cfg.affcon, "patch_position_mode", "contiguous")
        self.text_attention_mode = getattr(
            cfg.affcon, "text_attention_mode", "temporal_causal")
        self.text_position_mode = getattr(
            cfg.affcon, "text_position_mode", "token_aligned")
        self.text_injection_layout = getattr(
            cfg.affcon, "text_injection_layout", "distributed")
        self.force_flip_invariance = getattr(
            cfg.affcon, "force_flip_invariance", True)
        self.timesfm_infer_is_positive = getattr(
            cfg.affcon, "timesfm_infer_is_positive", False)
        self.timesfm_decoupled_flip = getattr(
            cfg.affcon, "timesfm_decoupled_flip", False)
        self.affflow_dummy_token = getattr(
            cfg.affcon, "affflow_dummy_token", False)
        self.affflow_dummy_init = getattr(
            cfg.affcon, "affflow_dummy_init", "zero")
        if self.affflow_dummy_init not in ("zero", "randn"):
            raise ValueError(
                f"Unsupported affflow_dummy_init: {self.affflow_dummy_init}"
            )
        self.use_lora = use_lora
        self.use_conditioned_zeroshot = (
            self.run_mode == 'zeroshot'
        )


        self.input_patch_len = self.timesfm_module.p
        self.output_patch_len = self.timesfm_module.o
        self.decode_stride_patches = self.timesfm_module.m
        self.num_q = self.timesfm_module.q
        self.decode_index = self.timesfm_module.aridx
        self.q_levels = torch.tensor(
            list(self.timesfm_module.config.quantiles), dtype=torch.float32)
        self.context_limit = self.timesfm_module.config.context_limit
        self.lora_backbone = self.timesfm_module
        self.lora_backbone_name = "timesfm"
        self.text_gate = (
            nn.Parameter(torch.zeros(()))
            if getattr(cfg.affcon, "text_gate", False) else None
        )
        if use_lora:
            apply_lora_inplace(
                self.timesfm_module,
                backbone_name="timesfm",
                settings=LoRASettings(
                    r=lora_r,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    modality_routed=lora_modality_routed,
                ),
            )

        self.use_affflow_adapter = False
        if getattr(cfg.affcon, "affflow_prior_type", "standard") == "textprior":
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
        self.use_affflow_adapter = getattr(
            cfg.affcon, "affflow_adapter", False)
        if self.use_affflow_adapter:
            self.affflow_adapter = nn.Linear(self.d_ts, self.d_ts)

    def _revin_encode(
        self,
        patched_inputs: torch.Tensor,
        patched_masks: torch.Tensor,
        model_patched_masks: Optional[torch.Tensor] = None,
        running_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if model_patched_masks is None:
            model_patched_masks = patched_masks
        batch_size = patched_inputs.shape[0]
        if running_stats is None:
            n = torch.zeros(batch_size, device=patched_inputs.device)
            mu = torch.zeros(batch_size, device=patched_inputs.device)
            sigma = torch.zeros(batch_size, device=patched_inputs.device)
        else:
            n = running_stats["last_n"]
            mu = running_stats["last_mu"]
            sigma = running_stats["last_sigma"]

        patch_mu = []
        patch_sigma = []
        for i in range(patched_inputs.shape[1]):
            (n, mu, sigma), _ = timesfm_util.update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        mu_seq = torch.stack(patch_mu, dim=1)
        sigma_seq = torch.stack(patch_sigma, dim=1)
        normed_inputs = timesfm_util.revin(
            patched_inputs,
            mu_seq,
            sigma_seq,
            reverse=False,
        )
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
        tokenizer_inputs = torch.cat(
            [normed_inputs, model_patched_masks.to(normed_inputs.dtype)],
            dim=-1,
        )
        input_embeddings = self.timesfm_module.tokenizer(tokenizer_inputs)
        stats = {
            "mu": mu_seq,
            "sigma": sigma_seq,
            "last_n": n,
            "last_mu": mu,
            "last_sigma": sigma,
        }
        return input_embeddings, model_patched_masks.all(dim=-1), stats

    def _revin_decode(
        self,
        normed_out: torch.Tensor,
        stats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return timesfm_util.revin(
            normed_out,
            stats["mu"],
            stats["sigma"],
            reverse=True,
        )

    def _tokenize_standardized_patches(
        self,
        patched_inputs: torch.Tensor,
    ) -> torch.Tensor:
        patch_mask = torch.zeros_like(patched_inputs, dtype=torch.bool)
        tokenizer_inputs = torch.cat(
            [patched_inputs, patch_mask.to(patched_inputs.dtype)],
            dim=-1,
        )
        return self.timesfm_module.tokenizer(tokenizer_inputs)

    def _prepare_context(
        self,
        context: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.shape[-1] > self.context_limit:
            context = context[:, -self.context_limit:]
            context_valid = context_valid[:, -self.context_limit:]

        pad_len = (-context.shape[-1]) % self.input_patch_len
        if pad_len > 0:
            context = F.pad(context, (pad_len, 0), value=0.0)
            context_valid = F.pad(context_valid, (pad_len, 0), value=False)

        pad_mask = ~context_valid
        inputs = torch.where(pad_mask, torch.zeros_like(context), context)
        return inputs, pad_mask

    def _causal_cache_attention_mask(
        self,
        valid_mask: torch.Tensor,
        query_len: int,
    ) -> torch.Tensor:
        total_len = valid_mask.shape[1]
        past_len = total_len - query_len
        kv_idx = torch.arange(total_len, device=valid_mask.device)
        q_idx = torch.arange(
            past_len, total_len, device=valid_mask.device)
        causal = q_idx[:, None] >= kv_idx[None, :]
        return (
            causal.unsqueeze(0).unsqueeze(0)
            & valid_mask[:, None, None, :].bool()
        )

    def _backbone_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        n_text_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Cache]]:
        batch_size, seq_len, _ = inputs_embeds.shape

        attn_mask_4d = None
        patch_mask = torch.zeros(
            batch_size,
            seq_len,
            device=inputs_embeds.device,
            dtype=torch.bool,
        )
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attn_mask_4d = attention_mask
            else:
                patch_mask = ~attention_mask.to(
                    device=inputs_embeds.device, dtype=torch.bool
                )

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
                past_len = 0
            else:
                if not isinstance(past_key_values, Cache):
                    past_key_values = DynamicCache.from_legacy_cache(
                        past_key_values)
                past_len = past_key_values.get_seq_length(0)

            if position_ids is None:
                position_ids = torch.arange(
                    past_len,
                    past_len + seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.float32,
                ).unsqueeze(0).expand(batch_size, -1)

            if attn_mask_4d is None:
                if attention_mask is None:
                    valid_mask = torch.ones(
                        batch_size,
                        past_len + seq_len,
                        device=inputs_embeds.device,
                        dtype=torch.bool,
                    )
                elif attention_mask.shape[1] == past_len + seq_len:
                    valid_mask = attention_mask.to(
                        device=inputs_embeds.device, dtype=torch.bool)
                else:
                    past_valid = torch.ones(
                        batch_size,
                        past_len,
                        device=inputs_embeds.device,
                        dtype=torch.bool,
                    )
                    valid_mask = torch.cat(
                        [
                            past_valid,
                            attention_mask.to(
                                device=inputs_embeds.device,
                                dtype=torch.bool,
                            ),
                        ],
                        dim=1,
                    )
                attn_mask_4d = self._causal_cache_attention_mask(
                    valid_mask=valid_mask,
                    query_len=seq_len,
                )

        output_embeddings = inputs_embeds
        for i, layer in enumerate(self.timesfm_module.stacked_xf):
            output_embeddings, _ = layer(
                output_embeddings,
                patch_mask,
                past_key_values if use_cache else None,
                position_ids=position_ids,
                attn_mask_4d=attn_mask_4d,
                layer_idx=i,
                n_text_tokens=n_text_tokens,
            )

        output_ts = self.timesfm_module.output_projection_point(
            output_embeddings)
        return output_embeddings, output_ts, past_key_values if use_cache else None

    def _sequence_positions(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start: int = 0,
    ) -> torch.Tensor:
        return (
            torch.arange(start, start + seq_len,
                         device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

    def _prepare_prefill_prefix(
        self,
        input_embeddings: torch.Tensor,
        patch_level_mask: torch.Tensor,
        text_embeds: Optional[torch.Tensor],
        text_invalid_mask: Optional[torch.Tensor],
        text_tp: Optional[torch.Tensor],
        text_temporal_positions: Optional[torch.Tensor],
        context_patch_positions: Optional[torch.Tensor],
        context_temporal_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        prefix_len = 0 if text_embeds is None else text_embeds.shape[1]
        batch_size, num_obs = input_embeddings.shape[:2]

        if prefix_len == 0 or text_invalid_mask is None:
            valid_mask = (~patch_level_mask).long()
            return input_embeddings, context_patch_positions, None, valid_mask, 0

        if text_tp is None or context_patch_positions is None:
            text_tp = self._sequence_positions(
                batch_size=batch_size,
                seq_len=prefix_len,
                device=input_embeddings.device,
            )
            context_patch_positions = self._sequence_positions(
                batch_size=batch_size,
                seq_len=num_obs,
                device=input_embeddings.device,
                start=prefix_len,
            )
        if context_temporal_positions is None:
            context_temporal_positions = context_patch_positions

        inputs_embeds, position_ids, attn_mask_4d, valid_mask, n_text_tokens = build_affiliated_token_inputs(
            text_tokens=text_embeds,
            text_tp=text_tp,
            text_mask=~text_invalid_mask,
            obs_tokens=input_embeddings,
            obs_tp=context_patch_positions,
            obs_mask=~patch_level_mask,
            text_temporal_positions=text_temporal_positions,
            obs_temporal_positions=context_temporal_positions,
            causal=True,
            invert_mask=False,
            temporal_text_restriction=(self.text_attention_mode == "temporal_causal"),
            text_gate=getattr(self, "text_gate", None),
        )
        return inputs_embeds, position_ids, attn_mask_4d, valid_mask, n_text_tokens

    def _prepare_conditioning_tokens(
        self,
        batch: Dict[str, Any],
        num_channels: int,
        batch_size_flat: int,
        device: torch.device,
        dtype: torch.dtype,
        affflow_x0: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        text_embeddings = batch.get("text_embeddings")
        if text_embeddings is None:
            return None, None, None

        text_mask = batch.get("text_mask")
        if text_mask is None:
            text_mask = torch.ones(
                text_embeddings.shape[:2],
                device=text_embeddings.device,
                dtype=torch.bool,
            )
        else:
            text_mask = text_mask.to(dtype=torch.bool)

        text_tp = batch.get("text_tp")

        batch_txt, num_text, txt_dim = text_embeddings.shape
        if affflow_x0 is None:
            raise RuntimeError("affflow requires aligned context x0")
        x0 = affflow_x0.reshape(batch_txt * num_text, -1).to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        affiliated_token_control = getattr(
            self.cfg.affcon, "affiliated_token_control", "none")
        if affiliated_token_control == "random":
            generator = None
            if not self.training:
                generator = torch.Generator(
                    device=x0.device).manual_seed(0)
            text_tokens = torch.randn(
                (batch_txt, num_text, x0.shape[-1]),
                device=x0.device,
                dtype=x0.dtype,
                generator=generator,
            )
        elif affiliated_token_control == "none":
            AffFlow = self.affflow
            text_flat = text_embeddings.reshape(batch_txt * num_text, txt_dim)
            freeze = getattr(self.cfg.train, "freeze_affflow", False)
            train_affflow = self.training and not freeze
            grad_ctx = torch.enable_grad() if train_affflow else torch.no_grad()
            with grad_ctx:
                text_tokens = AffFlow.sample(
                    text_flat, x0=x0, random_grid=(train_affflow and getattr(self.cfg.affcon, "affflow_train_random_grid", True))
                ).reshape(batch_txt, num_text, -1)
            if self.use_affflow_adapter:
                text_tokens = self.affflow_adapter(text_tokens)
        else:
            raise ValueError(
                f"Unsupported affiliated_token_control: {affiliated_token_control}"
            )

        text_tokens = text_tokens.to(device=device, dtype=dtype)
        text_tokens = text_tokens * \
            text_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)

        text_tokens = (
            text_tokens.unsqueeze(1)
            .expand(-1, num_channels, -1, -1)
            .reshape(batch_size_flat, text_tokens.shape[1], self.d_ts)
        )
        text_invalid_mask = (
            (~text_mask)
            .unsqueeze(1)
            .expand(-1, num_channels, -1)
            .reshape(batch_size_flat, -1)
            .to(device=device)
        )

        if text_tokens.shape[1] == 0 or (~text_invalid_mask).sum() == 0:
            return None, None, None

        text_tp_flat = None
        if text_tp is not None:
            text_tp_flat = (
                text_tp.float()
                .unsqueeze(1)
                .expand(-1, num_channels, -1)
                .reshape(batch_size_flat, -1)
                .to(device=device)
            )

        return text_tokens, text_invalid_mask, text_tp_flat

    def _decode_text_only(
        self,
        horizon: int,
        text_embeds: torch.Tensor,
        text_invalid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_text, _ = text_embeds.shape
        if text_invalid_mask is None:
            text_valid_mask = torch.ones(
                batch_size,
                num_text,
                device=text_embeds.device,
                dtype=torch.bool,
            )
        else:
            text_valid_mask = ~text_invalid_mask.to(
                device=text_embeds.device, dtype=torch.bool)

        text_embeds = text_embeds * text_valid_mask.unsqueeze(-1).to(
            dtype=text_embeds.dtype)
        dummy_prefill = self.affflow_dummy_token and num_text > 1
        prefill_position_ids = None
        if dummy_prefill:
            prefill_position_ids = torch.zeros(
                batch_size,
                num_text,
                device=text_embeds.device,
                dtype=torch.float32,
            )
        routed_text_count = num_text - 1 if dummy_prefill else num_text
        _, normed_outputs, past_key_values = self._backbone_forward(
            inputs_embeds=text_embeds,
            attention_mask=text_valid_mask,
            position_ids=prefill_position_ids,
            past_key_values=None,
            use_cache=True,
            n_text_tokens=routed_text_count,
        )

        last_output = normed_outputs[:, -1:, :].reshape(
            batch_size, 1, self.output_patch_len, self.num_q
        )[:, -1, :, self.decode_index]
        point_chunks = [last_output]
        if dummy_prefill:
            keep_tokens = num_text - 1
            past_key_values = crop_kv_cache_to_length(
                past_key_values,
                keep_tokens,
            )
            cached_valid_mask = text_valid_mask[:, :keep_tokens]
        else:
            cached_valid_mask = text_valid_mask
        num_decode_steps = (horizon - 1) // self.output_patch_len

        for _ in range(num_decode_steps):
            new_patched_input = last_output.detach().reshape(
                batch_size, self.decode_stride_patches, self.input_patch_len
            )
            new_embeddings = self._tokenize_standardized_patches(
                new_patched_input)
            step_valid_mask = torch.ones(
                batch_size,
                self.decode_stride_patches,
                device=text_embeds.device,
                dtype=torch.bool,
            )
            total_valid_mask = torch.cat(
                [cached_valid_mask, step_valid_mask], dim=1)
            step_attn_mask_4d = self._causal_cache_attention_mask(
                valid_mask=total_valid_mask,
                query_len=self.decode_stride_patches,
            )
            _, new_normed_output, past_key_values = self._backbone_forward(
                inputs_embeds=new_embeddings,
                attention_mask=step_attn_mask_4d,
                position_ids=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            new_output = new_normed_output.reshape(
                batch_size,
                self.decode_stride_patches,
                self.output_patch_len,
                self.num_q,
            )
            last_output = new_output[:, -1, :, self.decode_index]
            point_chunks.append(last_output)
            cached_valid_mask = total_valid_mask

        return torch.cat(point_chunks, dim=1)[:, :horizon]

    def _decode_with_prefix(
        self,
        horizon: int,
        inputs: torch.Tensor,
        pad_mask: torch.Tensor,
        stats_pad_mask: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        text_invalid_mask: Optional[torch.Tensor] = None,
        text_tp: Optional[torch.Tensor] = None,
        text_temporal_positions: Optional[torch.Tensor] = None,
        context_patch_positions: Optional[torch.Tensor] = None,
        context_temporal_positions: Optional[torch.Tensor] = None,
        future_patch_positions: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        _skip_flip_invariance: bool = False,
        precomputed: Optional[RevinEncoded] = None,
        return_quantiles: bool = False,
    ):
        batch_size, context_len = inputs.shape
        num_decode_steps = (horizon - 1) // self.output_patch_len

        patched_inputs = inputs.reshape(batch_size, -1, self.input_patch_len)
        patched_masks = pad_mask.reshape(batch_size, -1, self.input_patch_len)
        stats_patched_masks = (
            patched_masks
            if stats_pad_mask is None
            else stats_pad_mask.reshape(batch_size, -1, self.input_patch_len)
        )

        if precomputed is None:
            sequence_embeddings, patch_level_mask, running_stats = self._revin_encode(
                patched_inputs,
                stats_patched_masks,
                model_patched_masks=patched_masks,
            )
        else:
            sequence_embeddings = precomputed.sequence_embeddings
            patch_level_mask = precomputed.patch_level_mask
            running_stats = precomputed.running_stats
        sequence_embeddings, sequence_position_ids, sequence_attn_mask_4d, sequence_valid_mask, n_text_tokens = (
            self._prepare_prefill_prefix(
                input_embeddings=sequence_embeddings,
                patch_level_mask=patch_level_mask,
                text_embeds=text_embeds,
                text_invalid_mask=text_invalid_mask,
                text_tp=text_tp,
                text_temporal_positions=text_temporal_positions,
                context_patch_positions=context_patch_positions,
                context_temporal_positions=context_temporal_positions,
            )
        )

        sequence_temporal_positions = None
        if sequence_position_ids is not None:
            sequence_temporal_positions = sequence_position_ids.clone()
            if n_text_tokens > 0 and text_temporal_positions is not None:
                sequence_temporal_positions[:, :n_text_tokens] = text_temporal_positions
            if n_text_tokens < sequence_position_ids.shape[1]:
                observed_temporal = context_temporal_positions
                if observed_temporal is None:
                    observed_temporal = sequence_position_ids[:, n_text_tokens:]
                sequence_temporal_positions[:, n_text_tokens:] = observed_temporal

        prefill_attention_mask = (
            sequence_attn_mask_4d
            if sequence_attn_mask_4d is not None
            else sequence_valid_mask
        )
        routed_text_count = n_text_tokens

        past_key_values = None
        _, normed_outputs, past_key_values = self._backbone_forward(
            inputs_embeds=sequence_embeddings,
            attention_mask=prefill_attention_mask,
            position_ids=sequence_position_ids,
            past_key_values=None,
            use_cache=use_cache,
            n_text_tokens=routed_text_count,
        )
        sequence_mask = ~sequence_valid_mask.bool()
        cached_valid_mask = sequence_valid_mask.bool()
        last_normed_output = normed_outputs[:, -1:, :]
        last_full = self._revin_decode(
            last_normed_output,
            {
                "mu": running_stats["mu"][:, -1:],
                "sigma": running_stats["sigma"][:, -1:],
            },
        ).reshape(batch_size, 1, self.output_patch_len, self.num_q)[:, -1, :, :]
        last_renormed_output = last_full[..., self.decode_index]

        point_chunks = [last_renormed_output]
        quant_chunks = [last_full[..., 1:]] if return_quantiles else None

        for decode_step in range(num_decode_steps):
            new_patched_input = last_renormed_output.detach().reshape(
                batch_size, self.decode_stride_patches, self.input_patch_len
            )
            new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)
            new_embeddings, new_patch_mask, new_stats = self._revin_encode(
                new_patched_input,
                new_mask,
                running_stats=running_stats,
            )
            running_stats = new_stats

            step_position_ids = None
            if future_patch_positions is not None:
                lo = decode_step * self.decode_stride_patches
                hi = lo + self.decode_stride_patches
                step_position_ids = future_patch_positions[:, lo:hi]
            elif sequence_position_ids is not None and not use_cache:
                step_position_ids = (
                    sequence_position_ids[:, -1:].to(dtype=torch.float32)
                    + torch.arange(
                        1,
                        self.decode_stride_patches + 1,
                        device=sequence_position_ids.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )

            if use_cache:
                step_valid_mask = torch.ones(
                    batch_size,
                    self.decode_stride_patches,
                    device=sequence_embeddings.device,
                    dtype=torch.bool,
                )
                total_valid_mask = torch.cat(
                    [cached_valid_mask, step_valid_mask], dim=1)
                step_attn_mask_4d = self._causal_cache_attention_mask(
                    valid_mask=total_valid_mask,
                    query_len=self.decode_stride_patches,
                )
                _, new_normed_output, past_key_values = self._backbone_forward(
                    inputs_embeds=new_embeddings,
                    attention_mask=step_attn_mask_4d,
                    position_ids=step_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                cached_valid_mask = total_valid_mask
            else:
                sequence_embeddings = torch.cat(
                    [sequence_embeddings, new_embeddings], dim=1)
                sequence_mask = torch.cat(
                    [sequence_mask, new_patch_mask], dim=1)
                sequence_valid_mask = torch.cat(
                    [
                        sequence_valid_mask,
                        torch.ones(
                            batch_size,
                            self.decode_stride_patches,
                            device=sequence_embeddings.device,
                            dtype=sequence_valid_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

                if step_position_ids is not None and sequence_position_ids is not None:
                    sequence_position_ids = torch.cat(
                        [sequence_position_ids, step_position_ids], dim=1)
                    if sequence_temporal_positions is not None:
                        sequence_temporal_positions = torch.cat(
                            [sequence_temporal_positions, step_position_ids], dim=1
                        )

                if sequence_attn_mask_4d is not None:
                    sequence_attn_mask_4d = self._prepare_prefill_prefix(
                        input_embeddings=sequence_embeddings[:, n_text_tokens:],
                        patch_level_mask=sequence_mask[:, n_text_tokens:],
                        text_embeds=sequence_embeddings[:, :n_text_tokens],
                        text_invalid_mask=~sequence_valid_mask[:, :n_text_tokens].bool(
                        ),
                        text_tp=sequence_position_ids[:, :n_text_tokens],
                        text_temporal_positions=(
                            sequence_temporal_positions[:, :n_text_tokens]
                            if sequence_temporal_positions is not None else None
                        ),
                        context_patch_positions=sequence_position_ids[:,
                                                                      n_text_tokens:],
                        context_temporal_positions=(
                            sequence_temporal_positions[:, n_text_tokens:]
                            if sequence_temporal_positions is not None else None
                        ),
                    )[2]

                full_attention_mask = (
                    sequence_attn_mask_4d
                    if sequence_attn_mask_4d is not None
                    else sequence_valid_mask
                )
                _, normed_outputs, _ = self._backbone_forward(
                    inputs_embeds=sequence_embeddings,
                    attention_mask=full_attention_mask,
                    position_ids=sequence_position_ids,
                    past_key_values=None,
                    use_cache=False,
                    n_text_tokens=routed_text_count,
                )
                new_normed_output = normed_outputs[:, -self.decode_stride_patches:, :]

            new_renormed_output = self._revin_decode(
                new_normed_output,
                new_stats,
            ).reshape(batch_size, self.decode_stride_patches, self.output_patch_len, self.num_q)
            new_full = new_renormed_output[:, -1, :, :]
            last_renormed_output = new_full[..., self.decode_index]
            point_chunks.append(last_renormed_output)
            if return_quantiles:
                quant_chunks.append(new_full[..., 1:])

        point = torch.cat(point_chunks, dim=1)[:, :horizon]
        quantiles = (
            torch.cat(quant_chunks, dim=1)[:, :horizon, :] if return_quantiles else None
        )
        if (
            self.force_flip_invariance
            and not _skip_flip_invariance
            and self.timesfm_decoupled_flip
            and text_embeds is not None
            and not return_quantiles
        ):
            f_x0 = self._decode_with_prefix(
                horizon=horizon,
                inputs=inputs,
                pad_mask=pad_mask,
                stats_pad_mask=stats_pad_mask,
                text_embeds=None,
                text_invalid_mask=None,
                text_tp=None,
                text_temporal_positions=None,
                context_patch_positions=context_patch_positions,
                context_temporal_positions=context_temporal_positions,
                future_patch_positions=future_patch_positions,
                use_cache=True,
                _skip_flip_invariance=True,
                precomputed=precomputed,
                return_quantiles=False,
            )
            f_negx0 = self._decode_with_prefix(
                horizon=horizon,
                inputs=-inputs,
                pad_mask=pad_mask,
                stats_pad_mask=stats_pad_mask,
                text_embeds=None,
                text_invalid_mask=None,
                text_tp=None,
                text_temporal_positions=None,
                context_patch_positions=context_patch_positions,
                context_temporal_positions=context_temporal_positions,
                future_patch_positions=future_patch_positions,
                use_cache=True,
                _skip_flip_invariance=True,
                precomputed=None,
                return_quantiles=False,
            )
            point = (f_x0 - f_negx0) / 2 + (point - f_x0)
            return point
        if self.force_flip_invariance and not _skip_flip_invariance:
            flipped = self._decode_with_prefix(
                horizon=horizon,
                inputs=-inputs,
                pad_mask=pad_mask,
                stats_pad_mask=stats_pad_mask,
                text_embeds=text_embeds,
                text_invalid_mask=text_invalid_mask,
                text_tp=text_tp,
                text_temporal_positions=text_temporal_positions,
                context_patch_positions=context_patch_positions,
                context_temporal_positions=context_temporal_positions,
                future_patch_positions=future_patch_positions,
                use_cache=True,
                _skip_flip_invariance=True,
                return_quantiles=return_quantiles,
            )
            if return_quantiles:
                point_flipped, quant_flipped = flipped
                quantiles = (quantiles - torch.flip(quant_flipped, dims=[-1])) / 2
            else:
                point_flipped = flipped
            point = (point - point_flipped) / 2
        if return_quantiles:
            return point, quantiles
        return point

    def _median_forecast(
        self,
        batch: Dict[str, Any],
        forecast_horizon: int,
        return_quantiles: bool = False,
    ):
        observed_data = batch["observed_data"]
        observed_mask = batch.get("observed_mask")
        target_idx = int(batch.get("target_idx", 0))

        batch_size, _, num_channels = observed_data.shape

        context = observed_data[:, :, target_idx]
        context_valid_anchor = (
            observed_mask[:, :, target_idx].bool()
            if observed_mask is not None
            else torch.ones(batch_size, context.shape[1], device=context.device, dtype=torch.bool)
        )
        missing_policy = getattr(
            self.cfg.affcon, "observed_missing_policy", "mask_aware"
        )
        context_valid_model = model_observed_mask(
            context_valid_anchor.to(dtype=context.dtype),
            missing_policy,
        ).bool()
        inputs, pad_mask = self._prepare_context(context, context_valid_model)
        _, stats_pad_mask = self._prepare_context(context, context_valid_anchor)
        if self.text_attention_mode not in ("temporal_causal", "global_prefix"):
            raise ValueError(
                f"Unsupported text_attention_mode: {self.text_attention_mode}"
            )
        if self.text_position_mode not in ("token_aligned", "raw_timestamp", "zero"):
            raise ValueError(
                f"Unsupported text_position_mode: {self.text_position_mode}"
            )
        if self.text_injection_layout not in ("distributed", "collapsed_global"):
            raise ValueError(
                f"Unsupported text_injection_layout: {self.text_injection_layout}"
            )

        text_embeds = None
        text_invalid_mask = None
        text_tp_flat = None
        text_temporal_positions = None
        patched_inputs = inputs.reshape(batch_size, -1, self.input_patch_len)
        patched_masks = pad_mask.reshape(batch_size, -1, self.input_patch_len)
        stats_patched_masks = stats_pad_mask.reshape(
            batch_size, -1, self.input_patch_len)
        _seq_emb, _patch_mask, _stats = self._revin_encode(
            patched_inputs,
            stats_patched_masks,
            model_patched_masks=patched_masks,
        )
        precomputed = RevinEncoded(
            sequence_embeddings=_seq_emb,
            patch_level_mask=_patch_mask,
            running_stats=_stats,
        )

        context_patch_positions = None
        context_temporal_positions = None
        future_patch_positions = None
        raw_context_patch_positions = None
        observed_tp = batch.get("observed_tp")
        if observed_tp is not None:
            obs_tp_flat = observed_tp.float().to(device=inputs.device)
            if obs_tp_flat.shape[-1] > self.context_limit:
                obs_tp_flat = obs_tp_flat[:, -self.context_limit:]
            pad_len = (-obs_tp_flat.shape[-1]) % self.input_patch_len
            if pad_len > 0:
                obs_tp_flat = F.pad(obs_tp_flat, (pad_len, 0), value=0.0)
            raw_context_patch_positions = obs_tp_flat.reshape(
                batch_size, -1, self.input_patch_len
            ).max(dim=-1).values
            context_patch_positions = raw_context_patch_positions
            context_temporal_positions = raw_context_patch_positions

            tp_to_predict = batch.get("tp_to_predict")
            if tp_to_predict is not None:
                tp_fut = tp_to_predict.float().to(device=inputs.device)
                pad_len_f = (-tp_fut.shape[-1]) % self.input_patch_len
                if pad_len_f > 0:
                    tp_fut = F.pad(tp_fut, (0, pad_len_f), value=0.0)
                n_future_patches = tp_fut.shape[-1] // self.input_patch_len
                future_patch_positions = tp_fut.reshape(
                    batch_size, n_future_patches, self.input_patch_len
                ).max(dim=-1).values

        if self.patch_position_mode == "contiguous" and self.text_position_mode != "raw_timestamp":
            num_context_patches = inputs.shape[-1] // self.input_patch_len
            context_patch_positions = self._sequence_positions(
                batch_size=batch_size,
                seq_len=num_context_patches,
                device=inputs.device,
            )
            context_temporal_positions = context_patch_positions
            if future_patch_positions is not None:
                future_patch_positions = self._sequence_positions(
                    batch_size=batch_size,
                    seq_len=future_patch_positions.shape[1],
                    device=inputs.device,
                    start=num_context_patches,
                )

        if self.text_position_mode == "raw_timestamp":
            if raw_context_patch_positions is None or context_patch_positions is None:
                raise ValueError(
                    "TimesFM raw_timestamp text positions require observed_tp so "
                    "text and time-series patch positions share the raw elapsed-time axis."
                )
            if not torch.allclose(
                context_patch_positions.float(),
                raw_context_patch_positions.float(),
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError(
                    "TimesFM raw_timestamp text positions require context patch "
                    "positions to remain on the raw elapsed-time axis."
                )
            context_temporal_positions = context_patch_positions

        affflow_x0 = None
        text_tp_raw = batch.get("text_tp")
        text_embeddings_raw = batch.get("text_embeddings")
        if text_embeddings_raw is None:
            raise ValueError("affflow mode requires text_embeddings in batch")
        if text_tp_raw is None or raw_context_patch_positions is None:
            raise ValueError(
                "affflow mode requires text_tp and observed_tp for aligned x0"
            )
        text_mask_raw = batch.get("text_mask")
        if text_mask_raw is None:
            text_mask_raw = torch.ones(
                text_embeddings_raw.shape[:2],
                device=text_embeddings_raw.device,
                dtype=torch.bool,
            )
        context_patch_valid = ~stats_pad_mask.reshape(
            batch_size, -1, self.input_patch_len
        ).all(dim=-1)
        text_ref_idx = find_anchor_indices(
            text_tp=text_tp_raw.to(device=inputs.device, dtype=torch.float32),
            text_mask=text_mask_raw.to(device=inputs.device, dtype=torch.bool),
            ref_tp=raw_context_patch_positions.float(),
            ref_valid=context_patch_valid.bool(),
        )
        d_ts = precomputed.sequence_embeddings.shape[-1]
        affflow_x0 = precomputed.sequence_embeddings.gather(
            1, text_ref_idx.unsqueeze(-1).expand(-1, -1, d_ts)
        )
        if getattr(self.cfg.affcon, "flow_init", "anchor") == "noise":
            affflow_x0 = torch.randn_like(affflow_x0)

        text_embeds, text_invalid_mask, text_tp_flat = self._prepare_conditioning_tokens(
            batch=batch,
            num_channels=1,
            batch_size_flat=batch_size,
            device=inputs.device,
            dtype=inputs.dtype,
            affflow_x0=affflow_x0,
        )

        if text_embeds is not None and text_invalid_mask is not None:
            text_temporal_positions = text_tp_flat
            if self.text_position_mode == "token_aligned":
                if (
                    text_tp_flat is not None
                    and raw_context_patch_positions is not None
                    and context_patch_positions is not None
                ):
                    context_patch_valid = ~stats_pad_mask.reshape(
                        batch_size, -1, self.input_patch_len
                    ).all(dim=-1)
                    text_ref_idx = find_anchor_indices(
                        text_tp=text_tp_flat.float(),
                        text_mask=(~text_invalid_mask).bool(),
                        ref_tp=raw_context_patch_positions.float(),
                        ref_valid=context_patch_valid.bool(),
                    )
                    text_tp_flat = context_patch_positions.gather(1, text_ref_idx)
                    text_temporal_positions = text_tp_flat
            elif self.text_position_mode == "zero":
                if (
                    text_tp_flat is not None
                    and raw_context_patch_positions is not None
                    and context_patch_positions is not None
                ):
                    context_patch_valid = ~stats_pad_mask.reshape(
                        batch_size, -1, self.input_patch_len
                    ).all(dim=-1)
                    text_ref_idx = find_anchor_indices(
                        text_tp=text_tp_flat.float(),
                        text_mask=(~text_invalid_mask).bool(),
                        ref_tp=raw_context_patch_positions.float(),
                        ref_valid=context_patch_valid.bool(),
                    )
                    text_temporal_positions = context_patch_positions.gather(
                        1, text_ref_idx)
                text_tp_flat = torch.zeros(
                    batch_size,
                    text_embeds.shape[1],
                    device=inputs.device,
                    dtype=(context_patch_positions.dtype if context_patch_positions is not None else torch.long),
                )
            elif self.text_position_mode == "raw_timestamp":
                if text_tp_flat is None:
                    raise ValueError(
                        "TimesFM raw_timestamp text positions require text_tp in the batch."
                    )
                if raw_context_patch_positions is None or context_patch_positions is None:
                    raise ValueError(
                        "TimesFM raw_timestamp text positions require observed_tp so "
                        "text and time-series patch positions share the raw elapsed-time axis."
                    )
                if not torch.allclose(
                    context_patch_positions.float(),
                    raw_context_patch_positions.float(),
                    rtol=1e-5,
                    atol=1e-5,
                ):
                    raise ValueError(
                        "TimesFM raw_timestamp text positions require context patch "
                        "positions to remain on the raw elapsed-time axis."
                    )
                text_temporal_positions = text_tp_flat

        if (
            text_embeds is not None
            and text_invalid_mask is not None
            and self.text_injection_layout == "collapsed_global"
        ):
            orig_text_embeds = text_embeds
            orig_text_valid_mask = ~text_invalid_mask
            text_embeds, text_valid_mask, text_tp_flat = collapse_text_slots(
                orig_text_embeds,
                orig_text_valid_mask,
                text_tp_flat,
                obs_tp=context_patch_positions,
            )
            _, _, text_temporal_positions = collapse_text_slots(
                orig_text_embeds,
                orig_text_valid_mask,
                text_temporal_positions,
                obs_tp=context_patch_positions,
            )
            text_invalid_mask = ~text_valid_mask.bool()
        decode_out = self._decode_with_prefix(
            horizon=forecast_horizon,
            inputs=inputs,
            pad_mask=pad_mask,
            stats_pad_mask=stats_pad_mask,
            text_embeds=text_embeds,
            text_invalid_mask=text_invalid_mask,
            text_tp=text_tp_flat,
            text_temporal_positions=text_temporal_positions,
            context_patch_positions=context_patch_positions,
            context_temporal_positions=context_temporal_positions,
            future_patch_positions=future_patch_positions,
            use_cache=True,
            precomputed=precomputed,
            return_quantiles=return_quantiles,
        )
        if return_quantiles:
            point, quantiles = decode_out
        else:
            point = decode_out

        if self.timesfm_infer_is_positive:
            hist_nonneg = ((context >= 0) | ~context_valid_anchor).all(
                dim=1, keepdim=True)
            point = torch.where(hist_nonneg, point.clamp_min(0.0), point)
            if return_quantiles:
                quantiles = torch.where(
                    hist_nonneg.unsqueeze(-1),
                    quantiles.clamp_min(0.0),
                    quantiles,
                )

        y_hat = torch.zeros(
            batch_size, forecast_horizon, num_channels,
            device=point.device, dtype=point.dtype,
        )
        y_hat[:, :, target_idx] = point
        y_hat = y_hat.to(device=observed_data.device, dtype=observed_data.dtype)
        if return_quantiles:
            return y_hat, quantiles.to(device=observed_data.device, dtype=observed_data.dtype)
        return y_hat

    def _quantile_forecast(
        self,
        batch: Dict[str, Any],
        forecast_horizon: int,
    ):
        _, quantiles = self._median_forecast(
            batch, forecast_horizon, return_quantiles=True)
        return quantiles, self.q_levels.to(device=quantiles.device)

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, Any],
        horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        target_idx = int(batch.get("target_idx", 0))
        forecast_horizon = int(horizon)
        y_hat = self._median_forecast(batch, forecast_horizon)
        return {
            "y_hat": y_hat,
            "target_idx": target_idx,
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        observed_data = batch.get("observed_data")
        data_to_predict = batch.get("data_to_predict")
        target_idx = int(batch.get("target_idx", 0))

        if observed_data is None:
            raise ValueError(
                "TimesFM wrapper requires observed_data in batch.")

        forecast_horizon = data_to_predict.shape[1]
        y_hat = self._median_forecast(batch, forecast_horizon)

        mask_pred = None
        mask_forecast_target = None
        if batch.get("mask_predicted_data") is not None:
            mask_pred_all = batch["mask_predicted_data"].to(
                device=observed_data.device,
                dtype=observed_data.dtype,
            )
            mask_pred = mask_pred_all[:, :, target_idx]
            mask_forecast_target = torch.zeros_like(mask_pred_all)
            mask_forecast_target[:, :, target_idx] = mask_pred

        loss = None
        if torch.is_tensor(data_to_predict):
            target = data_to_predict[:, :, target_idx].to(
                device=observed_data.device,
                dtype=observed_data.dtype,
            )
            pred_target = y_hat[:, :, target_idx]
            if mask_pred is not None:
                denom = mask_pred.sum().clamp_min(1.0)
                diff = (pred_target - target) * mask_pred
                loss = (diff * diff).sum() / denom
            else:
                diff = pred_target - target
                loss = (diff * diff).mean()

        affflow_loss = None

        return {
            "y_hat": y_hat,
            "loss": loss,
            "target_idx": target_idx,
            "mask_forecast_target": mask_forecast_target,
            "affflow_loss": affflow_loss,
        }

    def forward_text_only(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = batch["text_embeddings"]
        gt_series = batch["gt_series"]
        lengths = batch.get("lengths")

        batch_size = text_embeddings.shape[0]
        num_text = text_embeddings.shape[1]
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
                        device=gt_series.device).manual_seed(0)
                dummy_patch = torch.randn(
                    batch_size,
                    1,
                    self.input_patch_len,
                    device=gt_series.device,
                    dtype=gt_series.dtype,
                    generator=generator,
                )
            else:
                dummy_patch = torch.zeros(
                    batch_size,
                    1,
                    self.input_patch_len,
                    device=gt_series.device,
                    dtype=gt_series.dtype,
                )
            dummy_token = self._tokenize_standardized_patches(dummy_patch).to(
                device=gt_series.device,
                dtype=gt_series.dtype,
            )

        affiliated_token_control = getattr(
            self.cfg.affcon, "affiliated_token_control", "none")
        if dummy_token is None:
            raise RuntimeError("affflow requires a dummy token for x0")
        d_ts = dummy_token.shape[-1]
        if affiliated_token_control == "random":
            generator = None
            if not self.training:
                generator = torch.Generator(
                    device=gt_series.device).manual_seed(0)
            text_tokens = torch.randn(
                (batch_size, num_text, d_ts),
                device=dummy_token.device,
                dtype=dummy_token.dtype,
                generator=generator,
            )
        elif affiliated_token_control == "none":
            x0 = (
                dummy_token.expand(batch_size, num_text, d_ts)
                .reshape(batch_size * num_text, d_ts)
                .to(dtype=text_embeddings.dtype)
            )
            if use_token_noise_x0:
                generator = None
                if not self.training:
                    generator = torch.Generator(
                        device=gt_series.device).manual_seed(0)
                x0 = torch.randn(
                    x0.shape,
                    device=x0.device,
                    dtype=x0.dtype,
                    generator=generator,
                )
            text_flat = text_embeddings.reshape(batch_size * num_text, -1)
            freeze = getattr(self.cfg.train, "freeze_affflow", False)
            train_affflow = self.training and not freeze
            grad_ctx = torch.enable_grad() if train_affflow else torch.no_grad()
            with grad_ctx:
                text_tokens = self.affflow.sample(
                    text_flat, x0=x0, random_grid=(train_affflow and getattr(self.cfg.affcon, "affflow_train_random_grid", True))
                ).reshape(batch_size, num_text, d_ts).to(
                    device=gt_series.device,
                    dtype=gt_series.dtype,
                )
        else:
            raise ValueError(
                f"Unsupported affiliated_token_control: {affiliated_token_control}"
            )

        if dummy_token is not None:
            text_tokens = torch.cat([text_tokens, dummy_token.to(dtype=text_tokens.dtype)], dim=1)
        text_invalid_mask = torch.zeros(
            batch_size,
            text_tokens.shape[1],
            device=gt_series.device,
            dtype=torch.bool,
        )

        pred = self._decode_text_only(
            horizon=target_len,
            text_embeds=text_tokens,
            text_invalid_mask=text_invalid_mask,
        ).to(dtype=gt_series.dtype)

        if lengths is not None:
            mask = (
                torch.arange(target_len, device=gt_series.device).unsqueeze(0)
                < lengths.to(gt_series.device).unsqueeze(1)
            ).to(gt_series.dtype)
        else:
            mask = torch.ones_like(gt_series)
        gt_z, _, _ = standardize_w_mask(gt_series, mask)
        diff = (pred - gt_z) * mask
        loss = (diff * diff).sum() / mask.sum().clamp_min(1.0)

        return {"y_hat": pred, "loss": loss}
