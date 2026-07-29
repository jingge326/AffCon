
from __future__ import annotations

from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.coms_text import (
    build_affiliated_token_inputs,
    collapse_text_slots,
    find_anchor_indices,
)
from models.coms_lora import LoRASettings, apply_lora_inplace, set_backbone_eval_keep_lora_train
from models.coms_affflow import AffFlow
from models.coms_sundial import (
    FlowLoss,
    SundialForPrediction,
    SundialModel,
    model_observed_mask,
    standardize_w_mask,
)

from dataclasses import dataclass

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)


@dataclass
class BackboneHandles:
    sundial_model: SundialModel
    qwen_model: Optional[AutoModel]
    qwen_tokenizer: Optional[AutoTokenizer]
    flow_loss: FlowLoss
    d_ts: int
    d_txt: int


def load_backbones(
    sundial_model_name: str,
    qwen_model_name: str,
    device: torch.device,
    use_flash_attn: bool = False,
    freeze_sundial: bool = True,
    freeze_text: bool = True,
    load_text_backbone: bool = True,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modality_routed: bool = False,
) -> BackboneHandles:
    sundial_for_pred = SundialForPrediction.from_pretrained(
        sundial_model_name,
        trust_remote_code=True,
    )
    sundial_model: SundialModel = sundial_for_pred.model
    flow_loss: FlowLoss = sundial_for_pred.flow_loss
    sundial_model.to(device)
    sundial_model.eval()
    if freeze_sundial:
        for p in sundial_model.parameters():
            p.requires_grad_(False)

    if use_lora:
        apply_lora_inplace(
            sundial_model,
            backbone_name="sundial",
            settings=LoRASettings(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                modality_routed=lora_modality_routed,
            ),
        )

    flow_loss.to(device)
    flow_loss.eval()
    for p in flow_loss.parameters():
        p.requires_grad_(False)

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

    d_ts = sundial_model.config.hidden_size

    return BackboneHandles(
        sundial_model=sundial_model,
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
        flow_loss=flow_loss,
        d_ts=d_ts,
        d_txt=d_txt,
    )


class SundialForecaster(nn.Module):

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
        self.sundial = backbones.sundial_model
        self.qwen_model = backbones.qwen_model
        self.qwen_tokenizer = backbones.qwen_tokenizer
        self.flow_loss = backbones.flow_loss
        self.d_ts = backbones.d_ts
        self.d_txt = backbones.d_txt
        self.cfg = cfg
        self.run_mode = run_mode
        self.use_lora = use_lora
        self.patch_position_mode = getattr(
            cfg.affcon, "patch_position_mode", "contiguous")
        self.affflow_dummy_token = getattr(
            cfg.affcon, "affflow_dummy_token", False)
        self.affflow_dummy_init = getattr(
            cfg.affcon, "affflow_dummy_init", "zero")
        if self.affflow_dummy_init not in ("zero", "randn"):
            raise ValueError(
                f"Unsupported affflow_dummy_init: {self.affflow_dummy_init}"
            )
        self.lora_backbone = self.sundial
        self.lora_backbone_name = "sundial"
        self.text_gate = (
            nn.Parameter(torch.zeros(()))
            if getattr(cfg.affcon, "text_gate", False) else None
        )

        for p in self.flow_loss.parameters():
            p.requires_grad_(False)
        self.flow_loss.eval()

        self.use_affflow_adapter = False
        self.use_conditioned_zeroshot = (
            self.run_mode == 'zeroshot'
        )

        if self.run_mode == "zeroshot" and not self.use_conditioned_zeroshot:
            pass
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
            num_sampling_steps=getattr(cfg.affcon, "affflow_nfe", 50),
            train_steps_min=getattr(cfg.affcon, "affflow_train_steps_min", 8),
            train_steps_max=getattr(cfg.affcon, "affflow_train_steps_max", 64),
            text_mode=getattr(cfg.affcon, "affflow_text_mode", None),
            prior_type=getattr(cfg.affcon, "affflow_prior_type", "standard"),
            text_concat=getattr(cfg.affcon, "affflow_text_concat", True),
        )
        self.use_affflow_adapter = getattr(cfg.affcon, "affflow_adapter", False)
        if self.use_affflow_adapter:
            self.affflow_adapter = nn.Linear(self.d_ts, self.d_ts)

    def forward(self, batch: Dict[str, Any], horizon: Optional[int] = None, num_samples: int = 1) -> Dict[str, torch.Tensor]:
        observed_data = batch.get("observed_data")
        observed_mask = batch.get("observed_mask")
        observed_tp = batch.get("observed_tp")
        data_to_predict = batch.get("data_to_predict")
        target_idx = batch.get("target_idx")
        text_embeddings = batch.get("text_embeddings")
        text_token_mask = batch.get("patch_text_mask")
        if text_token_mask is None:
            text_token_mask = batch.get("text_mask")
        text_tp = batch.get("text_tp")

        output_channels = batch.get("data_to_predict").size(-1)
        if horizon is None:
            horizon = data_to_predict.shape[1]

        x = observed_data[:, :, target_idx]
        x_anchor_mask = observed_mask[:, :, target_idx]
        missing_policy = getattr(
            self.cfg.affcon, "observed_missing_policy", "mask_aware"
        )
        x_model_mask = model_observed_mask(x_anchor_mask, missing_policy)
        x, means, stdev = standardize_w_mask(x, x_anchor_mask)

        input_token_len = self.sundial.config.input_token_len
        input_length = x_anchor_mask.shape[-1]
        padding_length = (input_token_len - (input_length %
                          input_token_len)) % input_token_len
        patch_anchor_mask = F.pad(x_anchor_mask, (padding_length, 0))
        patch_anchor_mask = patch_anchor_mask.unfold(
            dimension=-1, size=input_token_len, step=input_token_len)
        patch_model_mask = F.pad(x_model_mask, (padding_length, 0))
        patch_model_mask = patch_model_mask.unfold(
            dimension=-1, size=input_token_len, step=input_token_len)
        patch_anchor_attn_mask = (patch_anchor_mask.sum(dim=-1) > 0).long()
        patch_attn_mask = (patch_model_mask.sum(dim=-1) > 0).long()
        tp_padded = F.pad(observed_tp, (padding_length, 0))
        tp_patch = tp_padded.unfold(
            dimension=-1, size=input_token_len, step=input_token_len)

        ts_tokens = self.sundial.embed_layer(
            x, x_model_mask)
        batch_size, n_patches, _ = ts_tokens.shape
        patch_anchor_time = (tp_patch * patch_anchor_mask).max(dim=-1).values
        patch_time = (tp_patch * patch_model_mask).max(dim=-1).values
        patch_index = torch.arange(
            n_patches,
            device=ts_tokens.device,
            dtype=patch_time.dtype,
        ).unsqueeze(0).expand(batch_size, -1)
        patch_position_ids = (
            patch_index if self.patch_position_mode == "contiguous" else patch_time
        )
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
        ts_prefix_position_ids = (
            patch_time if text_position_mode == "raw_timestamp" else patch_position_ids
        )

        if text_embeddings is not None:
            if text_token_mask is None:
                text_token_mask = torch.ones(
                    text_embeddings.shape[:2],
                    device=text_embeddings.device,
                    dtype=patch_attn_mask.dtype,
                )
            else:
                text_token_mask = text_token_mask.to(
                    device=text_embeddings.device,
                    dtype=patch_attn_mask.dtype,
                )

        text_validity = text_token_mask

        def align_text_positions_to_patches(
            raw_text_positions: torch.Tensor | None,
            validity: torch.Tensor | None,
            num_slots: int,
        ) -> torch.Tensor:
            if raw_text_positions is None:
                return torch.zeros(
                    batch_size,
                    num_slots,
                    device=patch_position_ids.device,
                    dtype=patch_position_ids.dtype,
                )
            if validity is None:
                validity = torch.ones(
                    batch_size,
                    num_slots,
                    device=patch_position_ids.device,
                    dtype=patch_attn_mask.dtype,
                )
            text_ref_idx = find_anchor_indices(
                text_tp=raw_text_positions.float(),
                text_mask=validity.bool(),
                ref_tp=patch_anchor_time.float(),
                ref_valid=patch_anchor_attn_mask.bool(),
            )
            return patch_position_ids.gather(1, text_ref_idx)

        def select_text_positions(
            raw_text_positions: torch.Tensor | None,
            validity: torch.Tensor | None,
            num_slots: int,
        ) -> torch.Tensor:
            if text_position_mode == "zero":
                return torch.zeros(
                    batch_size,
                    num_slots,
                    device=patch_position_ids.device,
                    dtype=patch_position_ids.dtype,
                )
            if raw_text_positions is None:
                raise ValueError(
                    "affflow mode requires text_tp in batch; "
                    "missing text timestamps would silently collapse text "
                    "position_ids to zero."
                )
            if text_position_mode == "raw_timestamp":
                return raw_text_positions.to(device=patch_position_ids.device)
            return align_text_positions_to_patches(
                raw_text_positions, validity, num_slots)

        n_text_tokens = 0

        if self.run_mode == "zeroshot" and not self.use_conditioned_zeroshot:
            inputs_embeds = ts_tokens
            position_ids = patch_position_ids
            attn_mask_full = patch_attn_mask
        if text_embeddings is None:
            raise ValueError(
                "affflow mode requires text_embeddings in batch")
        if text_tp is None:
            raise ValueError(
                "affflow mode requires text_tp in batch; "
                "missing text timestamps would silently collapse text "
                "position_ids to zero."
            )

        affiliated_token_control = getattr(
            self.cfg.affcon, "affiliated_token_control", "none")
        B_txt, N_txt, d_txt = text_embeddings.shape
        d_ts = ts_tokens.shape[-1]
        if affiliated_token_control == "random":
            generator = None
            if not self.training:
                generator = torch.Generator(
                    device=ts_tokens.device).manual_seed(0)
            text_tokens = torch.randn(
                (B_txt, N_txt, d_ts),
                device=ts_tokens.device,
                dtype=ts_tokens.dtype,
                generator=generator,
            )
        elif affiliated_token_control == "none":
            AffFlow = self.affflow
            freeze = self.cfg.train.freeze_affflow
            text_flat = text_embeddings.reshape(B_txt * N_txt, d_txt)
            text_ref_idx = find_anchor_indices(
                text_tp=text_tp.float(),
                text_mask=text_validity.bool(),
                ref_tp=patch_anchor_time.float(),
                ref_valid=patch_anchor_attn_mask.bool(),
            )
            x0 = ts_tokens.gather(
                1, text_ref_idx.unsqueeze(-1).expand(-1, -1, d_ts)
            ).reshape(B_txt * N_txt, d_ts)
            if getattr(self.cfg.affcon, "flow_init", "anchor") == "noise":
                x0 = torch.randn_like(x0)
            train_affflow = self.training and not freeze
            grad_ctx = torch.enable_grad() if train_affflow else torch.no_grad()
            with grad_ctx:
                text_tokens = AffFlow.sample(
                    text_flat, x0=x0, random_grid=(train_affflow and getattr(self.cfg.affcon, "affflow_train_random_grid", True))
                ).reshape(B_txt, N_txt, -1)
            if self.use_affflow_adapter:
                text_tokens = self.affflow_adapter(text_tokens)
        else:
            raise ValueError(
                f"Unsupported affiliated_token_control: {affiliated_token_control}"
            )

        text_positions = text_tp

        text_temporal_positions = text_positions
        text_positions = select_text_positions(
            text_positions,
            text_validity,
            text_tokens.shape[1],
        )
        if text_position_mode == "zero":
            text_temporal_positions = align_text_positions_to_patches(
                text_temporal_positions,
                text_validity,
                text_tokens.shape[1],
            )
        else:
            text_temporal_positions = text_positions

        if text_injection_layout == "collapsed_global":
            orig_text_tokens = text_tokens
            orig_text_validity = text_validity
            orig_text_temporal_positions = text_temporal_positions
            text_tokens, text_validity, text_positions = collapse_text_slots(
                orig_text_tokens,
                orig_text_validity,
                text_positions,
                obs_tp=ts_prefix_position_ids,
            )
            _, _, text_temporal_positions = collapse_text_slots(
                orig_text_tokens,
                orig_text_validity,
                orig_text_temporal_positions,
                obs_tp=ts_prefix_position_ids,
            )
        inputs_embeds, position_ids, attn_mask_full, _, n_text_tokens = build_affiliated_token_inputs(
            text_tokens=text_tokens,
            text_tp=text_positions,
            text_mask=text_validity,
            obs_tokens=ts_tokens,
            obs_tp=ts_prefix_position_ids,
            obs_mask=patch_attn_mask,
            text_temporal_positions=text_temporal_positions,
            obs_temporal_positions=patch_time if text_position_mode == "raw_timestamp" else patch_position_ids,
            temporal_text_restriction=temporal_text_restriction,
            text_gate=self.text_gate,
        )

        outputs = self.sundial(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_full,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            n_text_tokens=n_text_tokens,
        )
        last_hidden_state = outputs.last_hidden_state

        if self.run_mode == "zeroshot" and not self.use_conditioned_zeroshot:
            ts_hidden = last_hidden_state
        ts_hidden = last_hidden_state[:, n_text_tokens:, :]
        z = ts_hidden[:, -1, :]

        samples = self.flow_loss.sample(z, num_samples=num_samples)

        pred = samples[:, :, :horizon].mean(dim=1)
        pred = pred * stdev + means
        samples_raw = samples[:, :, :horizon] * stdev.unsqueeze(1) + means.unsqueeze(1)

        y_hat = torch.zeros(
            pred.shape[0],
            horizon,
            output_channels,
            device=pred.device,
            dtype=pred.dtype,
        )
        y_hat[:, :, target_idx] = pred

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
                diff = (pred - target) * mask_pred
                loss = (diff * diff).sum() / denom
            else:
                diff = pred - target
                loss = (diff * diff).mean()

        affflow_loss = None

        return {
            "y_hat": y_hat,
            "loss": loss,
            "affflow_loss": affflow_loss,
            "target_idx": target_idx,
            "mask_forecast_target": mask_forecast_target,
            "samples": samples_raw,
        }

    def forward_text_only(
        self,
        batch: Dict[str, Any],
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = batch["text_embeddings"]
        gt_series = batch["gt_series"]
        B = text_embeddings.shape[0]
        N_txt = text_embeddings.shape[1]
        max_len = gt_series.shape[1]

        dummy_token = None
        use_token_noise_x0 = (
            getattr(self.cfg.affcon, "flow_init", "anchor") == "noise"
        )
        if self.affflow_dummy_token:
            input_token_len = self.sundial.config.input_token_len
            if self.affflow_dummy_init == "randn":
                generator = None
                if not self.training:
                    generator = torch.Generator(
                        device=text_embeddings.device).manual_seed(0)
                dummy_patch = torch.randn(
                    B,
                    input_token_len,
                    device=text_embeddings.device,
                    dtype=gt_series.dtype,
                    generator=generator,
                )
            else:
                dummy_patch = torch.zeros(
                    B,
                    input_token_len,
                    device=text_embeddings.device,
                    dtype=gt_series.dtype,
                )
            dummy_mask = torch.ones(
                B,
                input_token_len,
                device=text_embeddings.device,
                dtype=gt_series.dtype,
            )
            dummy_token = self.sundial.embed_layer(dummy_patch, dummy_mask)

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

        inputs_embeds = text_proj
        if dummy_token is not None:
            inputs_embeds = torch.cat(
                [text_proj, dummy_token.to(dtype=text_proj.dtype)], dim=1)

        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(
            seq_len, device=text_proj.device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)
        attention_mask = torch.ones(
            B, seq_len, device=text_proj.device, dtype=torch.long)

        outputs = self.sundial(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            n_text_tokens=N_txt,
        )

        z = outputs.last_hidden_state[:, -1, :]

        samples = self.flow_loss.sample(z, num_samples=num_samples)
        pred = samples[:, :, :max_len].mean(dim=1)

        if "lengths" in batch:
            lengths = batch["lengths"].to(gt_series.device)
            mask = (torch.arange(max_len, device=gt_series.device).unsqueeze(0)
                    < lengths.unsqueeze(1)).to(gt_series.dtype)
        else:
            mask = torch.ones_like(gt_series)
        gt_z, _, _ = standardize_w_mask(gt_series, mask)
        diff = (pred - gt_z) * mask
        loss = (diff * diff).sum() / mask.sum().clamp_min(1.0)

        return {"y_hat": pred, "loss": loss}

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, Any],
        horizon: Optional[int] = None,
        num_samples: int = 5,
    ) -> torch.Tensor:
        observed_data = batch.get("observed_data")

        if not torch.is_tensor(batch.get("data_to_predict")):
            H = int(horizon)
            batch = dict(batch)
            if observed_data.dim() == 3:
                B, _, D = observed_data.shape
                batch["data_to_predict"] = torch.zeros(
                    B,
                    H,
                    D,
                    device=observed_data.device,
                    dtype=observed_data.dtype,
                )
            elif observed_data.dim() == 2:
                B, _ = observed_data.shape
                batch["data_to_predict"] = torch.zeros(
                    B,
                    H,
                    device=observed_data.device,
                    dtype=observed_data.dtype,
                )

        return self.forward(batch, horizon=horizon, num_samples=num_samples)

    def train(self, mode: bool = True):
        super().train(mode)
        set_backbone_eval_keep_lora_train(self.sundial, mode)
        if self.qwen_model is not None:
            self.qwen_model.eval()
        self.flow_loss.eval()
        return self
