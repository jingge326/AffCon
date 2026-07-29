from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRASettings:

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    modality_routed: bool = False


def _is_sundial_or_timemoe_attention(module: nn.Module) -> bool:
    return (
        isinstance(getattr(module, "q_proj", None), nn.Linear)
        and isinstance(getattr(module, "k_proj", None), nn.Linear)
        and isinstance(getattr(module, "v_proj", None), nn.Linear)
        and isinstance(getattr(module, "o_proj", None), nn.Linear)
    )


def _is_timesfm_unfused_attention(module: nn.Module) -> bool:
    return (
        hasattr(module, "fuse_qkv")
        and not getattr(module, "fuse_qkv", True)
        and isinstance(getattr(module, "key", None), nn.Linear)
        and isinstance(getattr(module, "value", None), nn.Linear)
    )


def _clone_linear(src: nn.Linear) -> nn.Linear:
    dst = nn.Linear(
        src.in_features,
        src.out_features,
        bias=src.bias is not None,
        device=src.weight.device,
        dtype=src.weight.dtype,
    )
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)
    return dst


def add_modality_routed_text_projections(
    model: nn.Module,
    backbone_name: str,
) -> int:
    if backbone_name in ("sundial", "timemoe"):
        text_names = ("k_proj_text", "v_proj_text")
        src_names = ("k_proj", "v_proj")
        predicate = _is_sundial_or_timemoe_attention
    elif backbone_name == "timesfm":
        for module in model.modules():
            if hasattr(module, "fuse_qkv") and getattr(module, "fuse_qkv", False):
                raise ValueError(
                    "modality_routed LoRA requires the unfused TimesFM "
                    "attention (fuse_qkv=False); got fuse_qkv=True."
                )
        text_names = ("key_text", "value_text")
        src_names = ("key", "value")
        predicate = _is_timesfm_unfused_attention
    else:
        raise ValueError(
            f"modality_routed LoRA is not supported for backbone: {backbone_name}"
        )

    modified = 0
    for module in model.modules():
        if not predicate(module):
            continue
        if any(hasattr(module, name) for name in text_names):
            continue
        for tgt, src in zip(text_names, src_names):
            setattr(module, tgt, _clone_linear(getattr(module, src)))
        modified += 1

    if modified == 0:
        raise RuntimeError(
            f"No eligible attention modules found for modality-routed LoRA on "
            f"backbone {backbone_name!r}."
        )
    return modified


def _resolve_timesfm_targets(
    model: nn.Module, modality_routed: bool
) -> Tuple[str, ...]:
    for module in model.modules():
        if hasattr(module, "fuse_qkv") and hasattr(module, "out"):
            if getattr(module, "fuse_qkv", False):
                if modality_routed:
                    raise ValueError(
                        "modality_routed LoRA requires the unfused TimesFM "
                        "attention (fuse_qkv=False); got fuse_qkv=True."
                    )
                return ("qkv_proj", "out")
            targets: List[str] = ["query", "key", "value", "out"]
            if modality_routed:
                targets += ["key_text", "value_text"]
            return tuple(targets)
    raise ValueError("Could not resolve TimesFM attention modules for LoRA injection.")


def resolve_lora_target_modules(
    backbone_name: str,
    model: nn.Module,
    modality_routed: bool = False,
) -> Tuple[str, ...]:
    if backbone_name in ("sundial", "timemoe"):
        targets: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if modality_routed:
            targets += ["k_proj_text", "v_proj_text"]
        return tuple(targets)
    if backbone_name == "timesfm":
        return _resolve_timesfm_targets(model, modality_routed=modality_routed)
    raise ValueError(f"Unsupported LoRA backbone: {backbone_name}")


def apply_lora_inplace(
    model: nn.Module,
    *,
    backbone_name: str,
    settings: LoRASettings,
) -> Tuple[Tuple[str, ...], int]:
    from peft import LoraConfig, inject_adapter_in_model

    if settings.modality_routed:
        add_modality_routed_text_projections(model, backbone_name)

    target_modules = resolve_lora_target_modules(
        backbone_name, model, modality_routed=settings.modality_routed
    )
    for param in model.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        target_modules=list(target_modules),
        lora_dropout=settings.dropout,
        bias="none",
    )
    original_config: Any = getattr(model, "config", None)
    if original_config is not None and not hasattr(original_config, "get"):
        class _AttributeConfigProxy:
            def __init__(self, wrapped: Any) -> None:
                self._wrapped = wrapped

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

            def get(self, name: str, default: Any = None) -> Any:
                return getattr(self._wrapped, name, default)

        model.config = _AttributeConfigProxy(original_config)
    try:
        inject_adapter_in_model(lora_config, model)
    finally:
        if original_config is not None and getattr(model, "config", None) is not original_config:
            model.config = original_config

    num_trainable = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            num_trainable += param.numel()
        else:
            param.requires_grad_(False)

    return target_modules, num_trainable


def has_lora_parameters(model: nn.Module) -> bool:
    return any("lora_" in name for name, _ in model.named_parameters())


def set_backbone_eval_keep_lora_train(backbone: nn.Module, mode: bool) -> None:
    backbone.eval()
    if not mode:
        return
    for module in backbone.modules():
        if hasattr(module, "lora_dropout") or type(module).__name__.startswith("Lora"):
            module.train(True)


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_" in key
    }


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Iterable[str], Iterable[str]]:
    incompatible = model.load_state_dict(state_dict, strict=False)
    return incompatible.missing_keys, incompatible.unexpected_keys
