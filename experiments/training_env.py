
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


TRAIN_PROFILES = (
    "p1_affflow",
    "sp_affflow",
    "p2_affflow",
    "p2_lora_affflow",
    "p2_lora_affflow_fast",
    "p2_frozen_lora_affflow",
    "p2_warm_lora_affflow",
)

@dataclass(frozen=True)
class TrainProfile:
    name: str
    stage: str
    patience: int
    warmup_ratio: float
    pretrained_affflow: bool
    require_lora: bool = False
    lora_warm_start: bool = False
    freeze_lora: bool = False
    stage_a_lora: bool = False
    lora_lr_override: Optional[float] = None
    affflow_lr_override: Optional[float] = None


PROFILE_REGISTRY: Dict[str, TrainProfile] = {
    "p1_affflow": TrainProfile(
        name="p1_affflow",
        stage="p1",
        patience=10,
        warmup_ratio=0.05,
        pretrained_affflow=False,
    ),
    "sp_affflow": TrainProfile(
        name="sp_affflow",
        stage="sp",
        patience=10,
        warmup_ratio=0.05,
        pretrained_affflow=False,
    ),
    "p2_affflow": TrainProfile(
        name="p2_affflow",
        stage="p2",
        patience=10,
        warmup_ratio=0.10,
        pretrained_affflow=True,
    ),
    "p2_lora_affflow": TrainProfile(
        name="p2_lora_affflow",
        stage="p2",
        patience=10,
        warmup_ratio=0.10,
        pretrained_affflow=True,
        require_lora=True,
        lora_warm_start=True,
    ),
    "p2_lora_affflow_fast": TrainProfile(
        name="p2_lora_affflow_fast",
        stage="p2",
        patience=10,
        warmup_ratio=0.10,
        pretrained_affflow=True,
        require_lora=True,
        lora_warm_start=True,
        affflow_lr_override=5e-4,
    ),
    "p2_frozen_lora_affflow": TrainProfile(
        name="p2_frozen_lora_affflow",
        stage="p2",
        patience=10,
        warmup_ratio=0.10,
        pretrained_affflow=True,
        require_lora=True,
        freeze_lora=True,
        stage_a_lora=True,
    ),
    "p2_warm_lora_affflow": TrainProfile(
        name="p2_warm_lora_affflow",
        stage="p2",
        patience=10,
        warmup_ratio=0.10,
        pretrained_affflow=True,
        require_lora=True,
        stage_a_lora=True,
        lora_lr_override=1e-5,
    ),
}


@dataclass(frozen=True)
class BatchPlan:
    effective_batch_size: int
    micro_batch_size: int
    grad_accum_steps: int
    auto_batch_fallback_enabled: bool


def profile_names() -> Tuple[str, ...]:
    return TRAIN_PROFILES


def get_train_profile(name: str) -> TrainProfile:
    try:
        return PROFILE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown train profile {name!r}. Expected one of: {', '.join(TRAIN_PROFILES)}"
        ) from exc


def validate_and_resolve_train_profile(args: Any) -> Optional[TrainProfile]:
    if getattr(args, "run_mode", None) != "train":
        return None

    profile_name = getattr(args, "train_profile", None)
    if not profile_name:
        raise ValueError(
            "--train-profile is required for --run-mode train. "
            f"Use one of: {', '.join(TRAIN_PROFILES)}."
        )

    profile = get_train_profile(profile_name)
    errors: List[str] = []

    run_stage = getattr(args, "run_stage", None)
    active_dataset = getattr(args, "active_dataset", None)
    use_lora = bool(getattr(args, "use_lora", False))
    pretrain_checkpoint = getattr(args, "pretrain_checkpoint", None)

    if run_stage != profile.stage:
        errors.append(
            f"{profile.name} requires --run-stage {profile.stage}, got {run_stage!r}."
        )

    if profile.name == "p1_affflow":
        if active_dataset != "synthetic":
            errors.append("p1_affflow requires --active-dataset synthetic.")
        if use_lora:
            errors.append("p1_affflow does not support --use-lora.")
        if pretrain_checkpoint is not None:
            errors.append("p1_affflow must not use --pretrain-checkpoint.")
    else:
        if active_dataset == "synthetic":
            errors.append(f"{profile.name} is for real-data training, not synthetic.")
        if profile.pretrained_affflow and pretrain_checkpoint is None:
            errors.append(f"{profile.name} requires --pretrain-checkpoint.")
        if not profile.pretrained_affflow and pretrain_checkpoint is not None:
            errors.append(f"{profile.name} must not use --pretrain-checkpoint.")

    if profile.require_lora and not use_lora:
        errors.append(f"{profile.name} requires --use-lora.")
    if not profile.require_lora and use_lora:
        errors.append(f"{profile.name} must not use --use-lora.")

    stage_a_lora_checkpoint = getattr(args, "stage_a_lora_checkpoint", None)
    if profile.stage_a_lora and stage_a_lora_checkpoint is None:
        errors.append(f"{profile.name} requires --stage-a-lora-checkpoint.")
    if not profile.stage_a_lora and stage_a_lora_checkpoint is not None:
        errors.append(f"{profile.name} must not use --stage-a-lora-checkpoint.")

    if getattr(args, "patience", None) is None:
        args.patience = profile.patience

    if errors:
        raise ValueError("Invalid training profile configuration:\n- " + "\n- ".join(errors))
    return profile


def resolve_batch_plan(args: Any) -> BatchPlan:
    effective = int(getattr(args, "batch_size"))
    if effective <= 0:
        raise ValueError("--batch-size must be positive.")

    legacy_initial = getattr(args, "initial_micro_batch_size", None)
    micro = getattr(args, "micro_batch_size", None)
    accum = getattr(args, "grad_accum_steps", None)
    if legacy_initial is not None:
        if micro is not None:
            raise ValueError(
                "--initial-micro-batch-size is deprecated; do not combine it with --micro-batch-size."
            )
        micro = legacy_initial

    if micro is not None:
        micro = int(micro)
        if micro <= 0:
            raise ValueError("--micro-batch-size must be positive.")
        if micro > effective:
            raise ValueError("--micro-batch-size cannot exceed --batch-size.")

    if accum is not None:
        accum = int(accum)
        if accum <= 0:
            raise ValueError("--grad-accum-steps must be positive.")

    if micro is None and accum is None:
        micro = effective
        accum = 1
    elif micro is None:
        if effective % accum != 0:
            raise ValueError("--batch-size must divide evenly by --grad-accum-steps.")
        micro = effective // accum
    elif accum is None:
        if effective % micro != 0:
            raise ValueError("--micro-batch-size must divide --batch-size evenly.")
        accum = effective // micro
    elif micro * accum != effective:
        raise ValueError(
            "--batch-size must equal --micro-batch-size * --grad-accum-steps."
        )

    return BatchPlan(
        effective_batch_size=effective,
        micro_batch_size=micro,
        grad_accum_steps=accum,
        auto_batch_fallback_enabled=bool(
            getattr(args, "auto_batch_fallback", True)
        ),
    )


def next_fallback_micro_batch(current_micro: int, effective_batch: int) -> Optional[int]:
    for candidate in range(current_micro // 2, 0, -1):
        if effective_batch % candidate == 0:
            return candidate
    return None


def make_batch_metadata(
    *,
    plan: BatchPlan,
    micro_batch_size: int,
    grad_accum_steps: int,
    oom_fallback_history: Iterable[Dict[str, int]],
) -> Dict[str, Any]:
    history = list(oom_fallback_history)
    return {
        "effective_batch_size": plan.effective_batch_size,
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "auto_batch_fallback_enabled": plan.auto_batch_fallback_enabled,
        "oom_fallback_triggered": bool(history),
        "oom_fallback_history": history,
    }


def _module_lookup(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _is_no_decay_param(name: str, modules: Dict[str, nn.Module]) -> bool:
    if name.endswith(".bias"):
        return True
    module_name = name.rsplit(".", 1)[0] if "." in name else ""
    module = modules.get(module_name)
    if isinstance(module, nn.LayerNorm):
        return True
    lname = name.lower()
    return "layernorm" in lname or "layer_norm" in lname or ".norm" in lname


def _owning_group(name: str) -> Optional[str]:
    if "lora_" in name:
        return "lora"
    if name == "text_gate":
        return "text_gate"
    if name.startswith("affflow."):
        return "affflow"
    if name.startswith("affflow_adapter."):
        return "affflow_adapter"
    return None


def _group_hparams(group: str, profile: TrainProfile) -> Tuple[float, float, Tuple[float, float]]:
    if group == "affflow":
        lr = profile.affflow_lr_override if profile.affflow_lr_override is not None else (
            1e-4 if profile.pretrained_affflow else 5e-4
        )
        return lr, 1e-2, (0.9, 0.999)
    if group == "lora":
        lr = profile.lora_lr_override if profile.lora_lr_override is not None else 2e-4
        return lr, 0.0, (0.9, 0.95)
    if group == "affflow_adapter":
        return 5e-4, 1e-2, (0.9, 0.999)
    if group == "text_gate":
        return 1e-2, 0.0, (0.9, 0.999)
    raise ValueError(f"Unknown optimizer parameter group: {group}")


def build_v5_optimizer(
    model: nn.Module,
    profile: TrainProfile,
) -> Tuple[AdamW, List[torch.nn.Parameter], Dict[str, Any]]:
    modules = _module_lookup(model)
    grouped: Dict[Tuple[str, bool], List[Tuple[str, torch.nn.Parameter]]] = {}
    absent = {"affflow", "lora", "affflow_adapter", "text_gate"}
    unmatched: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = _owning_group(name)
        if group is None:
            unmatched.append(name)
            continue
        absent.discard(group)
        no_decay = _is_no_decay_param(name, modules)
        grouped.setdefault((group, no_decay), []).append((name, param))

    if unmatched:
        sample = ", ".join(unmatched[:10])
        raise RuntimeError(
            "Trainable parameters were not assigned to a v5 optimizer group: "
            f"{sample}"
        )

    optimizer_groups: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    trainable_params: List[torch.nn.Parameter] = []

    for group_name in ("affflow", "lora", "affflow_adapter", "text_gate"):
        for no_decay in (False, True):
            entries = grouped.get((group_name, no_decay), [])
            if not entries:
                continue
            lr, weight_decay, betas = _group_hparams(group_name, profile)
            if no_decay:
                weight_decay = 0.0
            names = [name for name, _ in entries]
            params = [param for _, param in entries]
            trainable_params.extend(params)
            display_name = f"{group_name}.norm_bias" if no_decay else group_name
            optimizer_groups.append(
                {
                    "params": params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "name": display_name,
                }
            )
            summaries.append(
                {
                    "name": display_name,
                    "parent_group": group_name,
                    "no_decay": no_decay,
                    "num_parameters": int(sum(param.numel() for param in params)),
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": list(betas),
                    "representative_names": names[:8],
                }
            )

    if not optimizer_groups:
        raise RuntimeError("No trainable parameters were available for v5 optimizer groups.")

    optimizer = AdamW(optimizer_groups)
    metadata = {
        "profile": profile.name,
        "optimizer": "AdamW",
        "parameter_groups": summaries,
        "absent_parameter_groups": sorted(absent),
        "total_trainable_parameters": int(sum(s["num_parameters"] for s in summaries)),
    }
    return optimizer, trainable_params, metadata


def build_warmup_cosine_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> Tuple[LambdaLR, Dict[str, Any]]:
    total_steps = max(int(total_steps), 1)
    warmup_steps = int(round(float(warmup_ratio) * total_steps))
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step <= warmup_steps:
            return max(step, 1) / float(warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / float(decay_steps), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    metadata = {
        "type": "linear_warmup_cosine",
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }
    return scheduler, metadata


def current_lrs(optimizer: AdamW) -> Dict[str, float]:
    return {
        str(group.get("name", f"group_{idx}")): float(group["lr"])
        for idx, group in enumerate(optimizer.param_groups)
    }
