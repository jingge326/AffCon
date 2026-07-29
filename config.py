
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJ_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJ_ROOT / "data"


def repo_data_path(*parts: str) -> str:
    return str(DATA_ROOT.joinpath(*parts))

SUNDIAL_INPUT_TOKEN_LEN: int = 16


@dataclass
class TimeIMMConfig:
    data_path: str = repo_data_path("Time-IMM")
    datasets: List[str] = field(default_factory=list)
    dataset_presets: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "GDELT": {
                "lookback": 14,
                "horizon": 14,
                "stride": 14,
                "time_unit": "days",
                "resample_interval_sec": 1800,
                "OT": "GoldsteinScale",
                "main_eval": False,
            },
            "RepoHealth": {
                "lookback": 31,
                "horizon": 31,
                "stride": 31,
                "time_unit": "days",
                "resample_interval_sec": 86400,
                "OT": "open_issues",
            },
            "MIMIC": {
                "lookback": 24,
                "horizon": 24,
                "stride": 24,
                "time_unit": "hours",
                "resample_interval_sec": 1800,
                "OT": "Value_label_26",
            },
            "FNSPID": {
                "lookback": 31,
                "horizon": 31,
                "stride": 31,
                "time_unit": "days",
                "resample_interval_sec": 86400,
                "OT": "adj close",
            },
            "ClusterTrace": {
                "lookback": 12,
                "horizon": 12,
                "stride": 12,
                "time_unit": "hours",
                "resample_interval_sec": 900,
                "OT": "n_inst",
            },
            "StudentLife": {
                "lookback": 31,
                "horizon": 31,
                "stride": 31,
                "time_unit": "days",
                "resample_interval_sec": 86400,
                "OT": "sleep_duration",
            },
            "ILINet": {
                "lookback": 36,
                "horizon": 36,
                "stride": 4,
                "time_unit": "weeks",
                "resample_interval_sec": 604800,
                "OT": "% WEIGHTED ILI",
            },
            "CESNET": {
                "lookback": 7,
                "horizon": 7,
                "stride": 7,
                "time_unit": "days",
                "resample_interval_sec": 3600,
                "OT": "n_flows",
                "main_eval": False,
            },
            "EPA-Air": {
                "lookback": 7,
                "horizon": 7,
                "stride": 7,
                "time_unit": "days",
                "resample_interval_sec": 3600,
                "OT": "temp",
            },
        }
    )

    split_method: str = "sample"
    train_ratio: float = 0.6
    val_ratio: float = 0.2

    enable_text: bool = True
    use_text_embeddings: bool = True
    llm_model_fusion: Optional[str] = "Qwen3_E4B"
    llm_layers_fusion: Optional[int] = None
    max_length: int = 1024
    use_time_rescale: bool = True

    target_column: Optional[str] = None

    def dataset_settings(self, dataset: Optional[str] = None) -> Dict[str, Any]:
        name = dataset or (self.datasets[0] if self.datasets else None)
        preset = self.dataset_presets.get(name)
        return preset

    def __post_init__(self) -> None:
        if not self.datasets:
            self.datasets = [
                name for name, preset in self.dataset_presets.items()
                if preset.get("main_eval", True)
            ]


@dataclass
class TimeMMDConfig:
    data_path: str = repo_data_path("Time-MMD")

    train_ratio: float = 0.7
    val_ratio: float = 0.1

    use_domain_freq_windows: bool = True

    dataset_presets: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "Agriculture": {
                "frequency": "monthly",
                "lookback": 8,
                "horizons": [6, 8, 10, 12],
            },
            "Climate": {
                "frequency": "monthly",
                "lookback": 8,
                "horizons": [6, 8, 10, 12],
            },
            "Economy": {
                "frequency": "monthly",
                "lookback": 8,
                "horizons": [6, 8, 10, 12],
            },
            "Energy": {
                "frequency": "weekly",
                "lookback": 36,
                "horizons": [12, 24, 36, 48],
            },
            "Health_US": {
                "frequency": "weekly",
                "lookback": 36,
                "horizons": [12, 24, 36, 48],
            },
            "Security": {
                "frequency": "monthly",
                "lookback": 8,
                "horizons": [6, 8, 10, 12],
            },
            "SocialGood": {
                "frequency": "monthly",
                "lookback": 8,
                "horizons": [6, 8, 10, 12],
            },
            "Traffic": {
                "frequency": "monthly",
                "lookback": 8,
                "horizons": [6, 8, 10, 12],
            },
        }
    )

    lookback: int = 96

    horizon: int = 96

    use_report: bool = True
    use_search: bool = True

    scale: bool = True

    use_text_embeddings: bool = True
    llm_model_fusion: str = "Qwen3_E4B"
    llm_layers_fusion: Optional[int] = None

    input_token_len: int = SUNDIAL_INPUT_TOKEN_LEN

    def dataset_settings(self, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.dataset_presets.get(domain)


@dataclass
class CGTSFConfig:
    data_path: str = repo_data_path("CGTSF")

    lookback: int = 0
    horizon: int = 0

    train_ratio: float = 0.6
    val_ratio: float = 0.2

    data_usage_ratio: float = 1.0

    dataset_presets: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "MSPG": {
                "horizon": 96,
                "lookback": None,
                "allowed_lookback_multipliers": [2, 3, 4, 5],
            },
            "LEU": {
                "horizon": 48,
                "lookback": None,
                "allowed_lookback_multipliers": [2, 3, 4, 5],
            },
            "PTF": {
                "horizon": 24,
                "lookback": None,
                "allowed_lookback_multipliers": [2, 3, 4, 5],
            },
        }
    )

    scale: bool = True
    use_text_embeddings: bool = True

    def dataset_settings(self, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.dataset_presets.get(domain)


@dataclass
class SyntheticConfig:
    data_path: str = repo_data_path("synthetic")
    train_ratio: float = 0.9


@dataclass
class TrainConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2

    max_grad_norm: float = 1.0

    log_every: int = 1

    sundial_model_name: str = "thuml/sundial-base-128m"
    qwen_model_name: str = "Qwen/Qwen3-Embedding-4B"

    freeze_sundial: bool = True
    freeze_text: bool = True

    freeze_affflow: bool = False

    affflow_loss_weight: float = 0.0

    use_flash_attn: bool = False

    timemoe_model_name: str = "Maple728/TimeMoE-50M"

    timesfm_model_name: str = "google/timesfm-2.5-200m-pytorch"


@dataclass
class AffConConfig:
    affflow_text_concat: bool = True
    affflow_text_mode: str = "affconcat"
    affflow_prior_type: str = "standard"
    affflow_adapter: bool = False
    generator_type: str = "affflow"
    affflow_num_blocks: int = 2
    affflow_mlp_ratio: int = 2
    affflow_nfe: int = 50
    flow_init: str = "anchor"
    affflow_train_steps_min: int = 8
    affflow_train_steps_max: int = 64
    affflow_train_random_grid: bool = True
    patch_position_mode: str = "contiguous"
    text_attention_mode: str = "temporal_causal"
    text_position_mode: str = "token_aligned"
    text_injection_layout: str = "distributed"
    text_embedding_control: str = "none"
    text_embedding_random_seed: int = 0
    affiliated_token_control: str = "none"
    observed_missing_policy: str = "mask_aware"
    affflow_dummy_token: bool = False
    affflow_dummy_init: str = "zero"
    force_flip_invariance: bool = True

    timesfm_infer_is_positive: bool = False
    timesfm_decoupled_flip: bool = False

    compute_crps: bool = True
    crps_num_samples: int = 100
    crps_dump: bool = False

    synthetic_mix_ratio: float = 0.0
    synthetic_mix_decay: float = 0.5
    synthetic_mix_min_ratio: float = 1e-3


@dataclass
class Config:
    timeimm: TimeIMMConfig = field(default_factory=TimeIMMConfig)


    time_mmd: TimeMMDConfig = field(default_factory=TimeMMDConfig)
    cgtsf: CGTSFConfig = field(default_factory=CGTSFConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    affcon: AffConConfig = field(default_factory=AffConConfig)

    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
