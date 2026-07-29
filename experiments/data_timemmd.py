

from __future__ import annotations

import os
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict

from tqdm import tqdm

from config import SUNDIAL_INPUT_TOKEN_LEN
from experiments.utils import atomic_torch_save, is_embedding_file_complete




DOMAIN_META: Dict[str, Dict[str, str]] = {
    "Agriculture": {
        "target_variable": "Retail Broiler Composite (monthly chicken retail price index, USDA ERS)",
        "relevance": "Reports and search discuss broiler supply/demand, costs, and sentiment that drive retail prices.",
    },
    "Climate": {
        "target_variable": "Drought Level (monthly drought indices, NOAA/US Drought Monitor)",
        "relevance": "Texts describe precipitation, heat, and soil moisture conditions reflected in drought levels.",
    },
    "Economy": {
        "target_variable": "International Trade Balance (monthly exports minus imports, Census/BEA)",
        "relevance": "Texts discuss export/import changes, tariffs, currency, and shocks that move the trade balance.",
    },
    "Energy": {
        "target_variable": "Gasoline Prices (weekly US gasoline price, EIA)",
        "relevance": "Texts cover crude prices, refinery outages, inventories, demand, and policy impacting gas prices.",
    },
    "Environment": {
        "target_variable": "Air Quality Index (daily AQI, EPA/NY focus)",
        "relevance": "Texts describe pollution events, smoke, alerts, and mitigation affecting AQI spikes/drops.",
    },
    "Health_US": {
        "target_variable": "Influenza-like Illness Proportion (weekly ILI from CDC ILINet)",
        "relevance": "Texts report flu activity, strains, outbreaks, and mitigation shaping ILI proportions.",
    },
    "Security": {
        "target_variable": "Disaster and Emergency Grants (monthly FEMA/NOAA-related funding)",
        "relevance": "Texts describe disasters and declarations that trigger emergency grant spending spikes.",
    },
    "SocialGood": {
        "target_variable": "Unemployment Rate by Race (monthly BLS employment statistics)",
        "relevance": "Texts explain job gains/losses, recessions, policy changes affecting unemployment rates.",
    },
    "Traffic": {
        "target_variable": "Travel Volume (monthly vehicle miles traveled, FHWA)",
        "relevance": "Texts describe mobility shifts, holidays, storms, and events impacting travel volume.",
    },
}


def get_domain_meta(domain: str) -> Tuple[str, str]:
    meta = DOMAIN_META.get(domain, {})
    target_variable = meta.get("target_variable", domain)
    relevance = meta.get("relevance", "")
    return target_variable, relevance


def get_text_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if col.lower() in ("text", "content", "snippet", "description", "fact"):
            return col
    for col in df.columns:
        if col not in ("start_date", "end_date", "date"):
            return col
    return None


def build_tmmd_text_header(
    domain: str,
    source_type: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    add_header: bool = True,
) -> str:
    if not add_header:
        return ""

    target_variable, relevance = get_domain_meta(domain)

    start_str = start_date.strftime(
        "%Y-%m-%d") if pd.notna(start_date) else "?"
    end_str = end_date.strftime("%Y-%m-%d") if pd.notna(end_date) else "?"

    header_parts = [
        "[FORECASTING TASK]",
        f"Target Variable: {target_variable}",
        f"Domain: {domain}",
        "",
        "[AUXILIARY INFORMATION]",
        f"Source Type: {source_type}",
        f"Coverage Period: {start_str} to {end_str}",
    ]
    if relevance:
        header_parts.extend(["", "[RELEVANCE]", relevance])
    header_parts.extend(["", "[TEXT]"])

    return "\n".join(header_parts) + "\n"


def build_tmmd_full_text(
    domain: str,
    raw_text: str,
    source_type: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    add_header: bool = True,
) -> str:
    header = build_tmmd_text_header(
        domain, source_type, start_date, end_date, add_header)
    return header + raw_text


def build_tmmd_patch_header(domain: str, add_header: bool = True) -> str:
    if not add_header:
        return ""

    target_variable, relevance = get_domain_meta(domain)

    header_parts = [
        "[FORECASTING TASK]",
        f"Target Variable: {target_variable}",
        f"Domain: {domain}",
    ]
    if relevance:
        header_parts.extend(["", "[RELEVANCE]", relevance])

    return "\n".join(header_parts)




def map_text_to_relative_position(
    text_start: pd.Timestamp,
    window_start: pd.Timestamp,
    step_size: pd.Timedelta,
    lookback: int,
) -> float:
    if step_size.total_seconds() > 0:
        rel_idx = (text_start - window_start) / step_size
    else:
        rel_idx = 0.0
    return max(0.0, min(float(lookback - 1), float(rel_idx)))


def assign_tmmd_texts_to_patches(
    texts_with_rel_idx: List[Tuple[float, Any]],
    input_token_len: int,
) -> Dict[int, List[Tuple[float, Any]]]:
    patches: Dict[int, List[Tuple[float, Any]]] = defaultdict(list)

    for rel_idx, payload in texts_with_rel_idx:
        patch_idx = int(rel_idx // input_token_len)
        patches[patch_idx].append((rel_idx, payload))

    for patch_idx in patches:
        patches[patch_idx].sort(key=lambda x: x[0])

    return dict(patches)


def compute_tmmd_patch_params(
    lookback: int,
    input_token_len: int,
) -> int:
    return int(np.ceil(lookback / input_token_len))


@dataclass
class TimeMMDWindow:
    x: np.ndarray
    y: np.ndarray
    text: str
    domain: str
    start_idx: int
    end_idx: int
    has_text: bool = True


class TimeMMDDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        lookback: int,
        horizon: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        use_report: bool = True,
        use_search: bool = True,
        domains: Optional[List[str]] = None,
        cache_dir: Optional[Path | str] = None,
        scale: bool = True,
        domain_window_overrides: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.lookback = lookback
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_report = use_report
        self.use_search = use_search
        self.scale = scale
        self.domain_window_overrides = domain_window_overrides or {}

        numerical_root = os.path.join(data_path, "numerical")
        textual_root = os.path.join(data_path, "textual")

        if domains is None or len(domains) == 0:
            domains = sorted(
                d
                for d in os.listdir(numerical_root)
                if os.path.isdir(os.path.join(numerical_root, d))
            )

        self.domains: List[str] = domains
        self.domain_lengths: Dict[str, int] = {}
        self.domain_scalers: Dict[str, Tuple[float, float]] = {}
        self.num_channels: int = 1

        cache_dir = Path(cache_dir) if cache_dir is not None else Path(
            self.data_path) / "cache_data"
        cache_path = self._maybe_cache_path(cache_dir)
        if cache_path is not None and cache_path.exists():
            payload = None
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                print(
                    f"[TimeMMDDataset] Cache load failed ({e}); rebuilding: {cache_path}")

            if payload is not None and payload.get("params") == self._cache_params():
                self.windows = payload["windows"]
                self.domain_lengths = payload.get("domain_lengths", {})
                self.domain_scalers = payload.get("domain_scalers", {})
                print(
                    f"[TimeMMDDataset] Loaded cached windows ({len(self.windows)}) from {cache_path}")
                return
            else:
                print(
                    f"[TimeMMDDataset] Cache params mismatch or unavailable, rebuilding: {cache_path}")

        self.windows: List[TimeMMDWindow] = []
        skipped_nonfinite = 0
        skipped_by_domain: dict[str, int] = {}

        for domain in self.domains:
            dom_lookback, dom_horizon = self._domain_window(domain)
            print(f"[TimeMMDDataset] Building windows for domain: {domain}")
            num_path = os.path.join(numerical_root, domain, f"{domain}.csv")
            rep_path = os.path.join(
                textual_root, domain, f"{domain}_report.csv")
            search_path = os.path.join(
                textual_root, domain, f"{domain}_search.csv")

            if not os.path.exists(num_path):
                print(
                    f"  [WARN] Numerical CSV not found: {num_path}, skipping")
                continue

            df_num = pd.read_csv(num_path)
            df_num = self._ensure_date_columns(
                df_num, num_path, kind="numeric")
            if df_num is None:
                print(
                    f"  [WARN] Missing usable date columns in {num_path}, skipping domain {domain}")
                continue
            if "OT" not in df_num.columns:
                print(
                    f"  [WARN] Numerical CSV missing 'OT' column: {num_path}, skipping domain {domain}")
                continue

            df_num["start_date"] = pd.to_datetime(df_num["start_date"])
            df_num["end_date"] = pd.to_datetime(df_num["end_date"])

            df_num = df_num.sort_values("start_date").drop_duplicates(
                subset=["start_date"], keep="last")
            freq = self._infer_freq(df_num, domain)
            df_num = self._regularize_frequency(df_num, freq)

            series = df_num["OT"].astype(np.float32).to_numpy()
            n = len(series)
            self.domain_lengths[domain] = n

            train_end, val_end = self._split_boundaries(n)
            scale_mean, scale_std = self._compute_scaler(series, train_end)
            self.domain_scalers[domain] = (scale_mean, scale_std)
            if self.scale:
                series_proc = (series - scale_mean) / scale_std
            else:
                series_proc = series

            df_rep = None
            df_search = None

            if self.use_report and os.path.exists(rep_path):
                df_rep = pd.read_csv(rep_path)
                df_rep = self._ensure_date_columns(
                    df_rep, rep_path, kind="text")
                if df_rep is not None:
                    for col in ("start_date", "end_date"):
                        if col in df_rep.columns:
                            df_rep[col] = pd.to_datetime(df_rep[col])
            if self.use_search and os.path.exists(search_path):
                df_search = pd.read_csv(search_path)
                df_search = self._ensure_date_columns(
                    df_search, search_path, kind="text")
                if df_search is not None:
                    for col in ("start_date", "end_date"):
                        if col in df_search.columns:
                            df_search[col] = pd.to_datetime(df_search[col])

            for t in range(dom_lookback - 1, n - dom_horizon):
                x = series_proc[t - dom_lookback +
                                1: t + 1]
                y = series_proc[t + 1: t + 1 + dom_horizon]

                window_start_time = df_num.loc[t -
                                               dom_lookback + 1, "start_date"]
                window_end_time = df_num.loc[t + dom_horizon, "end_date"]

                text_snippets: List[str] = []

                if df_rep is not None:
                    snippets = self._collect_text_for_window(
                        df_rep, window_start_time, window_end_time, source_label="report"
                    )
                    text_snippets.extend(snippets)

                if df_search is not None:
                    snippets = self._collect_text_for_window(
                        df_search, window_start_time, window_end_time, source_label="search result"
                    )
                    text_snippets.extend(snippets)

                has_text = len(text_snippets) > 0
                if has_text:
                    text = "\n".join(text_snippets)
                else:
                    text = ""

                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    skipped_nonfinite += 1
                    skipped_by_domain[domain] = skipped_by_domain.get(
                        domain, 0) + 1
                    continue

                self.windows.append(
                    TimeMMDWindow(
                        x=x,
                        y=y,
                        text=text,
                        domain=domain,
                        start_idx=t,
                        end_idx=t + dom_horizon,
                        has_text=has_text,
                    )
                )

        print(
            f"[TimeMMDDataset] Total windows across all domains: {len(self.windows)}")
        if skipped_nonfinite > 0:
            print(
                f"[TimeMMDDataset] Skipped windows with non-finite values: {skipped_nonfinite}")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "params": self._cache_params(),
                        "windows": self.windows,
                        "domain_lengths": self.domain_lengths,
                        "domain_scalers": self.domain_scalers,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"[TimeMMDDataset] Cached windows to {cache_path}")

            filter_log = {
                "params": self._cache_params(),
                "kept_windows": len(self.windows),
                "skipped_nonfinite_total": skipped_nonfinite,
                "skipped_nonfinite_by_domain": skipped_by_domain,
            }
            filter_log_path = cache_path.with_name(
                cache_path.stem + "_nonfinite.json")
            with open(filter_log_path, "w", encoding="utf-8") as f:
                json.dump(filter_log, f, ensure_ascii=False, indent=2)
            print(
                f"[TimeMMDDataset] Wrote non-finite filter log to {filter_log_path}")

    @staticmethod
    def _ensure_date_columns(df: pd.DataFrame, path: str, kind: str) -> Optional[pd.DataFrame]:
        rename_map = {}
        if "start_date" not in df.columns:
            for cand in ("date", "start", "start_time", "timestamp"):
                if cand in df.columns:
                    rename_map[cand] = "start_date"
                    break
        if "end_date" not in df.columns:
            for cand in ("end_date", "date.1", "end", "end_time", "stop_time", "timestamp_end"):
                if cand in df.columns:
                    rename_map[cand] = "end_date"
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

        missing = [c for c in ("start_date", "end_date")
                   if c not in df.columns]
        if missing:
            print(f"  [WARN] {kind} CSV {path} missing {missing}")
            return None
        return df

    @staticmethod
    def _infer_freq(df: pd.DataFrame, domain: str) -> Optional[str]:
        dates = df["start_date"].sort_values().drop_duplicates()
        if len(dates) < 2:
            return None
        diffs = dates.diff().dropna()
        mode_days = int(diffs.dt.days.mode().iloc[0])
        if 27 <= mode_days <= 31:
            return "MS"
        if mode_days == 7:
            return "7D"
        if mode_days == 1:
            return "D"
        return f"{mode_days}D"

    @staticmethod
    def _regularize_frequency(df: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
        if freq is None:
            return df.reset_index(drop=True)

        df = df.sort_values("start_date").drop_duplicates(
            subset=["start_date"], keep="last")
        df = df.set_index("start_date")

        full_index = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_index)

        df = df.ffill()

        df["start_date"] = df.index

        if "end_date" in df.columns:
            if freq == "MS":
                df["end_date"] = df.index + pd.offsets.MonthEnd(1)
            elif freq.endswith("D"):
                try:
                    step_days = int(freq.rstrip("D"))
                except ValueError:
                    step_days = 1
                step_days = max(step_days, 1)
                df["end_date"] = df.index + pd.Timedelta(days=step_days - 1)
            df["end_date"] = df["end_date"].ffill()

        return df.reset_index(drop=True)

    def _cache_params(self) -> dict:
        return {
            "data_path": os.path.abspath(self.data_path),
            "lookback": self.lookback,
            "horizon": self.horizon,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "use_report": self.use_report,
            "use_search": self.use_search,
            "domains": self.domains,
            "filter_nonfinite": True,
            "scale": self.scale,
            "domain_window_overrides": sorted(self.domain_window_overrides.items()),
            "regularize_missing": True,
        }

    def _maybe_cache_path(self, cache_dir: Optional[Path | str]) -> Optional[Path]:
        if cache_dir is None:
            return None
        params = self._cache_params()
        key = hashlib.md5(json.dumps(params, sort_keys=True).encode(
            "utf-8")).hexdigest()[:12]
        cache_dir_path = Path(cache_dir)
        return cache_dir_path / f"time_mmd_windows_{key}.pkl"

    def _split_boundaries(self, series_length: int) -> Tuple[int, int]:
        train_end = int(self.train_ratio * series_length)
        val_end = int((self.train_ratio + self.val_ratio) * series_length)
        val_end = min(val_end, series_length)
        train_end = min(train_end, val_end)
        return train_end, val_end

    @staticmethod
    def _compute_scaler(series: np.ndarray, train_end: int) -> Tuple[float, float]:
        train_slice = series[:train_end] if train_end > 0 else series
        mean = float(np.mean(train_slice))
        std = float(np.std(train_slice))
        if std <= 0:
            std = 1.0
        return mean, std

    @staticmethod
    def _collect_text_for_window(
        df_txt: pd.DataFrame,
        window_start_time: pd.Timestamp,
        window_end_time: pd.Timestamp,
        source_label: str,
    ) -> List[str]:
        if "start_date" not in df_txt.columns or "end_date" not in df_txt.columns:
            return []

        mask = (df_txt["start_date"] <= window_end_time) & (
            df_txt["end_date"] >= window_start_time
        )
        df_sub = df_txt.loc[mask].sort_values("start_date")

        snippets: List[str] = []
        rows = df_sub
        for _, row in rows.iterrows():
            parts = []
            if "fact" in row and isinstance(row["fact"], str):
                parts.append(str(row["fact"]))
            if "pred" in row and isinstance(row["pred"], str):
                parts.append(str(row["pred"]))
            if len(parts) == 0:
                for col in df_txt.columns:
                    val = row[col]
                    if isinstance(val, str):
                        parts.append(val)
            content = " ".join(parts).strip()
            if len(content) == 0:
                continue

            start_date = row.get("start_date")
            end_date = row.get("end_date")
            start_str = None
            end_str = None
            try:
                if pd.notna(start_date):
                    start_str = pd.to_datetime(start_date).date().isoformat()
            except Exception:
                start_str = None
            try:
                if pd.notna(end_date):
                    end_str = pd.to_datetime(end_date).date().isoformat()
            except Exception:
                end_str = None

            if start_str and end_str:
                prefix = f"This {source_label} was collected between {start_str} and {end_str}: "
            elif start_str:
                prefix = f"This {source_label} was collected on {start_str}: "
            elif end_str:
                prefix = f"This {source_label} was collected by {end_str}: "
            else:
                prefix = f"This {source_label} says: "

            snippets.append(prefix + content)
        return snippets

    def _domain_window(self, domain: str) -> Tuple[int, int]:
        if domain in self.domain_window_overrides:
            return self.domain_window_overrides[domain]
        return self.lookback, self.horizon

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        w = self.windows[idx]
        x = torch.from_numpy(w.x)
        y = torch.from_numpy(w.y)
        return x, y, w.text


def build_time_mmd_dataloaders(
    data_path: str,
    lookback: int,
    horizon: int,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    use_report: bool = True,
    use_search: bool = True,
    domains: Optional[List[str]] = None,
    cache_dir: Optional[Path | str] = None,
    scale: bool = True,
    domain_window_overrides: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cache_dir = cache_dir or (Path(data_path) / "cache_data")
    dataset = TimeMMDDataset(
        data_path=data_path,
        lookback=lookback,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_report=use_report,
        use_search=use_search,
        domains=domains,
        cache_dir=cache_dir,
        scale=scale,
        domain_window_overrides=domain_window_overrides,
    )

    domain_to_indices: dict[str, List[int]] = {d: [] for d in dataset.domains}
    for idx, w in enumerate(dataset.windows):
        domain_to_indices[w.domain].append(idx)

    for domain, idxs in domain_to_indices.items():
        if domain not in dataset.domain_lengths:
            max_end = max(
                dataset.windows[i].end_idx for i in idxs) if idxs else -1
            dataset.domain_lengths[domain] = max_end + 1

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for domain in dataset.domains:
        idxs = domain_to_indices.get(domain, [])
        n = dataset.domain_lengths.get(domain, 0)
        if n == 0:
            continue
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        for i in idxs:
            split_idx = dataset.windows[i].start_idx
            if split_idx < train_end:
                train_indices.append(i)
            elif split_idx < val_end:
                val_indices.append(i)
            else:
                test_indices.append(i)

    def filter_has_text(indices):
        return [i for i in indices if dataset.windows[i].has_text]

    total_before = len(train_indices) + len(val_indices) + len(test_indices)
    train_indices = filter_has_text(train_indices)
    val_indices = filter_has_text(val_indices)
    test_indices = filter_has_text(test_indices)
    total_after = len(train_indices) + len(val_indices) + len(test_indices)

    print(
        f"[build_time_mmd_dataloaders] Filtered to text-available samples: "
        f"{total_after}/{total_before} ({100*total_after/total_before:.1f}%)"
    )
    print(
        f"[build_time_mmd_dataloaders] Split sizes: "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    for subset in (train_ds, val_ds, test_ds):
        if subset is not None:
            setattr(subset, "num_channels", dataset.num_channels)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    for loader in (train_loader, val_loader, test_loader):
        if loader is not None:
            setattr(loader, "num_channels", dataset.num_channels)
    return train_loader, val_loader, test_loader




class TimestepEmbeddingCache:

    def __init__(
        self,
        cache_dir: Path,
        llm_model: str = "Qwen3_E4B",
        llm_layers: Optional[int] = None,
        data_root: Optional[Path] = None,
        device: str = "cuda",
        add_header: bool = True,
        encode_batch_size: int = 8,
        max_length: int = 1024,
        dtype: torch.dtype = torch.float16,
        use_report: bool = True,
        use_search: bool = True,
        domains: Optional[List[str]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.data_root = data_root or self.cache_dir.parent
        self.device = device
        self.add_header = add_header
        self.encode_batch_size = encode_batch_size
        self.max_length = max_length
        self.dtype = dtype
        self.use_report = use_report
        self.use_search = use_search

        self._tokenizer = None
        self._llm_model = None

        self.d_txt: Optional[int] = None
        self.data: Dict[str, Dict] = {}

        layers_str = llm_layers if llm_layers is not None else "full"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if domains is not None:
            domains = sorted(domains)
        else:
            textual_root = self.data_root / "textual"
            if not textual_root.exists():
                print(
                    f"[TimestepEmbeddingCache] Warning: textual directory not found: {textual_root}")
                return
            domains = sorted(d.name for d in textual_root.iterdir() if d.is_dir())

        for domain in domains:
            emb_file = self._get_embedding_filename(domain)
            emb_path = self.cache_dir / emb_file

            if not is_embedding_file_complete(emb_path):
                self._ensure_embeddings(domain, emb_path)

            if is_embedding_file_complete(emb_path):
                try:
                    payload = torch.load(
                        emb_path, map_location="cpu", weights_only=False)
                    embeddings = payload["embeddings"]

                    if self.d_txt is None:
                        self.d_txt = embeddings.shape[-1]

                    start_dates = [pd.to_datetime(d)
                                   for d in payload["start_dates"]]
                    end_dates = [pd.to_datetime(d)
                                 for d in payload["end_dates"]]

                    self.data[domain] = {
                        "embeddings": embeddings,
                        "start_dates": start_dates,
                        "end_dates": end_dates,
                    }
                    print(
                        f"[TimestepEmbeddingCache] Loaded {domain}: {len(embeddings)} embeddings")
                except Exception as e:
                    print(
                        f"[TimestepEmbeddingCache] Failed to load {emb_path}: {e}")

        print(
            f"[TimestepEmbeddingCache] Loaded {len(self.data)} domains, d_txt={self.d_txt}")

    def _get_embedding_filename(self, domain: str) -> str:
        layers_str = self.llm_layers if self.llm_layers is not None else "full"
        return f"{domain}_text_embeddings_model={self.llm_model}_layers={layers_str}.pt"

    def _ensure_embeddings(self, domain: str, emb_path: Path) -> None:
        print(f"[TimestepEmbeddingCache] Computing embeddings for {domain}...")

        textual_root = self.data_root / "textual"
        rep_path = textual_root / domain / f"{domain}_report.csv"
        search_path = textual_root / domain / f"{domain}_search.csv"

        texts: List[str] = []
        start_dates: List[str] = []
        end_dates: List[str] = []
        source_types: List[str] = []

        for csv_path, source_type, use_flag in [
            (rep_path, "report", self.use_report),
            (search_path, "search", self.use_search),
        ]:
            if not use_flag or not csv_path.exists():
                continue

            df_txt = pd.read_csv(csv_path)
            df_txt = TimeMMDDataset._ensure_date_columns(
                df_txt, str(csv_path), kind="text")
            if df_txt is None:
                continue

            for col in ("start_date", "end_date"):
                if col in df_txt.columns:
                    df_txt[col] = pd.to_datetime(df_txt[col])

            text_col = get_text_column(df_txt)
            if text_col is None:
                continue

            for _, row in df_txt.iterrows():
                raw_text = str(row[text_col]).strip()
                if not raw_text or raw_text.lower() == "nan":
                    continue

                final_text = build_tmmd_full_text(
                    domain, raw_text, source_type,
                    row.get("start_date"), row.get("end_date"),
                    add_header=self.add_header,
                )

                texts.append(final_text)
                start_dates.append(str(row.get("start_date", "")))
                end_dates.append(str(row.get("end_date", "")))
                source_types.append(source_type)

        if len(texts) == 0:
            print(
                f"[TimestepEmbeddingCache] Skipping {domain}: no valid text entries.")
            return

        print(
            f"[TimestepEmbeddingCache] Encoding {len(texts)} text entries for {domain}...")

        if self._tokenizer is None:
            from experiments.utils import load_llm
            print(
                f"[TimestepEmbeddingCache] Loading LLM model: {self.llm_model}")
            self._tokenizer, self._llm_model = load_llm(
                self.llm_model,
                self.llm_layers,
                self.device,
                use_device_map=False,
            )

        embeddings = self._encode_texts_batch(texts, desc=domain)
        embeddings = embeddings.to(dtype=self.dtype)

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "domain": domain,
            "embeddings": embeddings,
            "start_dates": start_dates,
            "end_dates": end_dates,
            "source_types": source_types,
            "dtype": str(self.dtype).replace("torch.", ""),
            "add_header": self.add_header,
            "llm_model": self.llm_model,
            "llm_layers": self.llm_layers,
        }
        atomic_torch_save(payload, emb_path)
        print(
            f"[TimestepEmbeddingCache] Saved {len(texts)} embeddings to {emb_path}")

    def _encode_texts_batch(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        from experiments.utils import embed_notes

        outputs: List[torch.Tensor] = []

        for i in tqdm(range(0, len(texts), self.encode_batch_size), desc=desc, leave=False):
            batch_texts = texts[i: i + self.encode_batch_size]

            for j, text in enumerate(batch_texts):
                token_count = len(self._tokenizer.encode(
                    text, add_special_tokens=True))
                if token_count > self.max_length:
                    print(
                        f"  [TRUNCATED] idx={i+j}, {token_count} -> {self.max_length} tokens")

            notes_batch = [[text] for text in batch_texts]
            emb, _ = embed_notes(notes_batch, self._tokenizer,
                                 self._llm_model, max_length=self.max_length)
            outputs.append(emb.squeeze(1).cpu())
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        if outputs:
            return torch.cat(outputs, dim=0)
        return torch.empty((0, 0), dtype=torch.float32)

    def get_entries_for_window(
        self,
        domain: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> List[Tuple[pd.Timestamp, torch.Tensor]]:
        if domain not in self.data:
            return []

        domain_data = self.data[domain]
        embeddings = domain_data["embeddings"]
        start_dates = domain_data["start_dates"]
        end_dates = domain_data["end_dates"]

        results = []
        for i, (t_start, t_end) in enumerate(zip(start_dates, end_dates)):
            if t_start <= window_end and t_end >= window_start:
                results.append((t_start, embeddings[i]))

        return results


@dataclass
class TextEntry:
    text: str
    source_type: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp


class PatchEmbeddingCache:

    def __init__(
        self,
        cache_dir: Path,
        input_token_len: int = SUNDIAL_INPUT_TOKEN_LEN,
        llm_model: str = "Qwen3_E4B",
        llm_layers: Optional[int] = None,
        data_root: Optional[Path] = None,
        device: str = "cuda",
        add_header: bool = True,
        encode_batch_size: int = 4,
        max_length: int = 4096,
        dtype: torch.dtype = torch.float16,
        use_report: bool = True,
        use_search: bool = True,
        lookback: int = 96,
        horizon: int = 24,
        domains: Optional[List[str]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.input_token_len = input_token_len
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.data_root = data_root or self.cache_dir.parent
        self.device = device
        self.add_header = add_header
        self.encode_batch_size = encode_batch_size
        self.max_length = max_length
        self.dtype = dtype
        self.use_report = use_report
        self.use_search = use_search
        self.lookback = lookback
        self.horizon = horizon
        self._domains = domains

        self._tokenizer = None
        self._llm_model = None

        self.d_txt: Optional[int] = None
        self.n_patches: Optional[int] = None
        self.data: Dict[str, Dict[str, Dict]] = {}

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self._domains is not None:
            domains = sorted(self._domains)
        else:
            textual_root = self.data_root / "textual"
            if not textual_root.exists():
                print(
                    f"[PatchEmbeddingCache] Warning: textual directory not found: {textual_root}")
                return
            domains = sorted(d.name for d in textual_root.iterdir() if d.is_dir())

        for domain in domains:
            emb_file = self._get_embedding_filename(domain)
            emb_path = self.cache_dir / emb_file

            if not is_embedding_file_complete(emb_path):
                self._ensure_embeddings(domain, emb_path)

            if is_embedding_file_complete(emb_path):
                try:
                    payload = torch.load(
                        emb_path, map_location="cpu", weights_only=False)
                    window_embeddings = payload.get("window_embeddings", {})

                    if not window_embeddings:
                        print(
                            f"[PatchEmbeddingCache] Warning: no window embeddings for {domain}")
                        continue

                    if self.d_txt is None or self.n_patches is None:
                        first_window = next(iter(window_embeddings.values()))
                        patch_embs = first_window.get("patch_embeddings", {})
                        if patch_embs:
                            first_emb = next(iter(patch_embs.values()))
                            self.d_txt = first_emb.shape[-1]
                        self.n_patches = first_window.get("n_patches")

                    self.data[domain] = window_embeddings
                    print(
                        f"[PatchEmbeddingCache] Loaded {domain}: {len(window_embeddings)} windows")
                except Exception as e:
                    print(
                        f"[PatchEmbeddingCache] Failed to load {emb_path}: {e}")

        print(
            f"[PatchEmbeddingCache] Loaded {len(self.data)} domains, "
            f"d_txt={self.d_txt}, n_patches={self.n_patches}"
        )

    def _get_embedding_filename(self, domain: str) -> str:
        layers_str = self.llm_layers if self.llm_layers is not None else "full"
        return (
            f"{domain}_patch_text_embeddings"
            f"_model={self.llm_model}"
            f"_layers={layers_str}"
            f"_patchlen={self.input_token_len}"
            f"_lb={self.lookback}_hz={self.horizon}.pt"
        )

    def _ensure_embeddings(self, domain: str, emb_path: Path) -> None:
        print(
            f"[PatchEmbeddingCache] Computing patch embeddings for {domain}...")

        numerical_root = self.data_root / "numerical"
        num_path = numerical_root / domain / f"{domain}.csv"
        if not num_path.exists():
            print(
                f"[PatchEmbeddingCache] Skipping {domain}: numerical data not found")
            return

        df_num = pd.read_csv(num_path)
        df_num = TimeMMDDataset._ensure_date_columns(
            df_num, str(num_path), kind="numeric")
        if df_num is None:
            print(f"[PatchEmbeddingCache] Skipping {domain}: no date columns")
            return

        df_num["start_date"] = pd.to_datetime(df_num["start_date"])
        df_num["end_date"] = pd.to_datetime(df_num["end_date"])
        df_num = df_num.sort_values("start_date").reset_index(drop=True)

        domain_texts = self._load_domain_texts(domain)
        if not domain_texts:
            print(
                f"[PatchEmbeddingCache] Skipping {domain}: no texts available")
            return

        n_patches = compute_tmmd_patch_params(
            self.lookback, self.input_token_len)
        print(
            f"[PatchEmbeddingCache] {domain}: lookback={self.lookback}, n_patches={n_patches}, {len(domain_texts)} texts")

        if self._tokenizer is None:
            from experiments.utils import load_llm
            print(f"[PatchEmbeddingCache] Loading LLM model: {self.llm_model}")
            self._tokenizer, self._llm_model = load_llm(
                self.llm_model,
                self.llm_layers,
                self.device,
                use_device_map=False,
            )

        n_rows = len(df_num)
        total_window = self.lookback + self.horizon

        all_window_embeddings = {}
        total_windows = 0
        total_patches_with_text = 0

        for end_idx in tqdm(range(total_window - 1, n_rows), desc=f"{domain} windows"):
            start_idx = end_idx - self.horizon
            hist_start_idx = start_idx - self.lookback + 1

            if hist_start_idx < 0:
                continue

            window_start_date = df_num.loc[hist_start_idx, "start_date"]
            window_end_date = df_num.loc[end_idx, "end_date"]

            if hist_start_idx + 1 < len(df_num):
                step_size = df_num.loc[hist_start_idx +
                                       1, "start_date"] - window_start_date
            else:
                step_size = pd.Timedelta(days=1)

            window_texts = self._get_texts_for_window(
                domain_texts, window_start_date, window_end_date)
            if not window_texts:
                continue

            texts_with_rel_idx = []
            for t in window_texts:
                rel_idx = map_text_to_relative_position(
                    t["start_date"], window_start_date, step_size, self.lookback
                )
                text_entry = TextEntry(
                    text=t["text"],
                    source_type=t["source_type"],
                    start_date=t["start_date"],
                    end_date=t["end_date"],
                )
                texts_with_rel_idx.append((rel_idx, text_entry))

            patch_texts = assign_tmmd_texts_to_patches(
                texts_with_rel_idx, self.input_token_len)
            if not patch_texts:
                continue

            concat_texts = self._concatenate_patch_texts(patch_texts, domain)

            window_id = f"{domain}_w{start_idx}"
            patch_embeddings = self._encode_patch_texts(
                concat_texts, domain, window_id)

            if patch_embeddings:
                all_window_embeddings[window_id] = {
                    "patch_embeddings": patch_embeddings,
                    "patch_texts": concat_texts,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "n_patches": n_patches,
                }
                total_windows += 1
                total_patches_with_text += len(patch_embeddings)

        if not all_window_embeddings:
            print(f"[PatchEmbeddingCache] No windows with text for {domain}")
            return

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "domain": domain,
            "window_embeddings": all_window_embeddings,
            "settings": {
                "lookback": self.lookback,
                "horizon": self.horizon,
                "input_token_len": self.input_token_len,
                "n_patches": n_patches,
            },
            "llm_model": self.llm_model,
            "llm_layers": self.llm_layers,
            "max_length": self.max_length,
        }
        atomic_torch_save(payload, emb_path)
        print(
            f"[PatchEmbeddingCache] Saved {total_windows} windows, {total_patches_with_text} patches to {emb_path}")

    def _load_domain_texts(self, domain: str) -> List[Dict]:
        textual_root = self.data_root / "textual"
        texts = []

        for source_type, use_flag in [("report", self.use_report), ("search", self.use_search)]:
            if not use_flag:
                continue

            csv_path = textual_root / domain / f"{domain}_{source_type}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            df = TimeMMDDataset._ensure_date_columns(
                df, str(csv_path), kind="text")
            if df is None:
                continue

            text_col = get_text_column(df)
            if text_col is None:
                continue

            df["start_date"] = pd.to_datetime(df["start_date"])
            df["end_date"] = pd.to_datetime(df["end_date"])

            for _, row in df.iterrows():
                raw_text = str(row[text_col]).strip()
                if not raw_text or raw_text.lower() == "nan":
                    continue
                texts.append({
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "text": raw_text,
                    "source_type": source_type,
                })

        return texts

    def _get_texts_for_window(
        self,
        texts: List[Dict],
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> List[Dict]:
        return [t for t in texts if t["start_date"] <= window_end and t["end_date"] >= window_start]

    def _concatenate_patch_texts(
        self,
        patch_texts: Dict[int, List[Tuple[float, TextEntry]]],
        domain: str,
    ) -> Dict[int, str]:
        result = {}
        shared_header = build_tmmd_patch_header(domain, self.add_header)

        for patch_idx, texts in patch_texts.items():
            formatted_entries = []
            for entry_num, (_, payload) in enumerate(texts, start=1):
                start_str = payload.start_date.strftime(
                    "%Y-%m-%d") if pd.notna(payload.start_date) else "?"
                end_str = payload.end_date.strftime(
                    "%Y-%m-%d") if pd.notna(payload.end_date) else "?"
                if self.add_header:
                    entry_str = f"[TEXT {entry_num}]\nSource: {payload.source_type} | Period: {start_str} to {end_str}\n{payload.text}"
                else:
                    entry_str = payload.text
                formatted_entries.append(entry_str)

            text_block = "\n\n".join(formatted_entries)

            if self.add_header and shared_header:
                result[patch_idx] = shared_header + "\n\n" + text_block
            else:
                result[patch_idx] = text_block

        return result

    def _encode_patch_texts(
        self,
        patch_texts: Dict[int, str],
        domain: str,
        window_id: str,
    ) -> Dict[int, torch.Tensor]:
        from experiments.utils import embed_notes

        items = [(idx, text)
                 for idx, text in patch_texts.items() if text.strip()]
        if not items:
            return {}

        indices, texts = zip(*items)

        for i, text in enumerate(texts):
            token_count = len(self._tokenizer.encode(
                text, add_special_tokens=True))
            if token_count > self.max_length:
                print(
                    f"  [TRUNCATED] {domain} {window_id} patch={indices[i]}, {token_count} -> {self.max_length} tokens")

        result = {}
        for i in range(0, len(texts), self.encode_batch_size):
            batch_indices = indices[i:i + self.encode_batch_size]
            batch_texts = texts[i:i + self.encode_batch_size]

            notes_batch = [[t] for t in batch_texts]
            emb, _ = embed_notes(notes_batch, self._tokenizer,
                                 self._llm_model, max_length=self.max_length)

            for j, idx in enumerate(batch_indices):
                result[idx] = emb[j].squeeze(0).cpu().to(dtype=self.dtype)

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        return result

    def get_patch_embeddings_for_window(
        self,
        domain: str,
        start_idx: int,
    ) -> Optional[Dict[int, torch.Tensor]]:
        if domain not in self.data:
            return None

        window_id = f"{domain}_w{start_idx}"
        window_data = self.data[domain].get(window_id)

        if window_data is None:
            return None

        return window_data.get("patch_embeddings")


class TimeMMDTimestepDataset(Dataset):

    def __init__(
        self,
        base: TimeMMDDataset,
        indices: List[int],
        emb_cache: Optional[TimestepEmbeddingCache],
        domain_date_info: Dict[str, pd.DataFrame],
    ) -> None:
        super().__init__()
        self.base = base
        self.indices = indices
        self.emb_cache = emb_cache
        self.domain_date_info = domain_date_info

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        w = self.base.windows[idx]

        lookback = len(w.x)
        horizon = len(w.y)

        df_num = self.domain_date_info.get(w.domain)

        mapped_notes: List[Tuple[float, torch.Tensor]] = []

        if df_num is not None and self.emb_cache is not None:
            hist_start_idx = w.start_idx - lookback + 1

            window_start_date = df_num.loc[hist_start_idx, "start_date"]
            window_end_date = df_num.loc[w.end_idx, "end_date"]

            if hist_start_idx + 1 < len(df_num):
                step_size = df_num.loc[hist_start_idx +
                                       1, "start_date"] - window_start_date
            else:
                step_size = pd.Timedelta(days=1)

            text_entries = self.emb_cache.get_entries_for_window(
                w.domain, window_start_date, window_end_date
            )

            for text_start, embedding in text_entries:
                if step_size.total_seconds() > 0:
                    rel_idx = (text_start - window_start_date) / step_size
                else:
                    rel_idx = 0.0
                rel_idx = max(0.0, min(float(lookback - 1), float(rel_idx)))
                mapped_notes.append((rel_idx, embedding))

        return {
            "x": torch.from_numpy(w.x).float(),
            "y": torch.from_numpy(w.y).float(),
            "lookback": lookback,
            "horizon": horizon,
            "mapped_notes": mapped_notes,
            "domain": w.domain,
            "start_idx": w.start_idx,
            "end_idx": w.end_idx,
        }


class TimeMMDPatchDataset(Dataset):

    def __init__(
        self,
        base: TimeMMDDataset,
        indices: List[int],
        emb_cache: Optional[PatchEmbeddingCache],
        input_token_len: int = SUNDIAL_INPUT_TOKEN_LEN,
    ) -> None:
        super().__init__()
        self.base = base
        self.indices = indices
        self.emb_cache = emb_cache
        self.input_token_len = input_token_len
        if len(base.windows) > 0:
            sample_lookback = len(base.windows[0].x)
            self.n_patches = compute_tmmd_patch_params(
                sample_lookback, input_token_len)
        else:
            self.n_patches = 1

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        w = self.base.windows[idx]

        lookback = len(w.x)
        horizon = len(w.y)

        patch_embeddings: Optional[Dict[int, torch.Tensor]] = None

        if self.emb_cache is not None:
            patch_embeddings = self.emb_cache.get_patch_embeddings_for_window(
                w.domain, w.start_idx
            )

        return {
            "x": torch.from_numpy(w.x).float(),
            "y": torch.from_numpy(w.y).float(),
            "lookback": lookback,
            "horizon": horizon,
            "patch_embeddings": patch_embeddings,
            "n_patches": self.n_patches,
            "domain": w.domain,
            "start_idx": w.start_idx,
            "end_idx": w.end_idx,
        }


def collate_timemmd_timestep(batch: List[Dict[str, Any]], d_txt: Optional[int] = None) -> Dict[str, Any]:
    B = len(batch)
    lookback = batch[0]["lookback"]
    horizon = batch[0]["horizon"]

    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)

    if d_txt is None:
        for b in batch:
            if b["mapped_notes"]:
                d_txt = b["mapped_notes"][0][1].shape[0]
                break

    if d_txt is not None:
        text_tp_list = []
        text_emb_list = []

        for b in batch:
            if b["mapped_notes"]:
                sample_tps = torch.tensor(
                    [rel_idx for rel_idx, _ in b["mapped_notes"]], dtype=torch.float32)
                sample_embs = torch.stack(
                    [emb for _, emb in b["mapped_notes"]], dim=0)
            else:
                sample_tps = torch.zeros(0, dtype=torch.float32)
                sample_embs = torch.zeros(0, d_txt, dtype=torch.float32)
            text_tp_list.append(sample_tps)
            text_emb_list.append(sample_embs)

        max_n_txt = max(t.size(0) for t in text_tp_list)
        max_n_txt = max(max_n_txt, 1)

        text_tp = torch.zeros(B, max_n_txt)
        text_emb = torch.zeros(B, max_n_txt, d_txt)
        text_mask = torch.zeros(B, max_n_txt)

        for i, (tp_i, emb_i) in enumerate(zip(text_tp_list, text_emb_list)):
            n_txt = tp_i.size(0)
            if n_txt > 0:
                text_tp[i, :n_txt] = tp_i
                text_emb[i, :n_txt, :] = emb_i
                text_mask[i, :n_txt] = 1.0
    else:
        text_emb = None
        text_tp = None
        text_mask = None

    result = {
        "observed_data": x.unsqueeze(-1),
        "observed_mask": torch.ones(B, lookback, 1),
        "observed_tp": torch.arange(lookback, dtype=torch.float).unsqueeze(0).expand(B, -1),
        "data_to_predict": y.unsqueeze(-1),
        "mask_predicted_data": torch.ones(B, horizon, 1),
        "tp_to_predict": torch.arange(lookback, lookback + horizon, dtype=torch.float).unsqueeze(0).expand(B, -1),
        "target_idx": 0,
        "sample_keys": [f"{b['domain']}_w{b['start_idx']}" for b in batch],
    }

    if text_emb is not None:
        result["text_embeddings"] = text_emb
        result["text_tp"] = text_tp
        result["text_mask"] = text_mask

    return result


def _make_timemmd_timestep_collate_fn(d_txt: Optional[int] = None):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_timemmd_timestep(batch, d_txt=d_txt)
    return collate_fn


def collate_timemmd_patch(
    batch: List[Dict[str, Any]],
    d_txt: Optional[int] = None,
    n_patches: Optional[int] = None,
) -> Dict[str, Any]:
    B = len(batch)
    lookback = batch[0]["lookback"]
    horizon = batch[0]["horizon"]

    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)

    if n_patches is None:
        n_patches = batch[0].get("n_patches", 1)

    if d_txt is None:
        for b in batch:
            patch_embs = b.get("patch_embeddings")
            if patch_embs:
                first_emb = next(iter(patch_embs.values()))
                d_txt = first_emb.shape[-1]
                break

    if d_txt is not None:
        patch_text_emb = torch.zeros(B, n_patches, d_txt)
        patch_text_mask = torch.zeros(B, n_patches)

        for i, b in enumerate(batch):
            patch_embs = b.get("patch_embeddings")
            if patch_embs:
                for patch_idx, emb in patch_embs.items():
                    if 0 <= patch_idx < n_patches:
                        patch_text_emb[i, patch_idx, :] = emb
                        patch_text_mask[i, patch_idx] = 1.0
    else:
        patch_text_emb = None
        patch_text_mask = None

    result = {
        "observed_data": x.unsqueeze(-1),
        "observed_mask": torch.ones(B, lookback, 1),
        "observed_tp": torch.arange(lookback, dtype=torch.float).unsqueeze(0).expand(B, -1),
        "data_to_predict": y.unsqueeze(-1),
        "mask_predicted_data": torch.ones(B, horizon, 1),
        "tp_to_predict": torch.arange(lookback, lookback + horizon, dtype=torch.float).unsqueeze(0).expand(B, -1),
        "target_idx": 0,
        "sample_keys": [f"{b['domain']}_w{b['start_idx']}" for b in batch],
    }

    if patch_text_emb is not None:
        result["text_embeddings"] = patch_text_emb
        result["patch_text_mask"] = patch_text_mask

    return result


def _make_timemmd_patch_collate_fn(d_txt: Optional[int] = None, n_patches: Optional[int] = None):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_timemmd_patch(batch, d_txt=d_txt, n_patches=n_patches)
    return collate_fn


def _load_domain_date_info(data_path: str, domains: List[str]) -> Dict[str, pd.DataFrame]:
    import os
    result = {}
    numerical_root = os.path.join(data_path, "numerical")

    for domain in domains:
        num_path = os.path.join(numerical_root, domain, f"{domain}.csv")
        if not os.path.exists(num_path):
            continue

        df = pd.read_csv(num_path)
        df = TimeMMDDataset._ensure_date_columns(df, num_path, kind="numeric")
        if df is None:
            continue

        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df = df.sort_values("start_date").reset_index(drop=True)
        result[domain] = df

    return result


def build_time_mmd_dataloaders_timestep(
    data_path: str,
    lookback: int,
    horizon: int,
    batch_size: int,
    emb_cache_dir: Optional[Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    use_report: bool = True,
    use_search: bool = True,
    domains: Optional[List[str]] = None,
    scale: bool = True,
    domain_window_overrides: Optional[Dict[str, Tuple[int, int]]] = None,
    cache_dir: Optional[Path | str] = None,
    tsfm_model: str = "sundial",
    llm_model: str = "Qwen3_E4B",
    llm_layers: Optional[int] = None,
    input_token_len: int = SUNDIAL_INPUT_TOKEN_LEN,
    device: str = "cuda",
    use_patch_text_embeds: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cache_dir = cache_dir or (Path(data_path) / "cache_data")
    use_patch_embeddings = (
        tsfm_model.lower() == "sundial" and use_patch_text_embeds
    )

    base_ds = TimeMMDDataset(
        data_path=data_path,
        lookback=lookback,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_report=use_report,
        use_search=use_search,
        domains=domains,
        cache_dir=cache_dir,
        scale=scale,
        domain_window_overrides=domain_window_overrides,
    )

    d_txt: Optional[int] = None
    n_patches: Optional[int] = None
    timestep_emb_cache: Optional[TimestepEmbeddingCache] = None
    patch_emb_cache: Optional[PatchEmbeddingCache] = None

    if emb_cache_dir is not None:
        try:
            if use_patch_embeddings:
                if base_ds.domains:
                    eff_lb, eff_hz = base_ds._domain_window(base_ds.domains[0])
                else:
                    eff_lb, eff_hz = lookback, horizon

                patch_emb_cache = PatchEmbeddingCache(
                    cache_dir=Path(emb_cache_dir),
                    input_token_len=input_token_len,
                    llm_model=llm_model,
                    llm_layers=llm_layers,
                    data_root=Path(data_path),
                    device=device,
                    use_report=use_report,
                    use_search=use_search,
                    lookback=eff_lb,
                    horizon=eff_hz,
                    domains=base_ds.domains,
                )
                d_txt = patch_emb_cache.d_txt
                n_patches = patch_emb_cache.n_patches
            else:
                timestep_emb_cache = TimestepEmbeddingCache(
                    cache_dir=Path(emb_cache_dir),
                    llm_model=llm_model,
                    llm_layers=llm_layers,
                    data_root=Path(data_path),
                    device=device,
                    use_report=use_report,
                    use_search=use_search,
                    domains=base_ds.domains,
                )
                d_txt = timestep_emb_cache.d_txt
        except Exception as e:
            print(f"[build_time_mmd_dataloaders_timestep] Warning: {e}")
            print("  Proceeding without text embeddings.")

    domain_date_info: Dict[str, pd.DataFrame] = {}
    if not use_patch_embeddings:
        domain_date_info = _load_domain_date_info(data_path, base_ds.domains)

    domain_to_indices: Dict[str, List[int]] = {d: [] for d in base_ds.domains}
    for idx, w in enumerate(base_ds.windows):
        domain_to_indices[w.domain].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for domain in base_ds.domains:
        idxs = domain_to_indices.get(domain, [])
        n = base_ds.domain_lengths.get(domain, 0)
        if n == 0:
            continue
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        for i in idxs:
            split_idx = base_ds.windows[i].start_idx
            if split_idx < train_end:
                train_indices.append(i)
            elif split_idx < val_end:
                val_indices.append(i)
            else:
                test_indices.append(i)

    def filter_has_text(indices):
        return [i for i in indices if base_ds.windows[i].has_text]

    total_before = len(train_indices) + len(val_indices) + len(test_indices)
    train_indices = filter_has_text(train_indices)
    val_indices = filter_has_text(val_indices)
    test_indices = filter_has_text(test_indices)
    total_after = len(train_indices) + len(val_indices) + len(test_indices)

    print(
        f"[build_time_mmd_dataloaders_timestep] Filtered to text-available samples: "
        f"{total_after}/{total_before} ({100*total_after/total_before:.1f}%)"
    )
    print(f"[build_time_mmd_dataloaders_timestep] Split sizes: "
          f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    print(
        f"[build_time_mmd_dataloaders_timestep] Using {'patch' if use_patch_embeddings else 'timestep'} embeddings")

    if use_patch_embeddings:
        train_ds = TimeMMDPatchDataset(
            base_ds, train_indices, patch_emb_cache, input_token_len)
        val_ds = TimeMMDPatchDataset(
            base_ds, val_indices, patch_emb_cache, input_token_len)
        test_ds = TimeMMDPatchDataset(
            base_ds, test_indices, patch_emb_cache, input_token_len)
        collate_fn = _make_timemmd_patch_collate_fn(
            d_txt=d_txt, n_patches=n_patches)
    else:
        train_ds = TimeMMDTimestepDataset(
            base_ds, train_indices, timestep_emb_cache, domain_date_info)
        val_ds = TimeMMDTimestepDataset(
            base_ds, val_indices, timestep_emb_cache, domain_date_info)
        test_ds = TimeMMDTimestepDataset(
            base_ds, test_indices, timestep_emb_cache, domain_date_info)
        collate_fn = _make_timemmd_timestep_collate_fn(d_txt=d_txt)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    for loader in (train_loader, val_loader, test_loader):
        setattr(loader, "num_channels", base_ds.num_channels)
        setattr(loader, "d_txt", d_txt)
        if use_patch_embeddings:
            setattr(loader, "n_patches", n_patches)

    return train_loader, val_loader, test_loader
