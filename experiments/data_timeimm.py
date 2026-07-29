
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import SUNDIAL_INPUT_TOKEN_LEN
from experiments.utils import atomic_torch_save, is_embedding_file_complete




def _left_pad_sequence(seq_list, padding_value: float = 0.0, device: torch.device | None = None):
    if len(seq_list) == 0:
        return None
    max_len = max(t.size(0) for t in seq_list)
    first = seq_list[0]
    if device is None:
        device = first.device
    out_shape = (len(seq_list), max_len, *first.shape[1:])
    out = first.new_full(out_shape, padding_value, device=device)
    for i, t in enumerate(seq_list):
        L = t.size(0)
        out[i, max_len - L: max_len] = t.to(device)
    return out


def _maybe_get_step_units(args, unit_seconds_map: dict):
    if not getattr(args, "use_time_rescale", False):
        return None
    resample_interval_sec = getattr(args, "resample_interval_sec", None)
    if resample_interval_sec is None:
        raise ValueError(
            "use_time_rescale is True but resample_interval_sec is not set."
        )
    time_unit = getattr(args, "time_unit", None)
    if time_unit == "custom":
        unit_seconds = getattr(args, "unit_scale", None)
        if unit_seconds is None:
            raise ValueError("unit_scale must be set when time_unit='custom'.")
        unit_seconds = float(unit_seconds)
    else:
        unit_seconds = unit_seconds_map.get(time_unit)
    if unit_seconds is None:
        raise ValueError(f"Unknown time_unit '{time_unit}' for time rescale.")
    return resample_interval_sec / unit_seconds


def normalize_masked_tp(data, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8
    if (scale != 0.).all():
        data_norm = (data - att_min) / scale
    return data_norm




DOMAIN_META: Dict[str, Dict[str, str]] = {
    "GDELT": {
        "task": "Forecast geopolitical conflict and cooperation levels.",
        "target_description": "The target is GoldsteinScale, a score from -10 to +10 measuring the theoretical impact of a geopolitical event on country stability, where negative values indicate conflict (protests, attacks) and positive values indicate cooperation (diplomacy, aid).",
    },
    "RepoHealth": {
        "task": "Forecast software repository maintenance pressure.",
        "target_description": "The target is the count of unresolved issues in a repository at each observation time, capturing the current backlog of reported bugs, feature requests, and other tasks that reflect maintenance pressure.",
    },
    "MIMIC": {
        "task": "Forecast patient IV dextrose administration volume.",
        "target_description": "The target indicates the administration of an intravenous solution containing 5g dextrose per 100mL water, reflecting fluid management and metabolic or glycemic needs during treatment.",
    },
    "FNSPID": {
        "task": "Forecast stock performance.",
        "target_description": "The target is the adjusted closing price, reflecting the true value of the stock by accounting for dilutive effects, capturing the underlying performance of the stock.",
    },
    "ClusterTrace": {
        "task": "Forecast GPU cluster workload.",
        "target_description": "The target is the number of concurrent instances actively running on a GPU machine at each observation time, indicating the current level of parallel workload on that node.",
    },
    "StudentLife": {
        "task": "Forecast student sleep duration.",
        "target_description": "The target captures how long the student actually sleeps each night, reflecting changes in rest, recovery, and potential impacts on mood, stress, and performance.",
    },
    "ILINet": {
        "task": "Forecast influenza-like illness prevalence.",
        "target_description": "The target is the percentage of outpatient visits for influenza-like illness, aggregated from state-level data weighted by population, serving as an indicator of influenza intensity over time and space.",
    },
    "CESNET": {
        "task": "Forecast network traffic volume and load.",
        "target_description": "The target captures network traffic flow measurements, reflecting data transmission patterns and network utilization over time.",
    },
    "EPA-Air": {
        "task": "Forecast ambient air temperature.",
        "target_description": "The target is the ambient outdoor air temperature recorded hourly in degrees Fahrenheit, reflecting local atmospheric conditions that influence air quality and pollutant concentrations.",
    },
}




def build_timeimm_text(
    data_name: str,
    raw_text: str,
    add_header: bool = True,
    timestamp: str | None = None,
) -> str:
    meta = DOMAIN_META.get(data_name, {})

    if not add_header:
        if timestamp is not None:
            return f"Time: {timestamp}\n\n{raw_text}"
        return raw_text

    task = meta.get("task", f"Forecast {data_name}.")
    target_description = meta.get("target_description", "")

    parts = [f"Task: {task}"]
    if target_description:
        parts.append(f"Target: {target_description}")

    if timestamp is not None:
        parts.append(f"Time: {timestamp}")

    parts.append("")
    parts.append(raw_text)

    return "\n".join(parts)




def build_patch_header(data_name: str) -> str:
    meta = DOMAIN_META.get(data_name, {})
    target_description = meta.get("target_description", "")

    task = meta.get("task", f"Forecast {data_name}.")
    header_parts = [
        "[FORECASTING TASK]",
        f"Task: {task}",
        f"Dataset: {data_name}",
    ]
    if target_description:
        header_parts.extend(["", "[TARGET]", target_description])

    return "\n".join(header_parts)


def concatenate_patch_texts(
    patch_texts: Dict[int, List[Tuple[int, str, str]]],
    data_name: str,
    add_header: bool = True,
) -> Dict[int, str]:
    shared_header = build_patch_header(data_name) if add_header else ""
    keep_timestamps = True

    result = {}
    for patch_idx, texts in patch_texts.items():
        formatted_entries = []
        for entry_num, (grid_idx, timestamp_str, text_str) in enumerate(texts, start=1):
            if keep_timestamps:
                entry_str = f"[TEXT {entry_num}] {timestamp_str}\n{text_str}"
            else:
                entry_str = f"[TEXT {entry_num}]\n{text_str}"
            formatted_entries.append(entry_str)

        text_block = "\n\n".join(formatted_entries)

        if shared_header:
            result[patch_idx] = shared_header + "\n\n" + text_block
        else:
            result[patch_idx] = text_block

    return result



_LAZY_LLM_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _get_lazy_llm(
    llm_model_fusion: str,
    llm_layers_fusion: Optional[int],
    device: str,
) -> Tuple[Any, Any]:
    cache_key = f"{llm_model_fusion}_{llm_layers_fusion}_{device}"
    if cache_key not in _LAZY_LLM_CACHE:
        from experiments.utils import load_llm
        print(f"[TimeIMM] Lazy loading LLM: {llm_model_fusion} (layers={llm_layers_fusion})")
        tokenizer, llm_model = load_llm(
            llm_model_fusion,
            llm_layers_fusion,
            device,
            use_device_map=False,
        )
        _LAZY_LLM_CACHE[cache_key] = (tokenizer, llm_model)
    return _LAZY_LLM_CACHE[cache_key]




class TimeIMMTextEmbeddingCache:

    def __init__(
        self,
        data_root: Path,
        dataset: str,
        llm_model: str = "Qwen3_E4B",
        llm_layers: Optional[int] = None,
        device: str = "cuda",
        add_header: bool = True,
        encode_batch_size: int = 8,
        max_length: int = 2048,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.device = device
        self.add_header = add_header
        self.encode_batch_size = encode_batch_size
        self.max_length = max_length
        self.dtype = dtype

        self._tokenizer = None
        self._llm_model = None

        self.d_txt: Optional[int] = None
        self.data: Dict[str, Dict[str, Any]] = {}

        self._load_or_compute_embeddings()

    def _get_embedding_filename(self) -> str:
        layers_str = self.llm_layers if self.llm_layers is not None else "full"
        return (
            f"{self.dataset}_text_embeddings"
            f"_model={self.llm_model}"
            f"_layers={layers_str}"
            f"_maxlen={self.max_length}.pt"
        )

    def _load_or_compute_embeddings(self) -> None:
        emb_dir = self.data_root / "text_embeddings"
        emb_path = emb_dir / self._get_embedding_filename()

        if is_embedding_file_complete(emb_path):
            payload = torch.load(emb_path, map_location="cpu", weights_only=False)
            self.data = payload.get("embeddings_by_record", {})
            if self.data:
                first_record = next(iter(self.data.values()))
                emb = first_record.get("embeddings")
                if emb is not None and emb.numel() > 0:
                    self.d_txt = emb.shape[-1]
            print(f"[TimeIMMTextEmbeddingCache] Loaded {self.dataset}: {len(self.data)} records, d_txt={self.d_txt}")
            return

        self._ensure_embeddings(emb_path)

    def _ensure_embeddings(self, emb_path: Path) -> None:
        from experiments.utils import embed_notes

        proc_dir = self.data_root / self.dataset / "processed"
        if not proc_dir.exists():
            print(f"[TimeIMMTextEmbeddingCache] No processed directory for {self.dataset}")
            return

        rec_ids = sorted(d for d in os.listdir(proc_dir) if (proc_dir / d).is_dir())
        if not rec_ids:
            print(f"[TimeIMMTextEmbeddingCache] No records found for {self.dataset}")
            return

        print(f"[TimeIMMTextEmbeddingCache] Computing embeddings for {self.dataset} ({len(rec_ids)} records)...")

        tokenizer, llm_model = _get_lazy_llm(self.llm_model, self.llm_layers, self.device)

        embeddings_by_record = {}
        total_texts = 0

        for rec_id in tqdm(rec_ids, desc=f"{self.dataset} records"):
            rec_path = proc_dir / rec_id
            text_path = rec_path / "text.csv"
            if not text_path.exists():
                continue

            tdf = pd.read_csv(text_path, parse_dates=["date_time"]).sort_values("date_time")
            cols = [c for c in tdf.columns if c not in ("date_time", "record_id")]
            if not cols:
                continue

            raw_notes = []
            timestamps = []

            for _, row in tdf.iterrows():
                txt = row[cols[0]]
                if pd.notna(txt):
                    raw_notes.append(str(txt))
                    timestamps.append(row["date_time"].strftime("%Y-%m-%d %H:%M:%S"))

            if not raw_notes:
                continue

            notes = [
                build_timeimm_text(self.dataset, note, add_header=self.add_header, timestamp=ts)
                for note, ts in zip(raw_notes, timestamps)
            ]

            for idx, note in enumerate(notes):
                token_count = len(tokenizer.encode(note, add_special_tokens=True))
                if token_count > self.max_length:
                    tqdm.write(f"  [TRUNCATED] {rec_id} idx={idx}, {token_count} -> {self.max_length} tokens")

            record_embeddings = []
            for i in range(0, len(notes), self.encode_batch_size):
                batch_notes = notes[i:i + self.encode_batch_size]
                notes_batch = [[n] for n in batch_notes]
                emb, _ = embed_notes(
                    notes_batch, tokenizer, llm_model, max_length=self.max_length,
                )
                for j in range(emb.shape[0]):
                    record_embeddings.append(emb[j].squeeze(0).cpu())
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            if record_embeddings:
                emb_tensor = torch.stack(record_embeddings, dim=0)
                embeddings_by_record[rec_id] = {
                    "embeddings": emb_tensor,
                    "timestamps": timestamps,
                }
                total_texts += len(timestamps)

                if self.d_txt is None:
                    self.d_txt = emb_tensor.shape[-1]

        if not embeddings_by_record:
            print(f"[TimeIMMTextEmbeddingCache] No text data found for {self.dataset}")
            return

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "embeddings_by_record": embeddings_by_record,
            "settings": {
                "llm_model": self.llm_model,
                "llm_layers": self.llm_layers,
                "max_length": self.max_length,
                "add_header": self.add_header,
            },
        }
        atomic_torch_save(payload, emb_path)
        print(f"[TimeIMMTextEmbeddingCache] Saved {len(embeddings_by_record)} records, {total_texts} texts to {emb_path}")

        self.data = embeddings_by_record

    def get_texts_for_record(
        self,
        record_id: str,
        base_time: pd.Timestamp,
        sec_per_unit: float,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tuple[float, torch.Tensor]]:
        if record_id not in self.data:
            return []

        record_data = self.data[record_id]
        embeddings = record_data["embeddings"]
        timestamps = record_data["timestamps"]

        result = []
        for i, ts_str in enumerate(timestamps):
            ts = pd.to_datetime(ts_str)
            rel_time = (ts - base_time).total_seconds() / sec_per_unit
            result.append((rel_time, embeddings[i].to(device)))

        return result


_TIMEIMM_TEXT_EMB_CACHE: Dict[str, TimeIMMTextEmbeddingCache] = {}


def get_timeimm_text_embedding_cache(
    data_root: Path,
    dataset: str,
    llm_model: str,
    llm_layers: Optional[int],
    device: str,
    max_length: int,
    add_header: bool = True,
) -> TimeIMMTextEmbeddingCache:
    cache_key = f"{data_root}_{dataset}_{llm_model}_{llm_layers}_{max_length}"
    if cache_key not in _TIMEIMM_TEXT_EMB_CACHE:
        _TIMEIMM_TEXT_EMB_CACHE[cache_key] = TimeIMMTextEmbeddingCache(
            data_root=data_root,
            dataset=dataset,
            llm_model=llm_model,
            llm_layers=llm_layers,
            device=device,
            max_length=max_length,
            add_header=add_header,
        )
    return _TIMEIMM_TEXT_EMB_CACHE[cache_key]


def load_texts_for_record(
    rec_path: Path,
    base_time,
    sec_per_unit: float,
    use_text_embeddings: bool = False,
    llm_model_fusion: Optional[str] = None,
    llm_layers_fusion: Optional[int] = None,
    max_length: int = 1024,
    device: torch.device = torch.device("cpu"),
    add_header: bool = True,
    data_name: Optional[str] = None,
) -> List[Tuple[float, Any]]:
    texts = []

    if use_text_embeddings and llm_model_fusion:
        record_id = rec_path.name
        dataset = data_name or rec_path.parent.parent.name
        data_root = rec_path.parent.parent.parent

        cache = get_timeimm_text_embedding_cache(
            data_root=data_root,
            dataset=dataset,
            llm_model=llm_model_fusion,
            llm_layers=llm_layers_fusion,
            device=str(device),
            max_length=max_length,
            add_header=add_header,
        )

        return cache.get_texts_for_record(record_id, base_time, sec_per_unit, device)

    text_path = rec_path / "text.csv"
    if text_path.exists():
        tdf = pd.read_csv(text_path, parse_dates=["date_time"]).sort_values("date_time")
        cols = [c for c in tdf.columns if c not in ("date_time", "record_id")]
        if cols:
            for _, row in tdf.iterrows():
                txt = row[cols[0]]
                if pd.notna(txt):
                    t_rel = (row["date_time"] - base_time).total_seconds() / sec_per_unit
                    texts.append((t_rel, str(txt)))

    return texts


def load_patch_text_embeddings_for_dataset(
    data_path: Path,
    dataset: str,
    llm_model_fusion: str,
    llm_layers_fusion: Optional[int] = None,
    input_token_len: int = SUNDIAL_INPUT_TOKEN_LEN,
    device: torch.device = torch.device("cpu"),
    max_length: int = 5120,
    add_header: bool = True,
) -> Optional[Dict[str, Any]]:
    emb_dir = Path(data_path) / "text_embeddings"
    fname = (
        f"{dataset}_patch_text_embeddings"
        f"_model={llm_model_fusion}"
        f"_layers={llm_layers_fusion or 'full'}"
        f"_patchlen={input_token_len}.pt"
    )
    path = emb_dir / fname

    if is_embedding_file_complete(path):
        data = torch.load(path, map_location=device, weights_only=False)
        if "chunk_embeddings" in data:
            for chunk_id, chunk_data in data["chunk_embeddings"].items():
                if "patch_embeddings" in chunk_data:
                    for patch_idx, emb in chunk_data["patch_embeddings"].items():
                        if torch.is_tensor(emb):
                            chunk_data["patch_embeddings"][patch_idx] = emb.to(device)
        return data

    return _compute_and_save_patch_embeddings(
        data_path=data_path,
        dataset=dataset,
        llm_model_fusion=llm_model_fusion,
        llm_layers_fusion=llm_layers_fusion,
        input_token_len=input_token_len,
        device=device,
        max_length=max_length,
        add_header=add_header,
        out_path=path,
    )


def _load_texts_with_timestamps(
    rec_path: Path,
    base_time: pd.Timestamp,
    sec_per_unit: float,
) -> List[Tuple[float, str, str]]:
    texts = []
    text_path = rec_path / "text.csv"

    if text_path.exists():
        tdf = pd.read_csv(text_path, parse_dates=["date_time"]).sort_values("date_time")
        cols = [c for c in tdf.columns if c not in ("date_time", "record_id")]
        if cols:
            for _, row in tdf.iterrows():
                txt = row[cols[0]]
                if pd.notna(txt):
                    t_rel = (row["date_time"] - base_time).total_seconds() / sec_per_unit
                    timestamp_str = row["date_time"].strftime("%Y-%m-%d %H:%M:%S")
                    texts.append((t_rel, timestamp_str, str(txt)))

    return texts


def _snap_texts_to_grid_with_timestamps(
    texts: List[Tuple[float, str, str]],
    step: float,
    hist_steps: int,
    lookback: float,
) -> List[Tuple[int, str, str]]:
    snapped = []

    for t_rel, timestamp_str, text_str in texts:
        if 0 <= t_rel < lookback:
            grid_idx = min(int(math.ceil(t_rel / step)), hist_steps - 1)
            snapped.append((grid_idx, timestamp_str, text_str))

    return snapped


def _assign_texts_to_patches_with_timestamps(
    snapped_texts: List[Tuple[int, str, str]],
    input_token_len: int,
) -> Dict[int, List[Tuple[int, str, str]]]:
    patches: Dict[int, List[Tuple[int, str, str]]] = defaultdict(list)

    for grid_idx, timestamp_str, text_str in snapped_texts:
        patch_idx = grid_idx // input_token_len
        patches[patch_idx].append((grid_idx, timestamp_str, text_str))

    for patch_idx in patches:
        patches[patch_idx].sort(key=lambda x: x[0])

    return dict(patches)


def _load_record_timeseries(
    rec_path: Path,
    sec_per_unit: float,
    feature_names: Optional[List[str]] = None,
    target_column: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.Timestamp, List[str], int, str]:
    ts_path = rec_path / "time_series.csv"
    df = pd.read_csv(ts_path)
    df["_ts_raw"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("_ts_raw")

    feat_cols = [c for c in df.columns if c not in ("date_time", "record_id", "_ts_raw")]

    if feature_names is None:
        feature_names = feat_cols

    target_idx = (
        feat_cols.index(target_column)
        if target_column and target_column in feat_cols
        else 0
    )
    target_name = feat_cols[target_idx]

    df = pd.concat([df[feature_names], df[["_ts_raw"]]], axis=1)

    for col in feature_names:
        std = df[col].std()
        mean = df[col].mean()
        df[col] = (df[col] - mean) / std if std else df[col] - mean

    base_time = df["_ts_raw"].min()
    secs = (df["_ts_raw"] - base_time).dt.total_seconds()
    tt = torch.tensor((secs / sec_per_unit).values, dtype=torch.float32)
    vals_np = df[feature_names].values.astype("float32")
    vals = torch.nan_to_num(torch.tensor(vals_np))
    mask = torch.tensor((~pd.isna(vals_np)).astype("float32"))

    return tt, vals, mask, base_time, feature_names, target_idx, target_name


def _compute_and_save_patch_embeddings(
    data_path: Path,
    dataset: str,
    llm_model_fusion: str,
    llm_layers_fusion: Optional[int],
    input_token_len: int,
    device: torch.device,
    max_length: int,
    add_header: bool,
    out_path: Path,
    encode_batch_size: int = 4,
) -> Optional[Dict[str, Any]]:
    from config import CONFIG
    from experiments.utils import embed_notes

    proc_dir = Path(data_path) / dataset / "processed"
    if not proc_dir.exists():
        print(f"[TimeIMM] No processed directory for {dataset}, skipping patch embeddings")
        return None

    time_imm_cfg = CONFIG.timeimm
    settings = time_imm_cfg.dataset_presets.get(dataset)
    if not settings:
        print(f"[TimeIMM] No preset found for {dataset}, skipping patch embeddings")
        return None

    lookback = settings["lookback"]
    horizon = settings["horizon"]
    stride = settings["stride"]
    time_unit = settings["time_unit"]
    resample_interval_sec = settings["resample_interval_sec"]
    target_column = settings.get("OT")

    sec_per_unit = SEC_PER_UNIT[time_unit]
    step = resample_interval_sec / sec_per_unit
    hist_steps, n_patches = compute_patch_params(lookback, step, input_token_len)

    print(f"[TimeIMM] Computing patch embeddings for {dataset}")
    print(f"[TimeIMM] Settings: lookback={lookback}, horizon={horizon}, stride={stride}")
    print(f"[TimeIMM] Grid: step={step:.4f}, hist_steps={hist_steps}, n_patches={n_patches}")

    rec_ids = sorted(d for d in os.listdir(proc_dir) if (proc_dir / d).is_dir())
    if not rec_ids:
        print(f"[TimeIMM] No records found for {dataset}")
        return None

    device_str = str(device) if isinstance(device, torch.device) else device
    tokenizer, llm_model = _get_lazy_llm(llm_model_fusion, llm_layers_fusion, device_str)

    all_chunk_embeddings = {}
    total_chunks = 0
    total_patches_with_text = 0
    feature_names = None

    for rec in tqdm(rec_ids, desc=f"{dataset} records"):
        rec_path = proc_dir / rec
        ts_path = rec_path / "time_series.csv"
        if not ts_path.exists():
            continue

        tt, vals, mask, base_time, feature_names, _, _ = _load_record_timeseries(
            rec_path, sec_per_unit, feature_names, target_column
        )

        texts_with_ts = _load_texts_with_timestamps(rec_path, base_time, sec_per_unit)
        if not texts_with_ts:
            continue

        chunks = create_chunks_from_record(
            rec_id=rec,
            tt=tt,
            vals=vals,
            mask=mask,
            record_texts=[],
            lookback=lookback,
            horizon=horizon,
            stride=stride,
        )

        for chunk_idx, (chunk_id, sub_tt, sub_vals, sub_mask, _, _) in enumerate(chunks):
            chunk_start = chunk_idx * stride

            chunk_texts = [
                (t_rel - chunk_start, timestamp_str, text_str)
                for t_rel, timestamp_str, text_str in texts_with_ts
                if chunk_start <= t_rel < chunk_start + lookback
            ]

            if not chunk_texts:
                continue

            snapped = _snap_texts_to_grid_with_timestamps(chunk_texts, step, hist_steps, lookback)
            if not snapped:
                continue

            patch_texts = _assign_texts_to_patches_with_timestamps(snapped, input_token_len)
            if not patch_texts:
                continue

            concat_texts = concatenate_patch_texts(patch_texts, dataset)

            items = [(idx, text) for idx, text in concat_texts.items() if text.strip()]
            if not items:
                continue

            indices, texts = zip(*items)

            for i, text in enumerate(texts):
                token_count = len(tokenizer.encode(text, add_special_tokens=True))
                if token_count > max_length:
                    tqdm.write(
                        f"  [TRUNCATED] {dataset} {chunk_id} patch={indices[i]}, "
                        f"{token_count} -> {max_length} tokens"
                    )

            patch_embeddings = {}
            for i in range(0, len(texts), encode_batch_size):
                batch_indices = indices[i:i + encode_batch_size]
                batch_texts = texts[i:i + encode_batch_size]

                notes_batch = [[t] for t in batch_texts]
                emb, _ = embed_notes(
                    notes_batch, tokenizer, llm_model, max_length=max_length,
                )

                for j, idx in enumerate(batch_indices):
                    patch_embeddings[idx] = emb[j].squeeze(0).cpu()

                if device_str.startswith("cuda"):
                    torch.cuda.empty_cache()

            if patch_embeddings:
                all_chunk_embeddings[chunk_id] = {
                    "patch_embeddings": patch_embeddings,
                    "n_patches": n_patches,
                    "hist_steps": hist_steps,
                }
                total_chunks += 1
                total_patches_with_text += len(patch_embeddings)

    if not all_chunk_embeddings:
        print(f"[TimeIMM] No chunks with text embeddings generated for {dataset}")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "chunk_embeddings": all_chunk_embeddings,
        "settings": {
            "lookback": lookback,
            "horizon": horizon,
            "stride": stride,
            "time_unit": time_unit,
            "resample_interval_sec": resample_interval_sec,
            "input_token_len": input_token_len,
            "hist_steps": hist_steps,
            "n_patches": n_patches,
            "step": step,
        },
        "llm_model": llm_model_fusion,
        "llm_layers": llm_layers_fusion,
        "max_length": max_length,
    }
    atomic_torch_save(payload, out_path)
    print(f"[TimeIMM] Saved {total_chunks} chunks, {total_patches_with_text} patches to {out_path}")

    for chunk_id, chunk_data in all_chunk_embeddings.items():
        if "patch_embeddings" in chunk_data:
            for patch_idx, emb in chunk_data["patch_embeddings"].items():
                if torch.is_tensor(emb):
                    chunk_data["patch_embeddings"][patch_idx] = emb.to(device)

    return payload


def get_patch_embeddings_for_chunk(
    patch_emb_data: Optional[Dict[str, Any]],
    chunk_id: str,
    n_patches: int,
    d_txt: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.zeros(n_patches, d_txt, device=device)
    mask = torch.zeros(n_patches, device=device)

    if patch_emb_data is None:
        return result, mask

    chunk_emb = patch_emb_data.get("chunk_embeddings", {}).get(chunk_id)
    if chunk_emb is None:
        return result, mask

    patch_embeddings = chunk_emb.get("patch_embeddings", {})
    for patch_idx, emb in patch_embeddings.items():
        idx = int(patch_idx)
        if 0 <= idx < n_patches:
            result[idx] = emb.to(device)
            mask[idx] = 1.0

    return result, mask


def create_chunks_from_record(
    rec_id: str,
    tt: torch.Tensor,
    vals: torch.Tensor,
    mask: torch.Tensor,
    record_texts: List[Tuple[float, Any]],
    lookback: float,
    horizon: float,
    stride: float,
    anchor: Optional[float] = None,
) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[float, Any]], bool]]:
    chunks = []
    total_window = lookback + horizon
    t_max = tt.max().item()
    st = tt.min().item()
    if anchor is not None:
        st += (anchor - lookback - st) % stride
    cnt = 0

    while st + total_window <= t_max:
        idx = ((tt >= st) & (tt < st + total_window)).nonzero(as_tuple=False).squeeze(1)

        if idx.numel() >= 2:
            sub_tt = tt[idx] - st
            sub_vals = vals[idx]
            sub_mask = mask[idx]

            hist_mask = sub_mask[sub_tt < lookback]
            pred_mask = sub_mask[sub_tt >= lookback]

            if hist_mask.sum() == 0 or pred_mask.sum() == 0:
                st += stride
                continue

            hist_end = st + lookback
            selected = [
                (t - st, payload)
                for (t, payload) in record_texts
                if st <= t < hist_end
            ]

            has_text = len(selected) > 0
            chunk_id = f"{rec_id}_chunk{cnt}"
            chunks.append((chunk_id, sub_tt, sub_vals, sub_mask, selected, has_text))
            cnt += 1

        st += stride

    return chunks


def snap_texts_to_grid(
    texts: List[Tuple[float, Any]],
    step: float,
    hist_steps: int,
    lookback: float,
) -> List[Tuple[int, float, Any]]:
    snapped = []

    for t_rel, payload in texts:
        if 0 <= t_rel < lookback:
            grid_idx = min(int(math.ceil(t_rel / step)), hist_steps - 1)
            snapped_time = grid_idx * step
            snapped.append((grid_idx, snapped_time, payload))

    return snapped


def assign_texts_to_patches(
    snapped_texts: List[Tuple[int, float, Any]],
    input_token_len: int,
) -> Dict[int, List[Tuple[int, float, Any]]]:
    patches: Dict[int, List[Tuple[int, float, Any]]] = defaultdict(list)

    for grid_idx, snapped_time, payload in snapped_texts:
        patch_idx = grid_idx // input_token_len
        patches[patch_idx].append((grid_idx, snapped_time, payload))

    for patch_idx in patches:
        patches[patch_idx].sort(key=lambda x: x[0])

    return dict(patches)


def compute_patch_params(
    lookback: float,
    step: float,
    input_token_len: int,
) -> Tuple[int, int]:
    hist_steps = max(1, int(np.ceil(lookback / step)))
    n_patches = int(np.ceil(hist_steps / input_token_len))
    return hist_steps, n_patches




def _process_texts_for_collate(
    batch: List[Tuple],
    history: float,
    step_units: Optional[float] = None,
    chunk_time_max: Optional[torch.Tensor] = None,
    left_pad: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    text_tp_list = []
    text_emb_list = []
    d_txt = None

    for sample in batch:
        if len(sample) == 5:
            _, tt, _, _, texts = sample
        else:
            texts = []

        sample_tps = []
        sample_embs = []

        for t_rel, payload in texts:
            if 0 <= t_rel < history:
                sample_tps.append(t_rel)
                if torch.is_tensor(payload):
                    sample_embs.append(payload)
                    if d_txt is None:
                        d_txt = payload.shape[-1]

        if sample_tps:
            tp_tensor = torch.tensor(sample_tps, dtype=torch.float32)
        else:
            tp_tensor = torch.zeros(0, dtype=torch.float32)

        if sample_embs and d_txt is not None:
            emb_tensor = torch.stack(sample_embs, dim=0)
        elif d_txt is not None:
            emb_tensor = torch.zeros(0, d_txt, dtype=torch.float32)
        else:
            emb_tensor = None

        text_tp_list.append(tp_tensor)
        text_emb_list.append(emb_tensor)

    has_embeddings = d_txt is not None and any(
        e is not None and e.numel() > 0 for e in text_emb_list
    )

    if not has_embeddings:
        return {
            "text_embeddings": None,
            "text_tp": None,
            "text_mask": None,
            "d_txt": None,
        }

    if left_pad:
        text_tp = _left_pad_sequence(
            text_tp_list, padding_value=0.0, device=device)
        emb_list_safe = [
            e if e is not None and e.numel() > 0 else torch.zeros(0, d_txt)
            for e in text_emb_list
        ]
        text_emb = _left_pad_sequence(
            emb_list_safe, padding_value=0.0, device=device)
    else:
        text_tp = pad_sequence(
            text_tp_list, batch_first=True, padding_value=0.0)
        emb_list_safe = [
            e if e is not None and e.numel() > 0 else torch.zeros(0, d_txt)
            for e in text_emb_list
        ]
        text_emb = pad_sequence(
            emb_list_safe, batch_first=True, padding_value=0.0)

    B = len(batch)
    max_len = text_tp.size(1) if text_tp is not None else 0
    text_mask = torch.zeros(B, max_len, dtype=torch.float32)
    for i, tp_tensor in enumerate(text_tp_list):
        n_texts = tp_tensor.size(0)
        if left_pad:
            text_mask[i, max_len - n_texts:] = 1.0
        else:
            text_mask[i, :n_texts] = 1.0

    if text_tp is not None:
        if step_units is not None:
            text_tp = text_tp / step_units

    if device is not None:
        if text_tp is not None:
            text_tp = text_tp.to(device)
        if text_emb is not None:
            text_emb = text_emb.to(device)
        text_mask = text_mask.to(device)

    return {
        "text_embeddings": text_emb,
        "text_tp": text_tp,
        "text_mask": text_mask,
        "d_txt": d_txt,
    }


def collate_varlen_rightpad(batch, args, time_max=None, unit_seconds_map=None):
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    unit_seconds_map = unit_seconds_map or {}
    step_units = _maybe_get_step_units(args, unit_seconds_map)
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )

    for sample in batch:
        if len(sample) == 5:
            _, tt, vals, mask, _ = sample
        else:
            _, tt, vals, mask = sample

        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])

    observed_tp = pad_sequence(
        observed_tp, batch_first=True, padding_value=0.0)
    observed_data = pad_sequence(
        observed_data, batch_first=True, padding_value=0.0)
    observed_mask = pad_sequence(
        observed_mask, batch_first=True, padding_value=0.0)
    predicted_tp = pad_sequence(
        predicted_tp, batch_first=True, padding_value=0.0)
    predicted_data = pad_sequence(
        predicted_data, batch_first=True, padding_value=0.0)
    predicted_mask = pad_sequence(
        predicted_mask, batch_first=True, padding_value=0.0)

    if step_units is not None:
        observed_tp = observed_tp / step_units
        predicted_tp = predicted_tp / step_units

    text_info = _process_texts_for_collate(
        batch,
        history=args.history,
        step_units=step_units,
        chunk_time_max=chunk_time_max,
        left_pad=False,
        device=args.device,
    )

    result = {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
        "sample_keys": [str(sample[0]) for sample in batch],
    }

    if text_info["text_embeddings"] is not None:
        result["text_embeddings"] = text_info["text_embeddings"]
        result["text_tp"] = text_info["text_tp"]
        result["text_mask"] = text_info["text_mask"]

    return result


def collate_varlen_leftpad(batch, args, time_max=None, unit_seconds_map=None):
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    unit_seconds_map = unit_seconds_map or {}
    step_units = _maybe_get_step_units(args, unit_seconds_map)
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )

    for sample in batch:
        if len(sample) == 5:
            _, tt, vals, mask, _ = sample
        else:
            _, tt, vals, mask = sample

        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])

    observed_tp = _left_pad_sequence(
        observed_tp, padding_value=0.0, device=args.device)
    observed_data = _left_pad_sequence(
        observed_data, padding_value=0.0, device=args.device)
    observed_mask = _left_pad_sequence(
        observed_mask, padding_value=0.0, device=args.device)
    predicted_tp = _left_pad_sequence(
        predicted_tp, padding_value=0.0, device=args.device)
    predicted_data = _left_pad_sequence(
        predicted_data, padding_value=0.0, device=args.device)
    predicted_mask = _left_pad_sequence(
        predicted_mask, padding_value=0.0, device=args.device)

    if step_units is not None:
        observed_tp = observed_tp / step_units
        predicted_tp = predicted_tp / step_units
    else:
        observed_tp = normalize_masked_tp(
            observed_tp, att_min=0.0, att_max=chunk_time_max
        )
        predicted_tp = normalize_masked_tp(
            predicted_tp, att_min=0.0, att_max=chunk_time_max
        )

    text_info = _process_texts_for_collate(
        batch,
        history=args.history,
        step_units=step_units,
        chunk_time_max=chunk_time_max,
        left_pad=True,
        device=args.device,
    )

    result = {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
        "sample_keys": [str(sample[0]) for sample in batch],
    }

    if text_info["text_embeddings"] is not None:
        result["text_embeddings"] = text_info["text_embeddings"]
        result["text_tp"] = text_info["text_tp"]
        result["text_mask"] = text_info["text_mask"]

    return result


def collate_varlen_autoreg(batch, args, time_max=None, unit_seconds_map=None):
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    unit_seconds_map = unit_seconds_map or {}
    step_units = _maybe_get_step_units(args, unit_seconds_map)
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )

    for sample in batch:
        if len(sample) == 5:
            _, tt, vals, mask, _ = sample
        else:
            _, tt, vals, mask = sample

        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])

    observed_tp = _left_pad_sequence(
        observed_tp, padding_value=0.0, device=args.device)
    observed_data = _left_pad_sequence(
        observed_data, padding_value=0.0, device=args.device)
    observed_mask = _left_pad_sequence(
        observed_mask, padding_value=0.0, device=args.device)

    predicted_tp = pad_sequence(
        predicted_tp, batch_first=True, padding_value=0.0)
    predicted_data = pad_sequence(
        predicted_data, batch_first=True, padding_value=0.0)
    predicted_mask = pad_sequence(
        predicted_mask, batch_first=True, padding_value=0.0)

    if step_units is not None:
        observed_tp = observed_tp / step_units
        predicted_tp = predicted_tp / step_units

    text_info = _process_texts_for_collate(
        batch,
        history=args.history,
        step_units=step_units,
        chunk_time_max=chunk_time_max,
        left_pad=True,
        device=args.device,
    )

    result = {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
        "sample_keys": [str(sample[0]) for sample in batch],
    }

    if text_info["text_embeddings"] is not None:
        result["text_embeddings"] = text_info["text_embeddings"]
        result["text_tp"] = text_info["text_tp"]
        result["text_mask"] = text_info["text_mask"]

    return result



SEC_PER_UNIT = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 604800.0,
}



@dataclass
class TimeIMMConfig:
    data_path: str
    dataset: str

    lookback: int
    horizon: int
    stride: int

    time_unit: str
    resample_interval_sec: int

    chunk_anchor_date: Optional[str] = None

    enable_text: bool = False
    use_text_embeddings: bool = False
    llm_model_fusion: Optional[str] = None
    llm_layers_fusion: Optional[int] = None
    max_length: int = 1024

    split_method: str = "sample"
    train_ratio: float = 0.6
    val_ratio: float = 0.2

    target_column: Optional[str] = None
    use_covariates: bool = False
    use_time_rescale: bool = True

    cache_dir: Optional[Path] = None

    @property
    def step_units(self) -> float:
        return self.resample_interval_sec / SEC_PER_UNIT[self.time_unit]

    @property
    def dataset_path(self) -> Path:
        return Path(self.data_path) / self.dataset



class ChunkedTimeSeriesDataset(Dataset):

    def __init__(self, config: TimeIMMConfig, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.config = config
        self.device = device
        self.feature_names: Optional[List[str]] = None
        self.num_channels: Optional[int] = None
        self.target_idx: Optional[int] = None
        self.target_name: Optional[str] = None
        self.record_stats: Dict[str, Dict[str, torch.Tensor]] = {}

        self._sec_per_unit = SEC_PER_UNIT[config.time_unit]

        cache_dir = config.cache_dir or (
            config.dataset_path.parent / "cache_data")
        self._cache_dir = Path(cache_dir)

        self.chunks = self._load_or_build_chunks()

    def _load_or_build_chunks(self) -> List[Tuple]:
        cache_path = self._cache_path()

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                if payload.get("params") == self._cache_params():
                    self.feature_names = payload.get("feature_names")
                    self.num_channels = payload.get("num_channels")
                    self.target_idx = payload.get("target_idx")
                    self.target_name = payload.get("target_name")
                    cached_stats = payload.get("record_stats")
                    if cached_stats:
                        self.record_stats = self._coerce_record_stats(cached_stats)
                    else:
                        self.record_stats = self._load_record_stats_from_raw()
                    print(
                        f"[ChunkedTimeSeriesDataset] Loaded {len(payload['chunks'])} chunks from cache")
                    return payload["chunks"]
            except Exception as e:
                print(
                    f"[ChunkedTimeSeriesDataset] Cache load failed ({e}); rebuilding")

        chunks = self._build_chunks()

        self._save_cache(chunks, cache_path)
        return chunks

    def _build_chunks(self) -> List[Tuple]:
        config = self.config
        proc_dir = config.dataset_path / "processed"
        rec_ids = sorted(d for d in os.listdir(
            proc_dir) if (proc_dir / d).is_dir())

        raw_data = []
        self.record_stats = {}
        for rec in rec_ids:
            rec_path = proc_dir / rec
            ts_path = rec_path / "time_series.csv"
            if not ts_path.exists():
                continue

            df = pd.read_csv(ts_path)
            df["_ts_raw"] = pd.to_datetime(df["date_time"])
            df = df.sort_values("_ts_raw")

            feat_cols = [c for c in df.columns if c not in (
                "date_time", "record_id", "_ts_raw")]

            if self.feature_names is None:
                self.feature_names = feat_cols
                self.target_idx = (
                    feat_cols.index(config.target_column)
                    if config.target_column and config.target_column in feat_cols
                    else 0
                )
                self.target_name = feat_cols[self.target_idx]
                self.num_channels = len(feat_cols)

            df = pd.concat([df[self.feature_names], df[["_ts_raw"]]], axis=1)
            means = df[self.feature_names].mean()
            stds = df[self.feature_names].std()
            self.record_stats[rec] = {
                "mean": torch.tensor(means.values.astype("float32"), device=self.device),
                "std": torch.tensor(stds.values.astype("float32"), device=self.device),
            }

            for col in self.feature_names:
                std = stds[col]
                mean = means[col]
                df[col] = (df[col] - mean) / std if std else df[col] - mean

            secs = (df["_ts_raw"] - df["_ts_raw"].min()).dt.total_seconds()
            tt = torch.tensor((secs / self._sec_per_unit).values,
                              dtype=torch.float32, device=self.device)
            vals_np = df[self.feature_names].values.astype("float32")
            vals = torch.nan_to_num(torch.tensor(vals_np, device=self.device))
            mask = torch.tensor((~pd.isna(vals_np)).astype(
                "float32"), device=self.device)

            texts = load_texts_for_record(
                rec_path=rec_path,
                base_time=df["_ts_raw"].min(),
                sec_per_unit=self._sec_per_unit,
                use_text_embeddings=config.use_text_embeddings,
                llm_model_fusion=config.llm_model_fusion,
                llm_layers_fusion=config.llm_layers_fusion,
                max_length=config.max_length,
                device=self.device,
                data_name=config.dataset,
            )
            raw_data.append((rec, tt, vals, mask, texts, df["_ts_raw"].min()))

        chunks = []
        for rec, tt, vals, mask, record_texts, base_time in raw_data:
            anchor = None
            if config.chunk_anchor_date is not None:
                anchor = (
                    pd.Timestamp(config.chunk_anchor_date) - base_time
                ).total_seconds() / self._sec_per_unit
            rec_chunks = create_chunks_from_record(
                rec_id=rec,
                tt=tt,
                vals=vals,
                mask=mask,
                record_texts=record_texts,
                lookback=config.lookback,
                horizon=config.horizon,
                stride=config.stride,
                anchor=anchor,
            )

            if rec_chunks:
                drop_count = sum(1 for c in rec_chunks if not c[5])
                no_text_ratio = drop_count / len(rec_chunks)
                print(
                    f"Record {rec}: {len(rec_chunks)} chunks, "
                    f"{drop_count} without text ({no_text_ratio:.1%})"
                )

            chunks.extend(rec_chunks)

        print(f"[ChunkedTimeSeriesDataset] Total: {len(chunks)} chunks")
        return chunks

    def _coerce_record_stats(self, stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for rec_id, rec_stats in stats.items():
            out[rec_id] = {
                "mean": torch.as_tensor(rec_stats["mean"], dtype=torch.float32, device=self.device),
                "std": torch.as_tensor(rec_stats["std"], dtype=torch.float32, device=self.device),
            }
        return out

    def _load_record_stats_from_raw(self) -> Dict[str, Dict[str, torch.Tensor]]:
        config = self.config
        proc_dir = config.dataset_path / "processed"
        stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.feature_names is None:
            return stats
        for rec in sorted(d for d in os.listdir(proc_dir) if (proc_dir / d).is_dir()):
            ts_path = proc_dir / rec / "time_series.csv"
            if not ts_path.exists():
                continue
            df = pd.read_csv(ts_path)
            df["_ts_raw"] = pd.to_datetime(df["date_time"])
            df = df.sort_values("_ts_raw")
            df = pd.concat([df[self.feature_names], df[["_ts_raw"]]], axis=1)
            means = df[self.feature_names].mean()
            stds = df[self.feature_names].std()
            stats[rec] = {
                "mean": torch.tensor(means.values.astype("float32"), device=self.device),
                "std": torch.tensor(stds.values.astype("float32"), device=self.device),
            }
        return stats

    def _cache_params(self) -> dict:
        c = self.config
        return {
            "dataset_path": str(c.dataset_path.resolve()),
            "lookback": c.lookback,
            "horizon": c.horizon,
            "stride": c.stride,
            "time_unit": c.time_unit,
            "use_text_embeddings": c.use_text_embeddings,
            "llm_model_fusion": c.llm_model_fusion,
            "llm_layers_fusion": c.llm_layers_fusion,
        }

    def _cache_path(self) -> Path:
        params = self._cache_params()
        key = hashlib.md5(json.dumps(
            params, sort_keys=True).encode()).hexdigest()[:12]
        return self._cache_dir / f"time_imm_chunks_{key}.pkl"

    def _save_cache(self, chunks: List[Tuple], cache_path: Path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        chunks_cpu = []
        for chunk_id, tt, vals, mask, texts, has_text in chunks:
            tt_cpu = tt.cpu() if torch.is_tensor(tt) else tt
            vals_cpu = vals.cpu() if torch.is_tensor(vals) else vals
            mask_cpu = mask.cpu() if torch.is_tensor(mask) else mask
            texts_cpu = [(t, p.cpu() if torch.is_tensor(p) else p)
                         for t, p in texts]
            chunks_cpu.append(
                (chunk_id, tt_cpu, vals_cpu, mask_cpu, texts_cpu, has_text))

        with open(cache_path, "wb") as f:
            pickle.dump({
                "params": self._cache_params(),
                "chunks": chunks_cpu,
                "feature_names": self.feature_names,
                "num_channels": self.num_channels,
                "target_idx": self.target_idx,
                "target_name": self.target_name,
                "record_stats": {
                    rec_id: {
                        "mean": rec_stats["mean"].cpu(),
                        "std": rec_stats["std"].cpu(),
                    }
                    for rec_id, rec_stats in self.record_stats.items()
                },
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[ChunkedTimeSeriesDataset] Cached to {cache_path}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]



class RawTimeIMMDataset(Dataset):

    def __init__(
        self,
        base_dataset: ChunkedTimeSeriesDataset,
        indices: List[int],
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        chunk_id, tt, vals, mask, texts, has_text = self.base_dataset[self.indices[idx]]
        return chunk_id, tt, vals, mask, texts



@dataclass
class ResampledSample:
    chunk_id: str
    hist_tp: torch.Tensor
    hist_data: torch.Tensor
    hist_mask: torch.Tensor
    fut_tp: torch.Tensor
    fut_data: torch.Tensor
    fut_mask: torch.Tensor
    notes: List[Tuple[float, Any]]
    raw_fut_tp: Optional[torch.Tensor] = None
    raw_fut_data: Optional[torch.Tensor] = None
    raw_fut_mask: Optional[torch.Tensor] = None
    raw_hist_tp: Optional[torch.Tensor] = None
    raw_hist_data: Optional[torch.Tensor] = None
    raw_hist_mask: Optional[torch.Tensor] = None
    record_mean: Optional[torch.Tensor] = None
    record_std: Optional[torch.Tensor] = None


VALID_FILL_POLICIES = {"global_mean", "sequence_mean"}


class ResampledTimeIMMDataset(Dataset):

    def __init__(
        self,
        base_dataset: ChunkedTimeSeriesDataset,
        indices: List[int],
        config: TimeIMMConfig,
        global_means: torch.Tensor,
        keep_raw: bool = False,
        fill_policy: str = "global_mean",
    ):
        super().__init__()
        if fill_policy not in VALID_FILL_POLICIES:
            raise ValueError(
                f"Unknown fill_policy '{fill_policy}'. "
                f"Expected one of {sorted(VALID_FILL_POLICIES)}."
            )
        self.config = config
        self.fill_policy = fill_policy
        self.samples: List[ResampledSample] = []

        step = config.step_units
        hist_steps = max(1, int(np.ceil(config.lookback / step)))
        fut_steps = max(1, int(np.ceil(config.horizon / step)))

        for idx in indices:
            chunk_id, tt, vals, mask, texts, has_text = base_dataset[idx]
            rec_id = str(chunk_id).rsplit("_chunk", 1)[0]
            tt = tt.cpu() if torch.is_tensor(tt) else torch.tensor(tt)
            vals = vals.cpu() if torch.is_tensor(vals) else torch.tensor(vals)
            mask = mask.cpu() if torch.is_tensor(mask) else torch.tensor(mask)

            sample = self._resample_chunk(
                chunk_id, tt, vals, mask, texts,
                hist_steps, fut_steps, step, global_means.cpu(), keep_raw
            )
            rec_stats = base_dataset.record_stats.get(rec_id)
            if rec_stats is not None:
                sample.record_mean = rec_stats["mean"].detach().cpu()
                sample.record_std = rec_stats["std"].detach().cpu()
            self.samples.append(sample)

    def _resample_chunk(
        self,
        chunk_id: str,
        tt: torch.Tensor,
        vals: torch.Tensor,
        mask: torch.Tensor,
        texts: List[Tuple[float, Any]],
        hist_steps: int,
        fut_steps: int,
        step: float,
        global_means: torch.Tensor,
        keep_raw: bool,
    ) -> ResampledSample:
        history = self.config.lookback
        horizon = self.config.horizon
        C = vals.shape[1]

        grid_hist = torch.arange(hist_steps, dtype=torch.float32) * step
        grid_fut = history + \
            torch.arange(fut_steps, dtype=torch.float32) * step

        hist_sel = tt < history
        fut_sel = tt >= history
        hist_tt, hist_vals, hist_mask = tt[hist_sel], vals[hist_sel], mask[hist_sel]
        fut_tt, fut_vals, fut_mask = tt[fut_sel], vals[fut_sel], mask[fut_sel]

        hist_fill_values = global_means
        if self.fill_policy == "sequence_mean":
            hist_fill_values = global_means.clone().float()
            for c in range(C):
                obs_c = hist_mask[:, c] > 0
                if obs_c.any():
                    hist_fill_values[c] = hist_vals[obs_c, c].float().mean()

        hist_fill, hist_mask_fill = self._resample_to_grid(
            hist_tt, hist_vals, hist_mask, grid_hist, hist_fill_values, C
        )

        fut_fill, fut_mask_fill = self._resample_to_grid(
            fut_tt, fut_vals, fut_mask, grid_fut, global_means, C
        )

        snapped = snap_texts_to_grid(texts, step, hist_steps, history)
        mapped_notes = [(snapped_time, payload) for _, snapped_time, payload in snapped]

        sample = ResampledSample(
            chunk_id=chunk_id,
            hist_tp=grid_hist,
            hist_data=hist_fill,
            hist_mask=hist_mask_fill,
            fut_tp=grid_fut,
            fut_data=fut_fill,
            fut_mask=fut_mask_fill,
            notes=mapped_notes,
        )

        if keep_raw:
            sample.raw_fut_tp = fut_tt
            sample.raw_fut_data = fut_vals
            sample.raw_fut_mask = fut_mask
            sample.raw_hist_tp = hist_tt
            sample.raw_hist_data = hist_vals
            sample.raw_hist_mask = hist_mask

        return sample

    def _resample_to_grid(
        self,
        tt: torch.Tensor,
        vals: torch.Tensor,
        mask: torch.Tensor,
        grid: torch.Tensor,
        fill_values: torch.Tensor,
        C: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_len = grid.size(0)
        fill = torch.zeros(grid_len, C, dtype=torch.float32)
        mask_fill = torch.zeros(grid_len, C, dtype=torch.float32)

        for c in range(C):
            obs_idx = mask[:, c] > 0
            col = torch.full(
                (grid_len,), fill_values[c].item(), dtype=torch.float32)
            col_mask = torch.zeros(grid_len, dtype=torch.float32)

            if obs_idx.any():
                t_obs = tt[obs_idx]
                v_obs = vals[obs_idx, c].float()
                diff = torch.abs(grid.view(-1, 1) - t_obs.view(1, -1))
                nn_idx = diff.argmin(dim=0)

                sum_col = torch.zeros_like(col)
                count_col = torch.zeros_like(col)
                sum_col.scatter_add_(0, nn_idx, v_obs)
                count_col.scatter_add_(0, nn_idx, torch.ones_like(v_obs))

                observed = count_col > 0
                col[observed] = sum_col[observed] / \
                    count_col[observed].clamp(min=1.0)
                col_mask[observed] = 1.0

            fill[:, c] = col
            mask_fill[:, c] = col_mask

        return fill, mask_fill

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> ResampledSample:
        return self.samples[idx]



def _process_texts_from_resampled(
    batch: List[ResampledSample],
    config: TimeIMMConfig,
    step_units: Optional[float] = None,
) -> Dict[str, Any]:
    if not (config.enable_text and config.use_text_embeddings):
        return {"text_embeddings": None, "text_tp": None, "text_mask": None}

    text_tp_list = []
    text_emb_list = []
    d_txt = None

    for s in batch:
        times = []
        embs = []
        for t, payload in s.notes:
            times.append(t)
            if torch.is_tensor(payload):
                embs.append(payload)
                if d_txt is None:
                    d_txt = payload.shape[-1]

        tp_tensor = torch.tensor(times, dtype=torch.float32) if times else torch.zeros(0)
        text_tp_list.append(tp_tensor)

        if embs and d_txt is not None:
            text_emb_list.append(torch.stack(embs, dim=0))
        elif d_txt is not None:
            text_emb_list.append(torch.zeros(0, d_txt))
        else:
            text_emb_list.append(None)

    has_embeddings = d_txt is not None and any(
        e is not None and e.numel() > 0 for e in text_emb_list
    )

    if not has_embeddings:
        return {"text_embeddings": None, "text_tp": None, "text_mask": None}

    text_tp = _left_pad_sequence(text_tp_list, padding_value=0.0)
    emb_list_safe = [
        e if e is not None and e.numel() > 0 else torch.zeros(0, d_txt)
        for e in text_emb_list
    ]
    text_emb = _left_pad_sequence(emb_list_safe, padding_value=0.0)

    B = len(batch)
    max_len = text_tp.size(1)
    text_mask = torch.zeros(B, max_len, dtype=torch.float32)
    for i, tp_tensor in enumerate(text_tp_list):
        n_texts = tp_tensor.size(0)
        if n_texts > 0:
            text_mask[i, -n_texts:] = 1.0

    if step_units is not None:
        text_tp = text_tp / step_units

    return {"text_embeddings": text_emb, "text_tp": text_tp, "text_mask": text_mask}


def collate_timeimm_resampled(
    batch: List[ResampledSample],
    config: TimeIMMConfig,
    target_idx: Optional[int] = None,
    target_name: Optional[str] = None,
    patch_emb_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hist_data = torch.stack([s.hist_data for s in batch], dim=0)
    hist_mask = torch.stack([s.hist_mask for s in batch], dim=0)
    fut_data = torch.stack([s.fut_data for s in batch], dim=0)
    fut_mask = torch.stack([s.fut_mask for s in batch], dim=0)
    obs_tp_raw = torch.stack([s.hist_tp for s in batch], dim=0)
    fut_tp_raw = torch.stack([s.fut_tp for s in batch], dim=0)

    step = config.step_units if config.use_time_rescale else None
    if step is not None:
        obs_tp = obs_tp_raw / step
        fut_tp = fut_tp_raw / step
    else:
        time_max = config.lookback + config.horizon
        obs_tp = obs_tp_raw / time_max
        fut_tp = fut_tp_raw / time_max

    out: Dict[str, Any] = {
        "observed_data": hist_data,
        "observed_tp": obs_tp,
        "observed_mask": hist_mask,
        "data_to_predict": fut_data,
        "tp_to_predict": fut_tp,
        "tp_to_predict_raw": fut_tp_raw,
        "mask_predicted_data": fut_mask,
        "sample_keys": [str(s.chunk_id) for s in batch],
    }

    if target_idx is not None:
        out["target_idx"] = target_idx
    if target_name is not None:
        out["target_name"] = target_name

    if patch_emb_data is not None:
        settings = patch_emb_data.get("settings", {})
        n_patches = settings.get("n_patches", 1)
        input_token_len = int(
            settings.get("input_token_len", SUNDIAL_INPUT_TOKEN_LEN)
        )
        d_txt = None
        for chunk_data in patch_emb_data.get("chunk_embeddings", {}).values():
            for emb in chunk_data.get("patch_embeddings", {}).values():
                if torch.is_tensor(emb):
                    d_txt = emb.shape[-1]
                    break
            if d_txt is not None:
                break

        if d_txt is not None:
            emb_list = []
            mask_list = []
            for s in batch:
                emb, mask = get_patch_embeddings_for_chunk(
                    patch_emb_data, s.chunk_id, n_patches, d_txt
                )
                emb_list.append(emb)
                mask_list.append(mask)
            out["text_embeddings"] = torch.stack(emb_list, dim=0)
            out["patch_text_mask"] = torch.stack(mask_list, dim=0)
            padding_length = n_patches * input_token_len - obs_tp.shape[1]
            tp_padded = F.pad(obs_tp, (padding_length, 0), value=0.0)
            out["text_tp"] = tp_padded.unfold(
                dimension=-1, size=input_token_len, step=input_token_len
            ).max(dim=-1).values
    else:
        text_info = _process_texts_from_resampled(batch, config, step_units=step)
        if text_info["text_embeddings"] is not None:
            out["text_embeddings"] = text_info["text_embeddings"]
            out["text_tp"] = text_info["text_tp"]
            out["text_mask"] = text_info["text_mask"]

    _add_raw_tensors(out, batch, hist_data.size(-1))

    return out


def _add_raw_tensors(out: Dict[str, Any], batch: List[ResampledSample], num_channels: int):
    raw_fut_list = [(s.raw_fut_tp, s.raw_fut_data, s.raw_fut_mask)
                    for s in batch]
    if any(r[0] is not None for r in raw_fut_list):
        fut_tt_raw, fut_vals_raw, fut_mask_raw = [], [], []
        for tp, vals, mask in raw_fut_list:
            if tp is None:
                fut_tt_raw.append(torch.zeros(0))
                fut_vals_raw.append(torch.zeros(0, num_channels))
                fut_mask_raw.append(torch.zeros(0, num_channels))
            else:
                fut_tt_raw.append(tp)
                fut_vals_raw.append(vals)
                fut_mask_raw.append(mask)
        out["raw_tp_to_predict"] = pad_sequence(
            fut_tt_raw, batch_first=True, padding_value=0.0)
        out["raw_data_to_predict"] = pad_sequence(
            fut_vals_raw, batch_first=True, padding_value=0.0)
        out["raw_mask_predicted_data"] = pad_sequence(
            fut_mask_raw, batch_first=True, padding_value=0.0)

    raw_hist_list = [(s.raw_hist_tp, s.raw_hist_data, s.raw_hist_mask)
                     for s in batch]
    if any(r[0] is not None for r in raw_hist_list):
        hist_tt_raw, hist_vals_raw, hist_mask_raw = [], [], []
        for tp, vals, mask in raw_hist_list:
            if tp is None:
                hist_tt_raw.append(torch.zeros(0))
                hist_vals_raw.append(torch.zeros(0, num_channels))
                hist_mask_raw.append(torch.zeros(0, num_channels))
            else:
                hist_tt_raw.append(tp)
                hist_vals_raw.append(vals)
                hist_mask_raw.append(mask)
        out["raw_tp_observed"] = pad_sequence(
            hist_tt_raw, batch_first=True, padding_value=0.0)
        out["raw_data_observed"] = pad_sequence(
            hist_vals_raw, batch_first=True, padding_value=0.0)
        out["raw_mask_observed"] = pad_sequence(
            hist_mask_raw, batch_first=True, padding_value=0.0)

    if all(s.record_mean is not None and s.record_std is not None for s in batch):
        out["record_mean"] = torch.stack([s.record_mean for s in batch], dim=0)
        out["record_std"] = torch.stack([s.record_std for s in batch], dim=0)



def split_indices(
    chunks: List[Tuple],
    method: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[List[int], List[int], List[int]]:
    if method == "instance":
        rec_ids = sorted({cid.rsplit("_chunk", 1)[0] for cid, *_ in chunks})
        train_recs, test_recs = train_test_split(
            rec_ids, train_size=0.8, random_state=42, shuffle=True
        )
        train_recs, val_recs = train_test_split(
            train_recs, train_size=0.75, random_state=42, shuffle=False
        )

        train_idx = [i for i, (cid, *_) in enumerate(chunks)
                     if cid.rsplit("_chunk", 1)[0] in train_recs]
        val_idx = [i for i, (cid, *_) in enumerate(chunks)
                   if cid.rsplit("_chunk", 1)[0] in val_recs]
        test_idx = [i for i, (cid, *_) in enumerate(chunks)
                    if cid.rsplit("_chunk", 1)[0] in test_recs]

    else:
        grouped = defaultdict(list)
        for i, (cid, *_) in enumerate(chunks):
            rec_id, idx_str = cid.rsplit("_chunk", 1)
            grouped[rec_id].append((int(idx_str), i))

        train_idx, val_idx, test_idx = [], [], []
        for rec_id, lst in grouped.items():
            lst.sort(key=lambda x: x[0])
            N = len(lst)
            t_end = int(N * train_ratio)
            v_end = int(N * (train_ratio + val_ratio))
            train_idx += [i for _, i in lst[:t_end]]
            val_idx += [i for _, i in lst[t_end:v_end]]
            test_idx += [i for _, i in lst[v_end:]]

    print(
        f"Split ({method}): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx



def compute_global_means(
    base_dataset: ChunkedTimeSeriesDataset,
    indices: List[int],
) -> torch.Tensor:
    sample = base_dataset[indices[0]]
    _, _, vals0, _, _, _ = sample
    C = vals0.shape[1]
    device = vals0.device if torch.is_tensor(vals0) else torch.device("cpu")

    sum_c = torch.zeros(C, dtype=torch.float64, device=device)
    count_c = torch.zeros(C, dtype=torch.float64, device=device)

    for idx in indices:
        _, _, vals, mask, _, _ = base_dataset[idx]
        vals = vals.to(torch.float64) if torch.is_tensor(
            vals) else torch.tensor(vals, dtype=torch.float64)
        mask = mask.to(torch.float64) if torch.is_tensor(
            mask) else torch.tensor(mask, dtype=torch.float64)
        sum_c += (vals * mask).sum(dim=0)
        count_c += mask.sum(dim=0)

    return (sum_c / count_c.clamp(min=1.0)).to(torch.float32)



def build_timeimm_dataloaders(
    data_path: str,
    dataset: str,
    lookback: int,
    horizon: int,
    stride: int,
    batch_size: int,
    model: str,
    split_method: str = "sample",
    time_unit: str = "hours",
    input_token_len: int = SUNDIAL_INPUT_TOKEN_LEN,
    enable_text: bool = True,
    use_text_embeddings: bool = False,
    llm_model_fusion: Optional[str] = None,
    llm_layers_fusion: Optional[int] = None,
    max_length: int = 1024,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    target_column: Optional[str] = None,
    use_time_rescale: bool = False,
    resample_interval_sec: Optional[int] = None,
    use_covariates: bool = False,
    cache_dir: Optional[Path | str] = None,
    timeimm_irregularity: str = "resampled",
    tsfm_model: str = "sundial",
    device: str = "cuda:0",
    selected_chunk_ids: Optional[List[str]] = None,
    chunk_anchor_date: Optional[str] = None,
    use_patch_text_embeds: bool = True,
    fill_policy: str = "global_mean",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    config = TimeIMMConfig(
        data_path=data_path,
        dataset=dataset,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        time_unit=time_unit,
        resample_interval_sec=resample_interval_sec,
        enable_text=enable_text,
        use_text_embeddings=use_text_embeddings,
        llm_model_fusion=llm_model_fusion,
        llm_layers_fusion=llm_layers_fusion,
        max_length=max_length,
        split_method=split_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_column=target_column,
        use_covariates=use_covariates,
        use_time_rescale=use_time_rescale,
        chunk_anchor_date=chunk_anchor_date,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    base_ds = ChunkedTimeSeriesDataset(config, device=torch.device(device))
    print(
        f"[build_timeimm_dataloaders] Dataset: {dataset}, {len(base_ds)} chunks, "
        f"irregularity={timeimm_irregularity}, fill_policy={fill_policy}")

    train_idx, val_idx, test_idx = split_indices(
        base_ds.chunks, split_method, train_ratio, val_ratio
    )

    def filter_has_text(indices):
        return [i for i in indices if base_ds.chunks[i][5]]

    total_before = len(train_idx) + len(val_idx) + len(test_idx)
    train_idx = filter_has_text(train_idx)
    val_idx = filter_has_text(val_idx)
    test_idx = filter_has_text(test_idx)
    total_after = len(train_idx) + len(val_idx) + len(test_idx)

    print(
        f"[build_timeimm_dataloaders] Filtered to text-available samples: "
        f"{total_after}/{total_before} ({100*total_after/total_before:.1f}%)"
    )
    print(
        f"[build_timeimm_dataloaders] Split sizes: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    if selected_chunk_ids:
        selected_set = set(selected_chunk_ids)
        split_membership: Dict[int, str] = {}
        for i in train_idx:
            split_membership[i] = "train"
        for i in val_idx:
            split_membership[i] = "val"
        for i in test_idx:
            split_membership[i] = "test"

        selected_eval_idx = [
            i for i, (chunk_id, *_rest) in enumerate(base_ds.chunks)
            if chunk_id in selected_set and i in split_membership
        ]
        if not selected_eval_idx:
            raise RuntimeError(
                "No selected Time-IMM chunks matched the current dataset/split configuration."
            )

        matched_chunk_ids = {base_ds.chunks[i][0] for i in selected_eval_idx}
        missing_chunk_ids = sorted(selected_set - matched_chunk_ids)
        split_counts = {"train": 0, "val": 0, "test": 0}
        for i in selected_eval_idx:
            split_counts[split_membership[i]] += 1

        print(
            "[build_timeimm_dataloaders] Selected chunk evaluation active: "
            f"{len(selected_eval_idx)} matched chunks "
            f"(train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']})"
        )
        if missing_chunk_ids:
            print(
                "[build_timeimm_dataloaders] Warning: requested chunk IDs not found: "
                + ", ".join(missing_chunk_ids)
            )

        val_idx = list(selected_eval_idx)
        test_idx = list(selected_eval_idx)

    if not test_idx:
        raise RuntimeError(
            "No test samples; check dataset path and split ratios.")

    if timeimm_irregularity == "resampled":
        global_means = compute_global_means(base_ds, train_idx)

        train_ds = ResampledTimeIMMDataset(
            base_ds, train_idx, config, global_means, keep_raw=False,
            fill_policy=fill_policy)
        val_ds = ResampledTimeIMMDataset(
            base_ds, val_idx, config, global_means, keep_raw=True,
            fill_policy=fill_policy)
        test_ds = ResampledTimeIMMDataset(
            base_ds, test_idx, config, global_means, keep_raw=True,
            fill_policy=fill_policy)

        patch_emb_data = None
        if (
            tsfm_model == "sundial"
            and use_text_embeddings
            and llm_model_fusion
            and use_patch_text_embeds
        ):
            patch_emb_data = load_patch_text_embeddings_for_dataset(
                data_path=Path(data_path),
                dataset=dataset,
                llm_model_fusion=llm_model_fusion,
                llm_layers_fusion=llm_layers_fusion,
                input_token_len=input_token_len,
                device=torch.device(device),
            )
            if patch_emb_data is not None:
                print(f"[build_timeimm_dataloaders] Loaded patch embeddings for Sundial")
            else:
                print(f"[build_timeimm_dataloaders] Warning: patch embeddings not found for Sundial")

        def collate_fn(batch):
            return collate_timeimm_resampled(
                batch, config, base_ds.target_idx, base_ds.target_name,
                patch_emb_data=patch_emb_data,
            )

    else:
        train_ds = RawTimeIMMDataset(base_ds, train_idx)
        val_ds = RawTimeIMMDataset(base_ds, val_idx)
        test_ds = RawTimeIMMDataset(base_ds, test_idx)

        if tsfm_model == "sundial":
            base_collate = collate_varlen_leftpad
        elif tsfm_model in ("timemoe", "timesfm"):
            base_collate = collate_varlen_autoreg
        else:
            base_collate = collate_varlen_rightpad

        class CollateArgs:
            pass
        collate_args = CollateArgs()
        collate_args.history = lookback
        collate_args.pred_window = horizon
        collate_args.device = device
        collate_args.use_time_rescale = use_time_rescale
        collate_args.resample_interval_sec = resample_interval_sec
        collate_args.time_unit = time_unit

        def collate_fn(batch):
            return base_collate(batch, collate_args, unit_seconds_map=SEC_PER_UNIT)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    C = base_ds.num_channels
    eff_C = C + 1 if enable_text and use_text_embeddings and timeimm_irregularity == "resampled" else C
    for loader in (train_loader, val_loader, test_loader):
        loader.num_channels = eff_C

    return train_loader, val_loader, test_loader
