
from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from experiments.utils import atomic_torch_save, is_embedding_file_complete




DOMAIN_META: Dict[str, Dict[str, str]] = {
    "LEU": {
        "task": "Forecast London household electricity usage (30-minute intervals).",
        "causal_closing": "These factors influence electricity consumption patterns.",
    },
    "MSPG": {
        "task": "Forecast Melbourne solar power generation (15-minute intervals).",
        "causal_closing": "These factors influence solar power output.",
    },
    "PTF": {
        "task": "Forecast Paris highway traffic flow (1-hour intervals).",
        "causal_closing": "These factors influence traffic flow patterns.",
    },
}


def build_cgtsf_text(
    domain: str,
    raw_text: str,
    add_header: bool = True,
) -> str:
    if not add_header:
        return raw_text

    meta = DOMAIN_META.get(domain, {})
    task = meta.get("task", f"Forecast {domain}.")
    causal_closing = meta.get("causal_closing", "These factors influence the forecast.")

    return f"Task: {task}\n\n{raw_text}\n\n{causal_closing}"


def get_cgtsf_default_text(domain: str) -> str:
    meta = DOMAIN_META.get(domain, {})
    task = meta.get("task", f"Forecast {domain}.")
    return f"Task: {task}"




@dataclass
class CGTSFSample:
    x: np.ndarray
    y: np.ndarray
    text: str
    domain: str
    order: int
    date: pd.Timestamp
    idx: int
    has_text: bool = True


class CGTSFDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        horizon_map: Dict[str, int],
        domains: Optional[List[str]] = None,
        lookback_overrides: Optional[Dict[str, int]] = None,
        allowed_lookback_multipliers: Optional[List[int]] = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        data_usage_ratio: float = 1.0,
        scale: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.horizon_map = horizon_map
        self.domains = domains or self._discover_domains()
        self.lookback_overrides = lookback_overrides or {}
        self.allowed_lookback_multipliers = allowed_lookback_multipliers or [2, 3, 4, 5]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.data_usage_ratio = float(data_usage_ratio)
        self.scale = scale
        self.cache_dir = cache_dir or str(self.data_path / "cache_data")

        self.samples: List[CGTSFSample] = []
        self.domain_lengths: Dict[str, int] = {}
        self.domain_scalers: Dict[str, Tuple[float, float]] = {}
        self.num_channels: int = 1

        cache_path = self._maybe_cache_path()
        if cache_path is not None and cache_path.exists():
            payload = None
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                print(
                    f"[CGTSF] Cache load failed ({e}); rebuilding: {cache_path}")

            if payload is not None and payload.get("params") == self._cache_params():
                self.samples = payload.get("samples", [])
                self.domain_lengths = payload.get("domain_lengths", {})
                self.domain_scalers = payload.get("domain_scalers", {})
                self.domains = payload.get("domains", self.domains)
                print(
                    f"[CGTSF] Loaded cached samples ({len(self.samples)}) from {cache_path}")
                return
            else:
                print(
                    f"[CGTSF] Cache params mismatch or unavailable, rebuilding: {cache_path}")

        for domain in self.domains:
            csv_path = self.data_path / domain / f"{domain}.csv"
            if not csv_path.exists():
                print(f"[CGTSF] Missing CSV for domain '{domain}': {csv_path}")
                continue
            if domain not in self.horizon_map:
                print(
                    f"[CGTSF] No horizon configured for domain '{domain}', skipping.")
                continue

            df = pd.read_csv(csv_path)
            if "Date" not in df.columns or "Hist" not in df.columns or "Pred" not in df.columns:
                print(
                    f"[CGTSF] CSV missing required columns (Date/Hist/Pred): {csv_path}")
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            total_rows = len(df)
            if self.data_usage_ratio < 1.0 and total_rows > 0:
                keep_n = max(1, int(total_rows * self.data_usage_ratio))
                df = df.iloc[:keep_n]
                pct = 100.0 * self.data_usage_ratio
                print(
                    f"[CGTSF] Using first {keep_n}/{total_rows} rows ({pct:.1f}%) for domain '{domain}'")

            horizon = self.horizon_map[domain]
            lookback = self._resolve_lookback(domain, horizon)

            raw_rows = []
            for row_idx, row in df.iterrows():
                try:
                    hist = np.asarray(ast.literal_eval(
                        row["Hist"]), dtype=np.float32)
                    pred = np.asarray(ast.literal_eval(
                        row["Pred"]), dtype=np.float32)
                except Exception:
                    continue

                if len(pred) != horizon or len(hist) < lookback:
                    continue

                text = row.get("Text", "")
                text = text if isinstance(text, str) else ""
                csv_idx = int(row.get("Idx", row_idx))
                raw_rows.append(
                    (
                        hist[-lookback:],
                        pred[:horizon],
                        text,
                        row["Date"],
                        csv_idx,
                    )
                )

            domain_n = len(raw_rows)
            self.domain_lengths[domain] = domain_n
            if domain_n == 0:
                print(
                    f"[CGTSF] No usable rows for domain '{domain}' after filtering, skipping.")
                continue

            train_end = int(self.train_ratio * domain_n)
            val_end = int((self.train_ratio + self.val_ratio) * domain_n)

            if self.scale and train_end > 0:
                hist_stack = np.concatenate(
                    [h for h, _, _, _, _ in raw_rows[:train_end]])
                mean = float(hist_stack.mean())
                std = float(hist_stack.std())
                if std < 1e-6:
                    std = 1.0
            else:
                mean, std = 0.0, 1.0
            self.domain_scalers[domain] = (mean, std)

            for order_idx, (hist, pred, text, date, csv_idx) in enumerate(raw_rows):
                if self.scale:
                    hist = (hist - mean) / std
                    pred = (pred - mean) / std
                has_text = len(text.strip()) > 0
                self.samples.append(
                    CGTSFSample(
                        x=hist.astype(np.float32),
                        y=pred.astype(np.float32),
                        text=text,
                        domain=domain,
                        order=order_idx,
                        date=date,
                        idx=csv_idx,
                        has_text=has_text,
                    )
                )

        cache_path = self._maybe_cache_path()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "params": self._cache_params(),
                "samples": self.samples,
                "domain_lengths": self.domain_lengths,
                "domain_scalers": self.domain_scalers,
                "domains": self.domains,
            }
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(payload, f)
                print(
                    f"[CGTSF] Cached samples ({len(self.samples)}) to {cache_path}")
            except Exception as e:
                print(f"[CGTSF] Failed to write cache ({e}): {cache_path}")

    def _cache_params(self) -> dict:
        return {
            "data_path": str(self.data_path),
            "domains": self.domains,
            "horizon_map": self.horizon_map,
            "lookback_overrides": self.lookback_overrides,
            "allowed_lookback_multipliers": self.allowed_lookback_multipliers,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "data_usage_ratio": self.data_usage_ratio,
            "scale": self.scale,
        }

    def _maybe_cache_path(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        params = self._cache_params()
        key = hashlib.md5(json.dumps(
            params, sort_keys=True).encode("utf-8")).hexdigest()
        cache_root = Path(self.cache_dir)
        filename = f"cgtsf_cache_{key}.pkl"
        return cache_root / filename

    def _discover_domains(self) -> List[str]:
        return sorted(
            p.name for p in self.data_path.iterdir() if p.is_dir() and (p / f"{p.name}.csv").exists()
        )

    def _resolve_lookback(self, domain: str, horizon: int) -> int:
        desired = self.lookback_overrides.get(domain)
        allowed = [horizon * m for m in self.allowed_lookback_multipliers]
        if desired is None:
            return max(allowed)
        return desired

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.from_numpy(s.x)
        y = torch.from_numpy(s.y)
        return x, y, s.text




class CGTSFTextEmbeddingCache:

    def __init__(
        self,
        cache_dir: Path,
        llm_model: str = "Qwen3_E4B",
        llm_layers: Optional[int] = None,
        data_root: Optional[Path] = None,
        device: str = "cuda",
        add_header: bool = True,
        encode_batch_size: int = 32,
        dtype: torch.dtype = torch.float16,
        domains: Optional[List[str]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.data_root = data_root or self.cache_dir.parent
        self.device = device
        self.add_header = add_header
        self.encode_batch_size = encode_batch_size
        self.dtype = dtype

        self.d_txt: Optional[int] = None
        self.data: Dict[str, Dict[int, torch.Tensor]] = {}

        self._tokenizer = None
        self._llm_model = None

        if domains is not None:
            domain_dirs = [(self.data_root / d, d) for d in sorted(domains)]
        else:
            domain_dirs = [
                (d, d.name) for d in sorted(self.data_root.iterdir())
                if d.is_dir() and d.name not in ("cache_data", "text_embeddings", "patch_text_embeddings")
            ]

        for domain_dir, domain in domain_dirs:
            csv_path = domain_dir / f"{domain}.csv"
            if not csv_path.exists():
                continue

            emb_file = self._get_embedding_filename(domain)
            emb_path = self.cache_dir / emb_file

            if not is_embedding_file_complete(emb_path):
                self._ensure_embeddings(domain, csv_path, emb_path)

            if is_embedding_file_complete(emb_path):
                try:
                    payload = torch.load(emb_path, map_location="cpu", weights_only=False)
                    embeddings_by_idx = payload["embeddings_by_idx"]
                    num_embeddings = len(embeddings_by_idx)
                    if num_embeddings > 0:
                        sample_emb = next(iter(embeddings_by_idx.values()))
                        if self.d_txt is None:
                            self.d_txt = sample_emb.shape[-1]

                    self.data[domain] = embeddings_by_idx
                    print(f"[CGTSFTextEmbeddingCache] Loaded {domain}: {num_embeddings} embeddings (keyed by Idx)")
                except Exception as e:
                    print(f"[CGTSFTextEmbeddingCache] Failed to load {emb_path}: {e}")

        if self.data:
            print(f"[CGTSFTextEmbeddingCache] Loaded {len(self.data)} domains, d_txt={self.d_txt}")
        else:
            print(f"[CGTSFTextEmbeddingCache] No embeddings found in {cache_dir}")

    def _ensure_embeddings(self, domain: str, csv_path: Path, emb_path: Path) -> None:
        print(f"[CGTSFTextEmbeddingCache] Computing embeddings for {domain}...")

        df = pd.read_csv(csv_path)
        if "Text" not in df.columns:
            print(f"[CGTSFTextEmbeddingCache] Skipping {domain}: 'Text' column not found.")
            return

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        if self._tokenizer is None:
            from experiments.utils import load_llm
            print(f"[CGTSFTextEmbeddingCache] Loading LLM model: {self.llm_model}")
            self._tokenizer, self._llm_model = load_llm(
                self.llm_model,
                self.llm_layers,
                self.device,
                use_device_map=False,
            )

        texts: List[str] = []
        indices: List[int] = []
        dates: List[str] = []
        valid_mask: List[bool] = []

        default_text = get_cgtsf_default_text(domain)

        for row_idx, row in df.iterrows():
            raw_text = str(row["Text"]).strip()
            has_valid_text = bool(raw_text) and raw_text.lower() != "nan"

            if has_valid_text:
                final_text = build_cgtsf_text(domain, raw_text, add_header=self.add_header)
            else:
                final_text = default_text

            texts.append(final_text)
            indices.append(int(row.get("Idx", row_idx)))
            dates.append(str(row.get("Date", "")))
            valid_mask.append(has_valid_text)

        if len(texts) == 0:
            print(f"[CGTSFTextEmbeddingCache] Skipping {domain}: no rows found.")
            return

        valid_count = sum(valid_mask)
        print(f"[CGTSFTextEmbeddingCache] Encoding {len(texts)} text entries ({valid_count} with valid text)...")

        from experiments.utils import embed_notes
        embeddings = self._encode_texts_batch(texts, desc=domain)
        embeddings = embeddings.to(dtype=self.dtype)

        embeddings_by_idx = {idx: embeddings[i] for i, idx in enumerate(indices)}

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "domain": domain,
            "embeddings_by_idx": embeddings_by_idx,
            "indices": indices,
            "dates": dates,
            "valid_mask": valid_mask,
            "dtype": str(self.dtype).replace("torch.", ""),
            "add_header": self.add_header,
            "llm_model": self.llm_model,
            "llm_layers": self.llm_layers,
        }
        atomic_torch_save(payload, emb_path)
        print(f"[CGTSFTextEmbeddingCache] Saved {len(texts)} embeddings (keyed by Idx) to {emb_path}")

    def _encode_texts_batch(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        from experiments.utils import embed_notes

        outputs: List[torch.Tensor] = []

        for i in tqdm(range(0, len(texts), self.encode_batch_size), desc=desc, leave=False):
            batch_texts = texts[i: i + self.encode_batch_size]
            notes_batch = [[text] for text in batch_texts]
            emb, _ = embed_notes(notes_batch, self._tokenizer, self._llm_model)
            outputs.append(emb.squeeze(1).cpu())
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        if outputs:
            return torch.cat(outputs, dim=0)
        return torch.empty((0, 0), dtype=torch.float32)

    def _get_embedding_filename(self, domain: str) -> str:
        layers_str = self.llm_layers if self.llm_layers else "full"
        return f"{domain}_text_embeddings_model={self.llm_model}_layers={layers_str}.pt"

    def get(self, domain: str, idx: int) -> Optional[torch.Tensor]:
        if domain not in self.data:
            return None

        embeddings_by_idx = self.data[domain]
        return embeddings_by_idx.get(idx, None)

    def has(self, domain: str) -> bool:
        return domain in self.data




class CGTSFTimestepDataset(Dataset):

    def __init__(
        self,
        base: CGTSFDataset,
        indices: List[int],
        emb_cache: Optional[CGTSFTextEmbeddingCache],
    ) -> None:
        super().__init__()
        self.base = base
        self.indices = indices
        self.emb_cache = emb_cache

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        s = self.base.samples[idx]

        lookback = len(s.x)
        horizon = len(s.y)

        text_embedding: Optional[torch.Tensor] = None
        if self.emb_cache is not None and self.emb_cache.has(s.domain):
            text_embedding = self.emb_cache.get(s.domain, s.idx)

        return {
            "x": torch.from_numpy(s.x).float(),
            "y": torch.from_numpy(s.y).float(),
            "lookback": lookback,
            "horizon": horizon,
            "text_embedding": text_embedding,
            "domain": s.domain,
            "order": s.order,
            "date": s.date,
            "idx": s.idx,
        }


def collate_cgtsf_timestep(batch: List[Dict[str, Any]], d_txt: Optional[int] = None) -> Dict[str, Any]:
    B = len(batch)
    lookback = batch[0]["lookback"]
    horizon = batch[0]["horizon"]

    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)

    if d_txt is None:
        for b in batch:
            if b["text_embedding"] is not None:
                d_txt = b["text_embedding"].shape[0]
                break

    if d_txt is not None:
        text_emb = torch.zeros(B, 1, d_txt)
        text_mask = torch.zeros(B, 1)
        for b_idx, b in enumerate(batch):
            if b["text_embedding"] is not None:
                text_emb[b_idx, 0, :] = b["text_embedding"]
                text_mask[b_idx, 0] = 1.0
        text_tp = torch.zeros(B, 1)
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
        "sample_keys": [f"{b['domain']}_idx{b['idx']}" for b in batch],
    }

    if text_emb is not None:
        result["text_embeddings"] = text_emb
        result["text_tp"] = text_tp
        result["text_mask"] = text_mask

    return result


def _make_cgtsf_timestep_collate_fn(d_txt: Optional[int] = None):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_cgtsf_timestep(batch, d_txt=d_txt)
    return collate_fn


def build_cgtsf_dataloaders(
    data_path: str,
    horizon_map: Dict[str, int],
    domains: Optional[List[str]] = None,
    lookback_overrides: Optional[Dict[str, int]] = None,
    allowed_lookback_multipliers: Optional[List[int]] = None,
    batch_size: int = 16,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    data_usage_ratio: float = 1.0,
    scale: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = CGTSFDataset(
        data_path=data_path,
        horizon_map=horizon_map,
        domains=domains,
        lookback_overrides=lookback_overrides,
        allowed_lookback_multipliers=allowed_lookback_multipliers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        data_usage_ratio=data_usage_ratio,
        scale=scale,
        cache_dir=cache_dir,
    )

    domain_to_indices: Dict[str, List[int]] = {d: [] for d in dataset.domains}
    for idx, sample in enumerate(dataset.samples):
        domain_to_indices[sample.domain].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for domain in dataset.domains:
        idxs = domain_to_indices.get(domain, [])
        n = dataset.domain_lengths.get(domain, len(idxs))
        if n == 0:
            continue
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        for i in idxs:
            order = dataset.samples[i].order
            if order < train_end:
                train_indices.append(i)
            elif order < val_end:
                val_indices.append(i)
            else:
                test_indices.append(i)

    def filter_has_text(indices):
        return [i for i in indices if dataset.samples[i].has_text]

    total_before = len(train_indices) + len(val_indices) + len(test_indices)
    train_indices = filter_has_text(train_indices)
    val_indices = filter_has_text(val_indices)
    test_indices = filter_has_text(test_indices)
    total_after = len(train_indices) + len(val_indices) + len(test_indices)

    print(
        f"[build_cgtsf_dataloaders] Filtered to text-available samples: "
        f"{total_after}/{total_before} ({100*total_after/total_before:.1f}%)"
    )
    print(
        f"[build_cgtsf_dataloaders] Split sizes: "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

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
    return train_loader, val_loader, test_loader


def build_cgtsf_dataloaders_timestep(
    data_path: str,
    horizon_map: Dict[str, int],
    domains: Optional[List[str]] = None,
    lookback_overrides: Optional[Dict[str, int]] = None,
    allowed_lookback_multipliers: Optional[List[int]] = None,
    batch_size: int = 16,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    data_usage_ratio: float = 1.0,
    scale: bool = True,
    cache_dir: Optional[str] = None,
    emb_cache_dir: Optional[Path] = None,
    llm_model: str = "Qwen3_E4B",
    llm_layers: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base_ds = CGTSFDataset(
        data_path=data_path,
        horizon_map=horizon_map,
        domains=domains,
        lookback_overrides=lookback_overrides,
        allowed_lookback_multipliers=allowed_lookback_multipliers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        data_usage_ratio=data_usage_ratio,
        scale=scale,
        cache_dir=cache_dir,
    )

    emb_cache: Optional[CGTSFTextEmbeddingCache] = None
    d_txt: Optional[int] = None
    if emb_cache_dir is not None:
        try:
            emb_cache = CGTSFTextEmbeddingCache(
                cache_dir=Path(emb_cache_dir),
                llm_model=llm_model,
                llm_layers=llm_layers,
                data_root=Path(data_path),
                device=device,
                domains=base_ds.domains,
            )
            d_txt = emb_cache.d_txt
        except Exception as e:
            print(f"[build_cgtsf_dataloaders_timestep] Warning: {e}")
            print("  Proceeding without text embeddings.")

    domain_to_indices: Dict[str, List[int]] = {d: [] for d in base_ds.domains}
    for idx, sample in enumerate(base_ds.samples):
        domain_to_indices[sample.domain].append(idx)

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
            order = base_ds.samples[i].order
            if order < train_end:
                train_indices.append(i)
            elif order < val_end:
                val_indices.append(i)
            else:
                test_indices.append(i)

    def filter_has_text(indices):
        return [i for i in indices if base_ds.samples[i].has_text]

    total_before = len(train_indices) + len(val_indices) + len(test_indices)
    train_indices = filter_has_text(train_indices)
    val_indices = filter_has_text(val_indices)
    test_indices = filter_has_text(test_indices)
    total_after = len(train_indices) + len(val_indices) + len(test_indices)

    print(
        f"[build_cgtsf_dataloaders_timestep] Filtered to text-available samples: "
        f"{total_after}/{total_before} ({100*total_after/total_before:.1f}%)"
    )
    print(f"[build_cgtsf_dataloaders_timestep] Split sizes: "
          f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    train_ds = CGTSFTimestepDataset(base_ds, train_indices, emb_cache)
    val_ds = CGTSFTimestepDataset(base_ds, val_indices, emb_cache)
    test_ds = CGTSFTimestepDataset(base_ds, test_indices, emb_cache)

    collate_fn = _make_cgtsf_timestep_collate_fn(d_txt=d_txt)

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

    return train_loader, val_loader, test_loader
