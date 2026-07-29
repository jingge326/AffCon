
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiments.utils import atomic_torch_save, is_embedding_file_complete



@dataclass
class SyntheticSample:
    id: int
    timeseries: np.ndarray
    text: str
    norm_stats_mean: float
    norm_stats_std: float


@dataclass
class SyntheticForecastSample:
    id: int
    history: np.ndarray
    forecast: np.ndarray
    text: str
    joint_norm_stats_mean: float
    joint_norm_stats_std: float



class SyntheticDataset(Dataset):

    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.samples: List[SyntheticSample] = []
        self._load()

    def _load(self) -> None:
        with self.jsonl_path.open("r") as f:
            for line in f:
                record = json.loads(line)
                self.samples.append(SyntheticSample(
                    id=record["id"],
                    timeseries=np.array(record["timeseries"], dtype=np.float32),
                    text=record["text"],
                    norm_stats_mean=record["norm_stats"]["mean"],
                    norm_stats_std=record["norm_stats"]["std"],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SyntheticSample:
        return self.samples[idx]


class SyntheticForecastDataset(Dataset):

    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.samples: List[SyntheticForecastSample] = []
        self._load()

    def _load(self) -> None:
        with self.jsonl_path.open("r") as f:
            for line in f:
                record = json.loads(line)
                self.samples.append(SyntheticForecastSample(
                    id=record["id"],
                    history=np.array(record["history"], dtype=np.float32),
                    forecast=np.array(record["forecast"], dtype=np.float32),
                    text=record["text"],
                    joint_norm_stats_mean=record["joint_norm_stats"]["mean"],
                    joint_norm_stats_std=record["joint_norm_stats"]["std"],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SyntheticForecastSample:
        return self.samples[idx]



class SyntheticTextEmbeddingCache:

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Path,
        llm_model: str = "Qwen3_E4B",
        llm_layers: Optional[int] = None,
        device: str = "cuda",
        encode_batch_size: int = 32,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.device = device
        self.encode_batch_size = encode_batch_size
        self.dtype = dtype

        self.d_txt: Optional[int] = None
        self.embeddings: Dict[int, torch.Tensor] = {}

        layers_str = llm_layers if llm_layers else "full"
        dataset_tag = getattr(self.dataset, "jsonl_path", Path("synthetic")).stem
        self._cache_file = (
            self.cache_dir
            / f"synthetic_text_embeddings_{dataset_tag}_model={llm_model}_layers={layers_str}.pt"
        )

        if not is_embedding_file_complete(self._cache_file):
            self._compute_and_cache()

        self._load_cache()

    def _compute_and_cache(self) -> None:
        from experiments.utils import embed_notes, load_llm

        print(f"[SyntheticTextEmbeddingCache] Computing embeddings for "
              f"{len(self.dataset)} synthetic samples...")

        tokenizer, llm_model = load_llm(
            self.llm_model, self.llm_layers, self.device, use_device_map=False,
        )

        all_embeddings: List[torch.Tensor] = []
        texts = [s.text for s in self.dataset.samples]

        for i in tqdm(range(0, len(texts), self.encode_batch_size),
                      desc="Encoding synthetic texts", leave=False):
            batch_texts = texts[i: i + self.encode_batch_size]
            notes_batch = [[t] for t in batch_texts]
            emb, _ = embed_notes(notes_batch, tokenizer, llm_model)
            all_embeddings.append(emb.squeeze(1).cpu())
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()

        embeddings = torch.cat(all_embeddings, dim=0).to(dtype=self.dtype)

        embeddings_by_id = {
            s.id: embeddings[i] for i, s in enumerate(self.dataset.samples)
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "embeddings_by_id": embeddings_by_id,
            "ids": [s.id for s in self.dataset.samples],
            "dtype": str(self.dtype).replace("torch.", ""),
            "llm_model": self.llm_model,
            "llm_layers": self.llm_layers,
        }
        atomic_torch_save(payload, self._cache_file)
        print(f"[SyntheticTextEmbeddingCache] Saved {len(embeddings_by_id)} "
              f"embeddings to {self._cache_file}")

        del tokenizer, llm_model
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def _load_cache(self) -> None:
        payload = torch.load(
            self._cache_file, map_location="cpu", weights_only=False)
        self.embeddings = payload["embeddings_by_id"]
        if self.embeddings:
            sample_emb = next(iter(self.embeddings.values()))
            self.d_txt = sample_emb.shape[-1]
        print(f"[SyntheticTextEmbeddingCache] Loaded {len(self.embeddings)} "
              f"embeddings, d_txt={self.d_txt}")

    def get(self, sample_id: int) -> Optional[torch.Tensor]:
        return self.embeddings.get(sample_id)



class SyntheticTimestepDataset(Dataset):

    def __init__(
        self,
        base: SyntheticDataset,
        indices: List[int],
        emb_cache: SyntheticTextEmbeddingCache,
    ) -> None:
        self.base = base
        self.indices = indices
        self.emb_cache = emb_cache

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = self.base[self.indices[i]]
        emb = self.emb_cache.get(sample.id)
        if emb is None:
            raise ValueError(f"Missing embedding for sample id={sample.id}")
        return {
            "timeseries": torch.from_numpy(sample.timeseries),
            "text_embedding": emb.float(),
            "id": sample.id,
        }


class SyntheticForecastTimestepDataset(Dataset):

    def __init__(
        self,
        base: SyntheticForecastDataset,
        indices: List[int],
        emb_cache: SyntheticTextEmbeddingCache,
    ) -> None:
        self.base = base
        self.indices = indices
        self.emb_cache = emb_cache

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = self.base[self.indices[i]]
        emb = self.emb_cache.get(sample.id)
        if emb is None:
            raise ValueError(f"Missing embedding for sample id={sample.id}")
        return {
            "history": torch.from_numpy(sample.history),
            "forecast": torch.from_numpy(sample.forecast),
            "text_embedding": emb.float(),
            "id": sample.id,
        }



def collate_synthetic(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = torch.tensor([b["timeseries"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    gt = torch.zeros(len(batch), max_len)
    for i, b in enumerate(batch):
        L = b["timeseries"].shape[0]
        gt[i, :L] = b["timeseries"]

    embs = torch.stack([b["text_embedding"] for b in batch], dim=0)
    return {
        "gt_series": gt,
        "text_embeddings": embs.unsqueeze(1),
        "lengths": lengths,
    }


def collate_synthetic_forecast(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    history_lengths = torch.tensor(
        [b["history"].shape[0] for b in batch], dtype=torch.long
    )
    forecast_lengths = torch.tensor(
        [b["forecast"].shape[0] for b in batch], dtype=torch.long
    )
    max_hist = int(history_lengths.max().item())
    max_forecast = int(forecast_lengths.max().item())
    batch_size = len(batch)

    observed_data = torch.zeros(batch_size, max_hist, 1)
    observed_mask = torch.zeros(batch_size, max_hist, 1)
    observed_tp = torch.zeros(batch_size, max_hist)
    data_to_predict = torch.zeros(batch_size, max_forecast, 1)
    mask_predicted_data = torch.zeros(batch_size, max_forecast, 1)
    tp_to_predict = torch.zeros(batch_size, max_forecast)

    for i, b in enumerate(batch):
        hist = b["history"]
        fut = b["forecast"]
        hist_len = hist.shape[0]
        fut_len = fut.shape[0]

        observed_tp[i, :] = torch.arange(max_hist, dtype=torch.float32)
        tp_to_predict[i, :] = torch.arange(
            hist_len, hist_len + max_forecast, dtype=torch.float32
        )
        observed_data[i, :hist_len, 0] = hist
        observed_mask[i, :hist_len, 0] = 1.0

        data_to_predict[i, :fut_len, 0] = fut
        mask_predicted_data[i, :fut_len, 0] = 1.0

    embs = torch.stack([b["text_embedding"] for b in batch], dim=0)
    text_tp = torch.zeros(batch_size, 1)
    text_mask = torch.ones(batch_size, 1)

    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "observed_tp": observed_tp,
        "data_to_predict": data_to_predict,
        "mask_predicted_data": mask_predicted_data,
        "tp_to_predict": tp_to_predict,
        "text_embeddings": embs.unsqueeze(1),
        "text_tp": text_tp,
        "text_mask": text_mask,
        "target_idx": 0,
        "history_lengths": history_lengths,
        "forecast_lengths": forecast_lengths,
    }



def build_synthetic_dataloaders(
    jsonl_path: str | Path,
    cache_dir: str | Path,
    batch_size: int = 32,
    train_ratio: float = 0.9,
    device: str = "cuda",
    llm_model: str = "Qwen3_E4B",
    llm_layers: Optional[int] = None,
    phase1_mode: str = "full_series",
) -> Tuple[DataLoader, DataLoader]:
    jsonl_path = Path(jsonl_path)
    cache_dir = Path(cache_dir)

    if phase1_mode == "full_series":
        base = SyntheticDataset(jsonl_path)
        dataset_cls = SyntheticTimestepDataset
        collate_fn = collate_synthetic
    elif phase1_mode == "forecast":
        base = SyntheticForecastDataset(jsonl_path)
        dataset_cls = SyntheticForecastTimestepDataset
        collate_fn = collate_synthetic_forecast
    else:
        raise ValueError(
            f"Unknown synthetic phase1 mode '{phase1_mode}'. "
            "Expected 'full_series' or 'forecast'."
        )

    emb_cache = SyntheticTextEmbeddingCache(
        dataset=base,
        cache_dir=cache_dir,
        llm_model=llm_model,
        llm_layers=llm_layers,
        device=device,
    )

    n = len(base)
    n_train = int(n * train_ratio)
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n))

    train_ds = dataset_cls(base, train_indices, emb_cache)
    val_ds = dataset_cls(base, val_indices, emb_cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    train_loader.d_txt = emb_cache.d_txt
    val_loader.d_txt = emb_cache.d_txt

    return train_loader, val_loader
