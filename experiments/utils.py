import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel




def atomic_torch_save(obj: Any, path: Path | str, **kwargs) -> None:
    path = Path(path)
    partial_path = path.with_suffix(path.suffix + ".partial")

    path.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists():
        partial_path.unlink()

    torch.save(obj, partial_path, **kwargs)

    os.replace(partial_path, path)




def is_embedding_file_complete(path: Path | str) -> bool:
    path = Path(path)
    partial_path = path.with_suffix(path.suffix + ".partial")

    if partial_path.exists():
        return False

    return path.exists()


_ALIAS = {
    "GPT2": "openai-community/gpt2",
    "GPT2M": "openai-community/gpt2-medium",
    "GPT2L": "openai-community/gpt2-large",
    "GPT2XL": "openai-community/gpt2-xl",
    "BERT": "google-bert/bert-base-uncased",
    "Llama": "meta-llama/Llama-3.1-8B",
    "DeepSeek": "deepseek-ai/deepseek-llm-7b-base",
    "Qwen3_E4B": "Qwen/Qwen3-Embedding-4B",
}


def get_context_window_size(
    llm_model_fusion: str,
    device: str | torch.device = "cpu",
) -> int:
    if isinstance(device, str):
        device = torch.device(device)

    tokenizer, llm_model = load_llm(llm_model_fusion, None, device)
    cfg = llm_model.config

    if hasattr(cfg, "n_positions"):
        size = cfg.n_positions
    elif hasattr(cfg, "max_position_embeddings"):
        size = cfg.max_position_embeddings
    elif hasattr(llm_model, "wpe"):
        size = llm_model.wpe.weight.size(0)

    del llm_model
    del tokenizer
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return size


def load_llm(
    llm_model_fusion: str,
    llm_layers_fusion: int | None = None,
    device: str | torch.device = "cpu",
    use_device_map: bool = False,
):
    model_id = _ALIAS.get(llm_model_fusion, llm_model_fusion)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_device_map:
        print(
            f"Detected {torch.cuda.device_count()} GPUs. Using model parallelism.")
        llm_model = AutoModel.from_pretrained(model_id, device_map="auto")
    else:
        print(f"Loading model on {device}.")
        llm_model = AutoModel.from_pretrained(model_id).to(device)

    if llm_layers_fusion is not None:
        if hasattr(llm_model, "encoder") and hasattr(llm_model.encoder, "layer"):
            llm_model.encoder.layer = nn.ModuleList(
                llm_model.encoder.layer[:llm_layers_fusion]
            )

    for param in llm_model.parameters():
        param.requires_grad = False

    llm_model.resize_token_embeddings(len(tokenizer))

    print(
        f"Loaded {llm_model_fusion} with "
        f"{'full' if llm_layers_fusion is None else llm_layers_fusion} layers for fusion."
    )
    return tokenizer, llm_model


def embed_notes(
    notes_text: list[list[str]],
    tokenizer,
    llm_model,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(notes_text)
    lengths = [len(x) for x in notes_text]
    N_max = max(lengths) if lengths else 0

    padded = [sample + [""] * (N_max - len(sample)) for sample in notes_text]

    flat_notes = [note for sample in padded for note in sample]

    if max_length is not None:
        enc = tokenizer(
            flat_notes,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        enc = tokenizer(
            flat_notes,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )
    L_tok = enc.input_ids.size(1)
    notes_input_ids = enc.input_ids.view(B, N_max, L_tok)
    attention_mask = enc.attention_mask.view(B, N_max, L_tok)

    note_mask = torch.tensor(
        [[True] * l + [False] * (N_max - l) for l in lengths],
        dtype=torch.bool,
        device=next(llm_model.parameters()).device,
    )

    flat_ids = notes_input_ids.view(B * N_max, L_tok)
    flat_attn = attention_mask.view(B * N_max, L_tok)

    device = next(llm_model.parameters()).device
    flat_ids = flat_ids.to(device)
    flat_attn = flat_attn.to(device)

    outputs = llm_model(input_ids=flat_ids, attention_mask=flat_attn)
    hidden_states = outputs.last_hidden_state

    mask_expanded = flat_attn.unsqueeze(-1)
    sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
    lengths_tok = mask_expanded.sum(dim=1).clamp(min=1)
    pooled = sum_hidden / lengths_tok

    hidden_size = pooled.size(-1)
    note_embeddings = pooled.view(B, N_max, hidden_size)

    return note_embeddings, note_mask


if __name__ == "__main__":
    llm_model_fusion = "GPT2"
    llm_layers_fusion = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, llm_model = load_llm(
        llm_model_fusion, llm_layers_fusion, device)

    notes_text = [
        [
            "This is my first note.",
            "I love machine learning.",
            "Let's test tokenization.",
        ],
        [],
        [
            "Here is another note.",
            "Data science is fun.",
            "PyTorch and transformers integration.",
            "Testing pad behavior.",
            "Final note in this batch.",
        ],
    ]

    embs, mask = embed_notes(notes_text, tokenizer, llm_model)
    print(embs.shape)
    print(mask)

    print(embs.device)
    print(mask.device)

    context_window_size = get_context_window_size(llm_model_fusion, device)
    print(f"Context window size: {context_window_size}")
