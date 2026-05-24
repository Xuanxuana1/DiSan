import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

try:
    from .config import ModelConfig, DataConfig
except ImportError:
    from config import ModelConfig, DataConfig


class JsonlSeq2SeqDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        tokenizer: AutoTokenizer,
        model_config: ModelConfig,
        src_field: str,
        tgt_field: str,
    ):
        self.examples: List[Tuple[str, str, Optional[List[Dict[str, Any]]]]] = []
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if src_field in obj and tgt_field in obj:
                        spans_raw = obj.get("pii_spans")
                        spans: Optional[List[Dict[str, Any]]] = None
                        if isinstance(spans_raw, str):
                            try:
                                spans = json.loads(spans_raw)
                            except json.JSONDecodeError:
                                spans = None
                        elif isinstance(spans_raw, list):
                            spans = spans_raw  # type: ignore[assignment]
                        self.examples.append((obj[src_field], obj[tgt_field], spans))
        self.tokenizer = tokenizer
        self.model_config = model_config

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt, spans = self.examples[idx]
        return {"source": src, "target": tgt, "spans": spans}


@dataclass
class ClientDataLoaders:
    train: DataLoader
    dev: Optional[DataLoader]
    test: Optional[DataLoader]


def _collect_files(root: str, client: str, split: str) -> List[str]:
    # Since data is flat (no client subdirs), collect all files for this client
    if not os.path.isdir(root):
        return []
    # Extract client name from filename pattern: Client_X_Name_..._annotated.jsonl
    return [os.path.join(root, f) for f in os.listdir(root)
            if f.endswith(".jsonl") and f.startswith(f"{client}_")]


def build_dataloaders(
    data_config: DataConfig,
    model_config: ModelConfig,
    batch_size: int,
    split: str,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Dict[str, ClientDataLoaders]:
    tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_path, use_fast=True)
    client_loaders: Dict[str, ClientDataLoaders] = {}
    for client in data_config.clients:
        files = _collect_files(data_config.data_root, client, split)
        if not files:
            continue
        dataset = JsonlSeq2SeqDataset(
            files, tokenizer, model_config, data_config.src_field, data_config.tgt_field
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_pad(tokenizer),
        )
        client_loaders[client] = ClientDataLoaders(train=loader, dev=None, test=None)
    return client_loaders


def _collate_pad(tokenizer: AutoTokenizer):
    return _collate_pad_with_gliner(tokenizer, None)


def _collate_pad_with_gliner(
    tokenizer: AutoTokenizer,
    gliner_model: Optional[Any],
    max_source_length: int = 1536,
    max_target_length: int = 1536,
    vocab_size: Optional[int] = None,
):
    def detect_with_gliner(text: str) -> List[Dict[str, Any]]:
        if gliner_model is None:
            return []
        labels = [
            "name",
            "first name",
            "last name",
            "email address",
            "phone number",
            "location address",
            "street_address",
            "location city",
            "location state",
            "location zip",
            "credit card",
            "credit card number",
            "account number",
            "bank account",
            "ssn",
            "dob",
            "date",
            "company",
            "organization",
            "user_name",
            "username",
            "ip address",
            "url",
            "passport number",
            "driver license",
        ]
        try:
            entities = gliner_model.predict_entities(text, labels, threshold=0.3)
            spans = []
            for ent in entities:
                s = ent.get("start", 0)
                e = ent.get("end", 0)
                lab = ent.get("label", "")
                if isinstance(s, int) and isinstance(e, int) and e > s >= 0 and e <= len(text):
                    spans.append({"start": s, "end": e, "label": lab})
            return spans
        except Exception:
            return []

    from .prototypes import LABEL_TO_MACRO, MACRO_TO_INDEX

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        TASK_PREFIX = "deidentify:"
        raw_sources = [item["source"] for item in batch]
        sources = [TASK_PREFIX + src for src in raw_sources]
        targets = [item["target"] for item in batch]
        spans_list = [item.get("spans") for item in batch]

        enc = tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        dec = tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")

        labels = dec["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        effective_vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size
        vocab_masks: List[torch.Tensor] = []
        entity_type_ids_list: List[torch.Tensor] = []
        for ex_idx, (raw_text, spans) in enumerate(zip(raw_sources, spans_list)):
            merged = []
            if spans:
                merged.extend(spans)
            gliner_spans = detect_with_gliner(raw_text)
            if gliner_spans:
                merged.extend(gliner_spans)
            seen = set()
            mask = torch.zeros(effective_vocab_size, dtype=torch.float32)
            for span in merged:
                try:
                    start = int(span.get("start", 0))
                    end = int(span.get("end", 0))
                except (TypeError, ValueError):
                    continue
                if end <= start or start < 0 or end > len(raw_text):
                    continue
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                span_text = raw_text[start:end]
                if not span_text:
                    continue
                tokens = tokenizer.tokenize(span_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                for i in ids:
                    if isinstance(i, int) and 0 <= i < effective_vocab_size:
                        mask[i] = 1.0
            vocab_masks.append(mask)
            # Build token-level entity type ids based ONLY on gold spans (not GLiNER),
            # mapped to macro-types.
            seq_len = enc["input_ids"].size(1)
            ent_ids = torch.full((seq_len,), -1, dtype=torch.long)
            example_offsets = offsets[ex_idx]  # [T, 2]
            prefix_len = len(TASK_PREFIX)
            gold_spans = spans or []
            for span in gold_spans:
                try:
                    start = int(span.get("start", 0))
                    end = int(span.get("end", 0))
                    lab = str(span.get("label", ""))
                except (TypeError, ValueError):
                    continue
                if end <= start or start < 0 or end > len(raw_text):
                    continue
                macro = LABEL_TO_MACRO.get(lab)
                if macro is None:
                    continue
                type_idx = MACRO_TO_INDEX[macro]
                span_start = start + prefix_len
                span_end = end + prefix_len
                for t, off in enumerate(example_offsets.tolist()):
                    tok_s, tok_e = off
                    if tok_e <= tok_s:
                        continue
                    # simple interval overlap
                    if tok_e > span_start and tok_s < span_end:
                        ent_ids[t] = type_idx
            entity_type_ids_list.append(ent_ids)

        batch_out: Dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "decoder_attention_mask": dec["attention_mask"],
            "vocab_mask": torch.stack(vocab_masks, dim=0),
            "raw_sources": raw_sources,
            "raw_targets": targets,
            "entity_type_ids": torch.stack(entity_type_ids_list, dim=0),
        }
        return batch_out

    return collate

