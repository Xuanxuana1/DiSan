#!/usr/bin/env python3
"""Generate placeholder-masked Enron samples with three PII detectors.

The expected input is the fixed sample JSONL produced by
`exp3b_eval_enron_stylometry_bert.py --save_samples`, with at least:
`original_text`, `sender`, and `label`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm


GLINER_LABELS = [
    "person",
    "name",
    "first name",
    "last name",
    "organization",
    "company",
    "location",
    "address",
    "street address",
    "city",
    "state",
    "zip code",
    "email",
    "email address",
    "phone number",
    "date",
    "time",
    "url",
    "ip address",
    "account number",
    "credit card",
    "ssn",
    "passport number",
    "driver license",
]

PLACEHOLDER_MAP = {
    "person": "NAME",
    "name": "NAME",
    "first name": "NAME",
    "last name": "NAME",
    "per": "NAME",
    "org": "ORG",
    "organization": "ORG",
    "company": "ORG",
    "location": "LOC",
    "address": "ADDRESS",
    "street address": "ADDRESS",
    "street_address": "ADDRESS",
    "city": "LOC",
    "state": "LOC",
    "zip": "ADDRESS",
    "zip code": "ADDRESS",
    "email": "EMAIL",
    "email address": "EMAIL",
    "phone": "PHONE",
    "phone number": "PHONE",
    "phone_number": "PHONE",
    "date": "DATE",
    "time": "DATE",
    "url": "URL",
    "ip": "IP",
    "ip address": "IP",
    "account": "ACCOUNT",
    "account number": "ACCOUNT",
    "bank account": "ACCOUNT",
    "credit card": "CARD",
    "credit card number": "CARD",
    "ssn": "SSN",
    "passport": "ID",
    "passport number": "ID",
    "driver license": "ID",
    "iban": "ACCOUNT",
    "swift bic code": "ACCOUNT",
}


def load_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if limit is not None and len(samples) >= limit:
                    break
    return samples


def normalize_label(label: str) -> str:
    label = str(label).strip()
    label = re.sub(r"^[BI]-", "", label)
    return label.replace("_", " ").lower()


def placeholder_for(label: str) -> str:
    normalized = normalize_label(label)
    placeholder = PLACEHOLDER_MAP.get(normalized, normalized.upper().replace(" ", "_"))
    return f"[{placeholder}]"


def clean_spans(spans: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for span in spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        label = str(span.get("label") or span.get("entity_group") or span.get("entity") or "PII")
        score = span.get("score")
        valid.append(
            {
                "start": start,
                "end": end,
                "label": normalize_label(label),
                "score": float(score) if score is not None else None,
                "text": text[start:end],
            }
        )

    valid.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    merged: list[dict[str, Any]] = []
    last_end = -1
    for span in valid:
        if span["start"] < last_end:
            continue
        merged.append(span)
        last_end = span["end"]
    return merged


def mask_text(text: str, spans: list[dict[str, Any]]) -> str:
    masked = text
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        masked = masked[: span["start"]] + placeholder_for(span["label"]) + masked[span["end"] :]
    return masked


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def span_token_count(text: str, spans: list[dict[str, Any]]) -> int:
    return sum(token_count(text[s["start"] : s["end"]]) for s in spans)


def resolve_model_path(detector: str, explicit_path: str | None) -> str:
    env_names = {
        "gliner": "GLINER_MODEL",
        "piiranha": "PIIRANHA_MODEL",
        "deberta": "DEBERTA_PII_MODEL",
    }
    model_path = explicit_path or os.environ.get(env_names[detector])
    if not model_path:
        raise ValueError(
            f"Missing model path for {detector}. Pass --{detector}_model or set {env_names[detector]}."
        )
    return model_path


def load_detector(name: str, model_path: str, device: str) -> Callable[[str], list[dict[str, Any]]]:
    if name == "gliner":
        from gliner import GLiNER

        model = GLiNER.from_pretrained(model_path)
        if device != "cpu":
            model.to(device)

        def detect(text: str) -> list[dict[str, Any]]:
            return model.predict_entities(text, GLINER_LABELS, flat_ner=True, threshold=0.3)

        return detect

    if name in {"piiranha", "deberta"}:
        import torch
        from transformers import pipeline

        pipeline_device = -1
        if device != "cpu" and torch.cuda.is_available():
            if device.startswith("cuda:"):
                pipeline_device = int(device.split(":", 1)[1])
            elif device == "cuda":
                pipeline_device = 0

        task = "ner" if name == "piiranha" else "token-classification"
        pipe = pipeline(task, model=model_path, device=pipeline_device)
        if name == "piiranha" and getattr(pipe, "tokenizer", None) is not None:
            pipe.tokenizer.model_max_length = min(getattr(pipe.tokenizer, "model_max_length", 256), 256)

        def detect(text: str) -> list[dict[str, Any]]:
            kwargs: dict[str, Any] = {"aggregation_strategy": "simple" if name == "piiranha" else "first"}
            if name == "piiranha":
                kwargs["stride"] = 50
            return pipe(text, **kwargs)

        return detect

    raise ValueError(f"Unknown detector: {name}")


def run_detector(
    samples: list[dict[str, Any]],
    detector: str,
    model_path: str,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    detect = load_detector(detector, model_path, device)
    output_path = output_dir / f"enron_placeholder_{detector}.jsonl"

    label_counts: Counter[str] = Counter()
    sender_counts: Counter[str] = Counter()
    total_spans = 0
    total_span_chars = 0
    total_span_tokens = 0
    total_text_chars = 0
    total_text_tokens = 0
    samples_with_spans = 0

    with output_path.open("w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"Masking with {detector}"):
            text = str(sample["original_text"])
            raw_spans = detect(text)
            spans = clean_spans(raw_spans, text)
            masked = mask_text(text, spans)

            n_span_tokens = span_token_count(text, spans)
            n_text_tokens = token_count(text)
            n_span_chars = sum(s["end"] - s["start"] for s in spans)

            total_spans += len(spans)
            total_span_chars += n_span_chars
            total_span_tokens += n_span_tokens
            total_text_chars += len(text)
            total_text_tokens += n_text_tokens
            samples_with_spans += int(bool(spans))
            sender_counts[str(sample.get("sender", "unknown"))] += len(spans)
            label_counts.update(s["label"] for s in spans)

            row = dict(sample)
            row["masked_text"] = masked
            row["pii_spans"] = spans
            row["mask_stats"] = {
                "span_count": len(spans),
                "masked_chars": n_span_chars,
                "masked_tokens": n_span_tokens,
                "text_chars": len(text),
                "text_tokens": n_text_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "detector": detector,
        "model_path": model_path,
        "output": str(output_path),
        "num_samples": len(samples),
        "samples_with_spans": samples_with_spans,
        "sample_coverage": samples_with_spans / len(samples) if samples else 0.0,
        "total_spans": total_spans,
        "avg_spans_per_sample": total_spans / len(samples) if samples else 0.0,
        "total_masked_chars": total_span_chars,
        "total_text_chars": total_text_chars,
        "char_mask_rate": total_span_chars / total_text_chars if total_text_chars else 0.0,
        "total_masked_tokens": total_span_tokens,
        "total_text_tokens": total_text_tokens,
        "token_mask_rate": total_span_tokens / total_text_tokens if total_text_tokens else 0.0,
        "label_counts": dict(label_counts.most_common()),
        "span_counts_by_sender": dict(sender_counts),
    }
    summary_path = output_dir / f"enron_placeholder_{detector}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask fixed Enron samples with PII placeholders.")
    parser.add_argument("--input", required=True, help="Fixed Enron sample JSONL.")
    parser.add_argument("--output_dir", default="./outputs/placeholder_masking")
    parser.add_argument("--detectors", nargs="+", default=["gliner", "piiranha", "deberta"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gliner_model")
    parser.add_argument("--piiranha_model")
    parser.add_argument("--deberta_model")
    args = parser.parse_args()

    supported = {"gliner", "piiranha", "deberta"}
    for detector in args.detectors:
        if detector not in supported:
            raise ValueError(f"Unsupported detector {detector}; choose from {sorted(supported)}")

    samples = load_samples(Path(args.input), limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_paths = {
        "gliner": args.gliner_model,
        "piiranha": args.piiranha_model,
        "deberta": args.deberta_model,
    }

    summaries = []
    for detector in args.detectors:
        model_path = resolve_model_path(detector, explicit_paths[detector])
        summaries.append(run_detector(samples, detector, model_path, output_dir, args.device))

    combined_path = output_dir / "enron_placeholder_masking_summary.json"
    combined_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary")
    print("detector\tsamples_with_spans\ttotal_spans\tavg_spans\tmasked_tokens\ttoken_rate")
    for summary in summaries:
        print(
            f"{summary['detector']}\t{summary['samples_with_spans']}/{summary['num_samples']}\t"
            f"{summary['total_spans']}\t{summary['avg_spans_per_sample']:.3f}\t"
            f"{summary['total_masked_tokens']}\t{summary['token_mask_rate']:.4%}"
        )
    print(f"\nSaved combined summary to {combined_path}")


if __name__ == "__main__":
    main()
