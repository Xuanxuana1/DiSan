#!/usr/bin/env python3
"""EXP-3b Enron stylometry attribution probe.

This script evaluates source attribution on raw Enron emails, optional
placeholder-masked JSONL files, and optional DiSan-sanitized outputs using the
same TF-IDF and BERT probe settings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fed_lightweight.config import ModelConfig
from fed_lightweight.model import FedDisPModel


def extract_sender_from_path(file_path: str) -> str:
    """Extract sender directory from an Enron CSV file path."""
    return str(file_path).split("/")[0] if file_path else "unknown"


def parse_email_body(message: str) -> str:
    """Extract email body and remove common forwarded-message delimiters."""
    lines = str(message).split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:])
    body = re.sub(r"-+\s*Forwarded by.*?-+", "", body, flags=re.DOTALL)
    body = re.sub(r"-+\s*Original Message\s*-+", "", body, flags=re.DOTALL)
    return body.strip()


def load_enron_records(
    data_path: str,
    top_k_senders: int = 7,
    samples_per_sender: int = 500,
    min_body_length: int = 100,
    max_body_length: int = 2000,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load a fixed balanced Enron sample set from Kaggle `emails.csv`."""
    csv_path = Path(data_path) / "emails.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Enron data not found: {csv_path}")

    print(f"Loading Enron data from {csv_path}")
    df = pd.read_csv(csv_path)
    df["sender"] = df["file"].apply(extract_sender_from_path)
    df["body"] = df["message"].apply(parse_email_body)
    df = df[df["body"].str.len().between(min_body_length, max_body_length)]

    sender_counts = df["sender"].value_counts()
    top_senders = sender_counts.head(top_k_senders).index.tolist()
    sender_to_idx = {sender: i for i, sender in enumerate(top_senders)}

    records: list[dict[str, Any]] = []
    for sender in top_senders:
        sender_df = df[df["sender"] == sender]
        sampled = sender_df.sample(
            n=min(samples_per_sender, len(sender_df)),
            random_state=random_state,
        )
        for _, row in sampled.iterrows():
            records.append(
                {
                    "id": row["file"],
                    "sender": sender,
                    "label": sender_to_idx[sender],
                    "original_text": row["body"],
                }
            )

    print(f"Top senders: {top_senders}")
    print(f"Total samples: {len(records)}")
    return records, top_senders


def load_records_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_records_jsonl(records: list[dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def records_to_xy(
    records: list[dict[str, Any]],
    text_field: str,
    label_field: str = "label",
) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    sender_to_idx: dict[str, int] = {}

    for record in records:
        text = str(record.get(text_field, "")).strip()
        if not text:
            continue
        if label_field in record:
            label = int(record[label_field])
        else:
            sender = str(record.get("sender", "unknown"))
            if sender not in sender_to_idx:
                sender_to_idx[sender] = len(sender_to_idx)
            label = sender_to_idx[sender]
        texts.append(text)
        labels.append(label)

    return texts, labels


def evaluate_tfidf_probe(
    texts: list[str],
    labels: list[int],
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate stylometry using TF-IDF features and standard classifiers."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifiers = {
        "LR": LogisticRegression(max_iter=1000, random_state=random_state),
        "LinearSVC": LinearSVC(max_iter=1000, random_state=random_state),
        "RF": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

    results: dict[str, Any] = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }

    best_clf = max(results, key=lambda k: results[k]["f1_macro"])
    results["best"] = {"classifier": best_clf, **results[best_clf]}
    return results


def evaluate_bert_probe(
    texts: list[str],
    labels: list[int],
    model_path: str,
    batch_size: int = 32,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate stylometry using sentence embeddings and shallow classifiers."""
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import KNeighborsClassifier

    print(f"  Loading sentence encoder: {model_path}")
    encoder = SentenceTransformer(model_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    X_train_emb = encoder.encode(X_train, batch_size=batch_size, show_progress_bar=True)
    X_test_emb = encoder.encode(X_test, batch_size=batch_size, show_progress_bar=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    classifiers = {
        "SVM-RBF": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "LR": LogisticRegression(max_iter=1000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    }

    results: dict[str, Any] = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }

    best_clf = max(results, key=lambda k: results[k]["f1_macro"])
    results["best"] = {"classifier": best_clf, **results[best_clf]}
    return results


def load_disan_model(
    checkpoint_path: str,
    base_model_path: str,
    device: torch.device,
    num_clients: int,
) -> tuple[FedDisPModel, Any]:
    """Load a trained DiSan checkpoint for optional sanitized-text evaluation."""
    model_cfg = ModelConfig(pretrained_model_path=base_model_path)
    model_cfg.residual_mode = "projected"
    model = FedDisPModel(model_cfg, num_clients=num_clients).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.pretrained_model_path, use_fast=True)
    return model, tokenizer


def sanitize_text(
    text: str,
    model: FedDisPModel,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 1536,
) -> str:
    inputs = tokenizer(
        "deidentify:" + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def sanitize_batch(
    texts: list[str],
    model: FedDisPModel,
    tokenizer: Any,
    device: torch.device,
    batch_size: int = 8,
) -> list[str]:
    sanitized_texts: list[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sanitizing"):
        for text in texts[i : i + batch_size]:
            try:
                sanitized_texts.append(sanitize_text(text, model, tokenizer, device))
            except Exception as exc:
                print(f"Warning: sanitization failed for one sample: {exc}")
                sanitized_texts.append(text)
    return sanitized_texts


def evaluate_setting(
    name: str,
    texts: list[str],
    labels: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("\n[TF-IDF Probe]")
    tfidf = evaluate_tfidf_probe(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"  Best: {tfidf['best']['classifier']}, F1={tfidf['best']['f1_macro']:.4f}")

    bert = None
    if not args.skip_bert:
        print("\n[BERT Probe]")
        bert = evaluate_bert_probe(
            texts,
            labels,
            model_path=args.bert_model_path,
            batch_size=args.bert_batch_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"  Best: {bert['best']['classifier']}, F1={bert['best']['f1_macro']:.4f}")

    result = {"tfidf": tfidf}
    if bert is not None:
        result["bert"] = bert
    return result


def parse_named_paths(values: list[str]) -> dict[str, str]:
    named_paths: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        named_paths[name.strip()] = path.strip()
    return named_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-3b Enron stylometry attribution probe.")
    parser.add_argument("--enron_path", default=str(REPO_ROOT / "data" / "enron_data"))
    parser.add_argument("--samples_file", help="Load a fixed Enron sample JSONL instead of sampling emails.csv.")
    parser.add_argument("--save_samples", help="Save the sampled raw records to this JSONL path.")
    parser.add_argument("--masked_jsonl", nargs="*", default=[], help="Optional masked inputs as NAME=PATH.")
    parser.add_argument("--masked_text_field", default="masked_text")
    parser.add_argument("--checkpoint", help="Optional DiSan checkpoint for sanitized-text evaluation.")
    parser.add_argument("--base_model_path", default=os.environ.get("DISAN_BASE_MODEL", str(REPO_ROOT / "long-t5-tglobal-base")))
    parser.add_argument("--num_senders", type=int, default=7)
    parser.add_argument("--samples_per_sender", type=int, default=500)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs" / "enron_stylometry"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bert_model_path", default=os.environ.get("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--bert_batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--skip_bert", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.samples_file:
        records = load_records_jsonl(args.samples_file)
        sender_names = sorted({str(r.get("sender", "unknown")) for r in records})
    else:
        records, sender_names = load_enron_records(
            args.enron_path,
            top_k_senders=args.num_senders,
            samples_per_sender=args.samples_per_sender,
            random_state=args.random_state,
        )

    if args.save_samples:
        save_records_jsonl(records, args.save_samples)
        print(f"Saved fixed sample set to {args.save_samples}")

    texts, labels = records_to_xy(records, text_field="original_text")
    random_baseline = 1.0 / len(set(labels))

    results: dict[str, Any] = {
        "config": vars(args),
        "sender_names": sender_names,
        "random_baseline": random_baseline,
        "settings": {},
    }

    results["settings"]["raw"] = evaluate_setting("RAW EMAIL STYLOMETRY", texts, labels, args)

    for name, path in parse_named_paths(args.masked_jsonl).items():
        masked_records = load_records_jsonl(path)
        masked_texts, masked_labels = records_to_xy(masked_records, text_field=args.masked_text_field)
        if len(masked_texts) != len(texts):
            print(f"Warning: {name} has {len(masked_texts)} samples; raw has {len(texts)} samples.")
        results["settings"][f"placeholder_{name}"] = evaluate_setting(
            f"PLACEHOLDER MASKING: {name}",
            masked_texts,
            masked_labels,
            args,
        )

    if args.checkpoint:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_disan_model(
            args.checkpoint,
            base_model_path=args.base_model_path,
            device=device,
            num_clients=len(set(labels)),
        )
        sanitized_texts = sanitize_batch(texts, model, tokenizer, device)
        results["settings"]["disan_sanitized"] = evaluate_setting(
            "DiSan-SANITIZED EMAIL STYLOMETRY",
            sanitized_texts,
            labels,
            args,
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for setting, setting_results in results["settings"].items():
        tfidf_best = setting_results["tfidf"]["best"]["f1_macro"]
        if "bert" in setting_results:
            bert_best = setting_results["bert"]["best"]["f1_macro"]
            print(f"{setting:<24} TF-IDF F1={tfidf_best:.4f}  BERT F1={bert_best:.4f}")
        else:
            print(f"{setting:<24} TF-IDF F1={tfidf_best:.4f}")
    print(f"Random baseline: {random_baseline:.4f}")

    output_path = output_dir / "exp3b_enron_stylometry_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
