"""
Data utilities for attack experiments.
- Group Split: ensure same document doesn't appear in train/test
- Matched Test Set: balanced sampling across doc_type x length_bucket
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupShuffleSplit
import torch


def load_client_data(data_dir: str, client_name: str) -> List[Dict]:
    """Load all samples for a client."""
    data_dir = Path(data_dir)
    samples = []

    for jsonl_file in data_dir.glob(f"{client_name}*.jsonl"):
        # Extract doc_type from filename
        parts = jsonl_file.stem.split("_")
        if len(parts) > 3:
            doc_type = "_".join(parts[3:-1]) if parts[-1] == "annotated" else "_".join(parts[3:])
        else:
            doc_type = "unknown"

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    data['client'] = client_name
                    data['doc_type'] = doc_type
                    data['doc_id'] = f"{jsonl_file.stem}_{line_idx}"  # Unique doc ID for grouping
                    samples.append(data)
                except:
                    continue
    return samples


def get_length_bucket(text: str, buckets: List[int] = [500, 1000, 1500, 2000]) -> str:
    """Assign text to a length bucket."""
    length = len(text)
    for i, threshold in enumerate(buckets):
        if length < threshold:
            return f"len_{i}"
    return f"len_{len(buckets)}"


def prepare_attack_data(
    data_dir: str,
    clients: List[str],
    text_field: str = "rewritten_text",  # or "original_text" for raw
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Prepare attack data with Group Split.

    Args:
        data_dir: Directory containing client data
        clients: List of client names
        text_field: Which field to use as text ("rewritten_text" for sanitized, "original_text" for raw)
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        train_data, test_data, client_to_id mapping
    """
    all_samples = []
    client_to_id = {client: i for i, client in enumerate(sorted(clients))}

    for client in clients:
        samples = load_client_data(data_dir, client)
        for s in samples:
            text = s.get(text_field, s.get("original_text", ""))
            if not text:
                continue
            all_samples.append({
                'text': text,
                'client': s['client'],
                'client_id': client_to_id[s['client']],
                'doc_type': s['doc_type'],
                'doc_id': s['doc_id'],
                'length_bucket': get_length_bucket(text),
            })

    # Group split by doc_id
    groups = [s['doc_id'] for s in all_samples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(all_samples, groups=groups))

    train_data = [all_samples[i] for i in train_idx]
    test_data = [all_samples[i] for i in test_idx]

    print(f"[Data] Total: {len(all_samples)}, Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"[Data] Clients: {list(client_to_id.keys())}")

    return train_data, test_data, client_to_id


def create_matched_test_set(
    test_data: List[Dict],
    n_per_stratum: int = 10,
    random_state: int = 42
) -> List[Dict]:
    """
    Create a matched test set balanced across clients.
    Falls back to simpler balancing if strict stratification fails.

    Args:
        test_data: Original test data
        n_per_stratum: Number of samples per stratum
        random_state: Random seed

    Returns:
        Matched test data
    """
    np.random.seed(random_state)

    # Group by client
    by_client = defaultdict(list)
    for sample in test_data:
        by_client[sample['client']].append(sample)

    clients = list(by_client.keys())

    # Strategy 1: Try strict (doc_type x length_bucket) matching
    strata = defaultdict(list)
    for sample in test_data:
        key = (sample['client'], sample['doc_type'], sample['length_bucket'])
        strata[key].append(sample)

    doc_types = set(s['doc_type'] for s in test_data)
    length_buckets = set(s['length_bucket'] for s in test_data)

    matched_data = []

    for doc_type in doc_types:
        for length_bucket in length_buckets:
            min_count = float('inf')
            for client in clients:
                key = (client, doc_type, length_bucket)
                count = len(strata.get(key, []))
                if count > 0:
                    min_count = min(min_count, count)

            if min_count == 0 or min_count == float('inf'):
                continue

            n_sample = min(n_per_stratum, min_count)

            for client in clients:
                key = (client, doc_type, length_bucket)
                samples = strata.get(key, [])
                if len(samples) >= n_sample:
                    selected = np.random.choice(len(samples), n_sample, replace=False)
                    matched_data.extend([samples[i] for i in selected])

    # Strategy 2: If strict matching failed, do simple balanced sampling
    if len(matched_data) < len(clients) * 5:
        print("[Matched] Strict stratification failed, using balanced sampling")
        matched_data = []
        min_client_count = min(len(samples) for samples in by_client.values())
        n_per_client = min(min_client_count, n_per_stratum * 3)

        for client, samples in by_client.items():
            if len(samples) >= n_per_client:
                selected = np.random.choice(len(samples), n_per_client, replace=False)
                matched_data.extend([samples[i] for i in selected])
            else:
                matched_data.extend(samples)

    print(f"[Matched] Original test: {len(test_data)}, Matched test: {len(matched_data)}")

    # Print distribution
    client_counts = defaultdict(int)
    for s in matched_data:
        client_counts[s['client']] += 1
    print(f"[Matched] Per-client counts: {dict(client_counts)}")

    return matched_data


def prepare_embedding_data(
    model,
    tokenizer,
    data: List[Dict],
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract role and style embeddings for attack data.

    Returns:
        role_embeddings, style_embeddings, concat_embeddings, labels
    """
    model.eval()

    all_role = []
    all_style = []
    all_labels = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        texts = [s['text'] for s in batch]
        labels = [s['client_id'] for s in batch]

        # Tokenize
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = model.base.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"]
            )
            hidden = encoder_outputs.last_hidden_state

            # Get role and style projections
            role = model.role_proj(hidden)
            style = model.style_proj(hidden)

            # Mean pooling
            mask = enc["attention_mask"].unsqueeze(-1)
            role_mean = (role * mask).sum(dim=1) / mask.sum(dim=1)
            style_mean = (style * mask).sum(dim=1) / mask.sum(dim=1)

            all_role.append(role_mean.cpu().numpy())
            all_style.append(style_mean.cpu().numpy())
            all_labels.extend(labels)

    role_emb = np.vstack(all_role)
    style_emb = np.vstack(all_style)
    concat_emb = np.hstack([role_emb, style_emb])
    labels = np.array(all_labels)

    return role_emb, style_emb, concat_emb, labels


def load_prototypes(prototype_dir: str, clients: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load local prototypes for each client."""
    prototype_dir = Path(prototype_dir)
    local_dir = prototype_dir / "local"

    prototypes = {}
    for client in clients:
        proto_file = local_dir / f"{client}_protos.pt"
        if proto_file.exists():
            proto_data = torch.load(proto_file, map_location="cpu")
            prototypes[client] = {
                k: v.numpy() if torch.is_tensor(v) else v
                for k, v in proto_data.items()
            }

    return prototypes


def prepare_prototype_attack_data(
    prototypes: Dict[str, Dict[str, np.ndarray]],
    client_to_id: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare prototype data for E2-D attack.
    Each sample = concatenation of all entity type prototypes for a client.

    Returns:
        features (n_clients, total_proto_dim), labels (n_clients,)
    """
    # Get all entity types
    all_types = set()
    for client_protos in prototypes.values():
        all_types.update(client_protos.keys())
    all_types = sorted(all_types)

    features = []
    labels = []

    for client, protos in prototypes.items():
        if client not in client_to_id:
            continue

        # Concatenate all prototypes
        proto_vec = []
        for etype in all_types:
            if etype in protos:
                vec = protos[etype]
                if isinstance(vec, np.ndarray) and vec.ndim == 1:
                    proto_vec.append(vec)
                else:
                    # Skip invalid prototypes
                    proto_vec.append(np.zeros(128))  # Default dim
            else:
                proto_vec.append(np.zeros(128))

        if proto_vec:
            features.append(np.concatenate(proto_vec))
            labels.append(client_to_id[client])

    return np.array(features), np.array(labels)
