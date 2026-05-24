"""
EXP-2: Representation Probe (LoRA-compatible version)

Tests whether embeddings leak client identity:
- E2-A: Role embedding (should NOT leak client - domain agnostic)
- E2-B: Style embedding (expected to leak client - client specific)
- E2-C: Concat [role; style] (check combined leakage)
- E2-D: Prototype side-channel (can prototypes reveal client?)

Probe models:
- P1: Logistic Regression (strong, interpretable)
- P2: MLP (nonlinear probe)

Modified for fed_lightweight LoRA models.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GroupShuffleSplit

from .data_utils import (
    prepare_attack_data,
    create_matched_test_set,
    prepare_embedding_data,
    load_prototypes,
    prepare_prototype_attack_data
)
from .attackers import EmbeddingProbe, AttackMetrics


def run_exp2_probe(
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    clients: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cuda",
    max_samples: int = 500,  # Limit samples per client for speed
    model_path: Optional[str] = None,  # Direct path to model checkpoint
    use_lora: bool = True,  # Whether the model uses LoRA
) -> Dict:
    """
    Run EXP-2: Representation Probe experiment.

    Args:
        data_dir: Directory with client data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        clients: List of client names
        test_size: Test set fraction
        random_state: Random seed
        device: Device for model inference
        max_samples: Max samples per client
        model_path: Direct path to model checkpoint
        use_lora: Whether the model uses LoRA (default: True for fed_lightweight)

    Returns:
        Dictionary with all results
    """
    # Import from fed_lightweight for LoRA support
    from fed_lightweight.config import ModelConfig
    from fed_lightweight.model import FedDisPModel
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(clients)
    results = {
        'exp': 'EXP-2',
        'num_clients': num_classes,
        'clients': clients,
        'probes': {}
    }

    # Load model and tokenizer
    print("\n[Loading model and tokenizer]")
    model_config = ModelConfig()
    model_config.use_lora = use_lora
    tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_path)

    # Determine model path
    if model_path and Path(model_path).exists():
        # Use directly specified model path
        actual_model_path = Path(model_path)
    else:
        # We'll test with one client's model first (or a shared global model if available)
        # For fair comparison, use the first client's model
        actual_model_path = Path(checkpoint_dir) / "local" / clients[0] / "best.pt"
        if not actual_model_path.exists():
            print(f"Warning: Model not found at {actual_model_path}, trying other clients...")
            for client in clients:
                actual_model_path = Path(checkpoint_dir) / "local" / client / "best.pt"
                if actual_model_path.exists():
                    break

    # Create model with LoRA if needed
    model = FedDisPModel(model_config, num_clients=len(clients))

    # Load checkpoint - for LoRA models, we only load trainable params
    state_dict = torch.load(actual_model_path, map_location="cpu")
    if use_lora:
        model.load_trainable_state_dict(state_dict, strict=False)
        print(f"  Loaded LoRA model from {actual_model_path}")
    else:
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded full model from {actual_model_path}")

    model.to(device)
    model.eval()


    # Prepare data
    print("\n[Preparing data]")
    train_data, test_data, client_to_id = prepare_attack_data(
        data_dir, clients,
        text_field="original_text",  # Use original text for embedding extraction
        test_size=test_size,
        random_state=random_state
    )

    # Limit samples for speed
    if max_samples and len(train_data) > max_samples * num_classes:
        np.random.seed(random_state)
        train_data = sample_balanced(train_data, max_samples, num_classes)
        print(f"  Limited train to {len(train_data)} samples")

    if max_samples and len(test_data) > max_samples * num_classes // 2:
        np.random.seed(random_state)
        test_data = sample_balanced(test_data, max_samples // 2, num_classes)
        print(f"  Limited test to {len(test_data)} samples")

    matched_test = create_matched_test_set(test_data, random_state=random_state)

    # Extract embeddings
    print("\n[Extracting embeddings]")
    train_role, train_style, _, train_labels = prepare_embedding_data(
        model, tokenizer, train_data, device=device
    )
    test_role, test_style, _, test_labels = prepare_embedding_data(
        model, tokenizer, test_data, device=device
    )

    if matched_test:
        matched_role, matched_style, _, matched_labels = prepare_embedding_data(
            model, tokenizer, matched_test, device=device
        )
    else:
        matched_role = matched_style = matched_labels = None

    print(f"  Train embeddings: role={train_role.shape}")
    print(f"  Test embeddings: role={test_role.shape}")

    # ========== E2-A: Role embedding probe (SVM only) ==========
    print("\n" + "="*60)
    print("E2-A: Role Embedding Probe with SVM")
    print("(Testing if role embeddings leak client identity)")
    print("="*60)

    results['probes']['E2A_role'] = run_probe_on_embeddings(
        train_role, train_labels,
        test_role, test_labels,
        matched_role, matched_labels,
        num_classes
    )

    # ========== Summary ==========
    print("\n" + "="*60)
    print("SUMMARY: Role Embedding Probe (SVM) Results")
    print("="*60)
    print_probe_summary(results)

    # Save results
    results_file = output_dir / "exp2_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"\nResults saved to {results_file}")

    return results


def sample_balanced(data: List[Dict], n_per_class: int, num_classes: int) -> List[Dict]:
    """Sample balanced data from each class."""
    from collections import defaultdict

    by_class = defaultdict(list)
    for sample in data:
        by_class[sample['client_id']].append(sample)

    sampled = []
    for class_id in range(num_classes):
        class_samples = by_class[class_id]
        if len(class_samples) > n_per_class:
            indices = np.random.choice(len(class_samples), n_per_class, replace=False)
            sampled.extend([class_samples[i] for i in indices])
        else:
            sampled.extend(class_samples)

    return sampled


def run_probe_on_embeddings(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    matched_emb: Optional[np.ndarray],
    matched_labels: Optional[np.ndarray],
    num_classes: int,
    probe_types: List[str] = None
) -> Dict:
    """Run multiple probes on embeddings to test client identity leakage."""

    if probe_types is None:
        probe_types = ['svm']  # Only SVM for Role embedding attack

    results = {}

    for probe_type in probe_types:
        print(f"\n[{probe_type.upper()}] {probe_type} Probe")
        probe = EmbeddingProbe(probe_type=probe_type, num_classes=num_classes)
        probe.fit(train_emb, train_labels)

        metrics_full = probe.evaluate(test_emb, test_labels)
        print(f"  Full Test:    Acc={metrics_full.accuracy:.4f}, F1={metrics_full.macro_f1:.4f}")

        if matched_emb is not None and len(matched_emb) > 0:
            metrics_matched = probe.evaluate(matched_emb, matched_labels)
            print(f"  Matched Test: Acc={metrics_matched.accuracy:.4f}, F1={metrics_matched.macro_f1:.4f}")
        else:
            metrics_matched = None

        results[probe_type.upper()] = {
            'full_test': metrics_full.to_dict(),
            'matched_test': metrics_matched.to_dict() if metrics_matched else None
        }

    return results


def run_prototype_probe(
    prototype_dir: Path,
    clients: List[str],
    client_to_id: Dict[str, int],
    num_classes: int
) -> Dict:
    """
    E2-D: Test if prototypes leak client identity.

    Since prototypes are per-client, this is a trivial classification.
    More meaningful: test if global prototype reveals entity type semantics.
    """

    results = {}

    # Load local prototypes
    prototypes = load_prototypes(str(prototype_dir), clients)

    if not prototypes:
        print("  No prototypes found, skipping E2-D")
        return {'error': 'No prototypes found'}

    print(f"  Loaded prototypes for {len(prototypes)} clients")

    # Prepare prototype features
    features, labels = prepare_prototype_attack_data(prototypes, client_to_id)

    if len(features) < 2:
        print("  Not enough prototype data for probe")
        return {'error': 'Insufficient data'}

    print(f"  Prototype features shape: {features.shape}")

    # Note: With only 7 clients and 7 samples, this is a degenerate case
    # Report this as a side-channel risk indicator rather than a proper probe
    results['note'] = (
        "With N_clients prototypes for N_clients classes, "
        "this is a degenerate classification problem. "
        "The relevant question is: do prototypes need to be shared, "
        "and if so, what information do they leak?"
    )

    # Compute prototype similarity matrix as a proxy for leakage risk
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(features)
    avg_within_client_sim = np.mean(np.diag(sim_matrix))
    avg_between_client_sim = (np.sum(sim_matrix) - np.trace(sim_matrix)) / (len(features)**2 - len(features))

    results['prototype_analysis'] = {
        'num_clients_with_protos': len(features),
        'avg_within_client_similarity': float(avg_within_client_sim),
        'avg_between_client_similarity': float(avg_between_client_sim),
        'separability_ratio': float(avg_within_client_sim / (avg_between_client_sim + 1e-8))
    }

    print(f"  Within-client sim: {avg_within_client_sim:.4f}")
    print(f"  Between-client sim: {avg_between_client_sim:.4f}")
    print(f"  Separability ratio: {results['prototype_analysis']['separability_ratio']:.4f}")

    return results


def print_probe_summary(results: Dict):
    """Print summary table for probe results."""

    random_baseline = 1.0 / results['num_clients']

    print("\n" + "="*80)
    print("ROLE EMBEDDING PROBE (SVM) RESULTS")
    print("="*80)

    for probe_name, probe_results in results['probes'].items():
        if 'error' in probe_results:
            continue

        print(f"\n{probe_name}:")
        print("-"*80)
        print(f"{'Classifier':<12} {'Full Acc':<12} {'Full F1':<12} {'Matched F1':<12} {'Status':<10}")
        print("-"*80)

        for classifier_name, classifier_results in probe_results.items():
            if 'full_test' not in classifier_results:
                continue

            full_acc = classifier_results['full_test']['accuracy']
            full_f1 = classifier_results['full_test']['macro_f1']
            matched_f1 = classifier_results.get('matched_test', {}).get('macro_f1', 'N/A')

            if isinstance(matched_f1, float):
                matched_f1_str = f"{matched_f1:.4f}"
            else:
                matched_f1_str = str(matched_f1)

            status = '✅' if full_f1 <= random_baseline + 0.15 else '❌'
            print(f"{classifier_name:<12} {full_acc:.4f}       {full_f1:.4f}       {matched_f1_str:<12} {status:<10}")

    print("-"*80)

    # Analysis
    print("\n[Privacy Assessment]")
    print(f"  Random baseline: Acc={random_baseline:.4f}, F1={random_baseline:.4f}")

    role_results = results['probes'].get('E2A_role', {})

    # Find best classifier for role
    best_role_f1 = 0
    best_role_acc = 0
    for clf_name, clf_results in role_results.items():
        if 'full_test' in clf_results:
            f1 = clf_results['full_test'].get('macro_f1', 0)
            acc = clf_results['full_test'].get('accuracy', 0)
            if f1 > best_role_f1:
                best_role_f1 = f1
                best_role_acc = acc

    print(f"  Role embedding (SVM): Acc={best_role_acc:.4f}, F1={best_role_f1:.4f}")

    if best_role_f1 <= random_baseline + 0.15:
        print("  ✅ Role embedding is domain-agnostic (low client leakage)")
    else:
        print(f"  ⚠️ Role embedding leaks client info (above random by {best_role_f1 - random_baseline:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EXP-2 Representation Probe (LoRA-compatible)")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoint_dir", default="../checkpoints/fed_lora")
    parser.add_argument("--output_dir", default="../attack_results/exp2_probe_lora")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Direct path to model checkpoint (overrides checkpoint_dir search)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Whether the model uses LoRA (default: True)")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora",
                        help="Disable LoRA mode (for full model checkpoints)")
    args = parser.parse_args()

    clients = [
        "Client_1_CorporateBank",
        "Client_2_AssetManager",
        "Client_3_FinTechPay",
        "Client_4_CorpGroup",
        "Client_5_MarketForecaster",
        "Client_6_ComplianceConsult",
        "Client_7_SupplierCo",
    ]

    run_exp2_probe(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        clients=clients,
        device=args.device,
        max_samples=args.max_samples,
        model_path=args.model_path,
        use_lora=args.use_lora,
    )
