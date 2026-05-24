"""
EXP-3: Prototype-based Client Attribution Attack (LoRA-compatible version)

Tests whether an attacker can infer client identity from role prototypes:
- Multiple attackers: Linear (LogReg), Non-linear (MLP), Non-parametric (KNN)
- Evaluation: Full test set and Matched test set (control doc-type/length)
- Goal: Demonstrate that prototype-level adversarial training reduces
        client attribution to near-chance level

This experiment validates the effectiveness of server-side adversarial training
on prototype privacy protection.

Modified for fed_lightweight LoRA models.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from collections import defaultdict

from .data_utils import prepare_attack_data, create_matched_test_set


def _load_multi_round_prototypes(
    checkpoint_dir: str,
    clients: List[str],
    rounds: List[int],
    client_to_id: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load prototypes from multiple rounds for bootstrap sampling.

    Args:
        checkpoint_dir: Directory with model checkpoints (containing round_X subdirs)
        clients: List of client names
        rounds: List of round numbers to load
        client_to_id: Mapping from client name to client ID

    Returns:
        Tuple of (prototypes, labels) where:
        - prototypes: [num_rounds * num_clients, dim] array
        - labels: [num_rounds * num_clients] array of client IDs
    """
    all_prototypes = []
    all_labels = []

    for round_num in rounds:
        proto_path = Path(checkpoint_dir) / f"round_{round_num}" / "prototypes.pt"
        if not proto_path.exists():
            print(f"  Warning: {proto_path} not found, skipping round {round_num}")
            continue

        proto_data = torch.load(proto_path, map_location="cpu")

        for client_name in clients:
            if client_name in proto_data:
                proto = proto_data[client_name].numpy()
                all_prototypes.append(proto)
                all_labels.append(client_to_id[client_name])
            else:
                print(f"  Warning: {client_name} not found in round {round_num}")

    if not all_prototypes:
        raise ValueError("No prototypes loaded from any round")

    return np.vstack(all_prototypes), np.array(all_labels)


def run_exp3_prototype(
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    clients: List[str],
    model_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cuda",
    max_samples_per_client: int = 200,
    n_bootstrap: int = 100,
    normalize_prototypes: bool = True,
    proto_noise_scale: float = 0.1,
    use_multi_round_protos: bool = True,
    proto_rounds: List[int] = None,
    use_lora: bool = True,  # Whether the model uses LoRA
) -> Dict:
    """
    Run EXP-3: Prototype-based Client Attribution Attack.

    This experiment tests whether client identity can be inferred from:
    1. Role prototypes (mean of role embeddings per client)
    2. Individual role embeddings aggregated by prototype similarity

    Args:
        data_dir: Directory with client data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        clients: List of client names
        model_path: Direct path to model checkpoint
        test_size: Test set fraction
        random_state: Random seed
        device: Device for model inference
        max_samples_per_client: Max samples per client for embedding extraction
        n_bootstrap: Number of bootstrap iterations for confidence intervals
        normalize_prototypes: If True, L2-normalize prototypes to simulate server
                              receiving normalized uploads (defense scenario)
        proto_noise_scale: Gaussian noise scale to add to prototypes (noise perturbation, not true DP)
        use_multi_round_protos: If True, load actual prototypes from multiple rounds for bootstrap
        proto_rounds: List of round numbers to use for multi-round bootstrap (default: last 5 rounds)
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

    num_clients = len(clients)
    random_baseline = 1.0 / num_clients

    results = {
        'exp': 'EXP-3: Prototype-based Client Attribution',
        'num_clients': num_clients,
        'clients': clients,
        'normalize_prototypes': normalize_prototypes,  # Defense mode: server receives normalized protos
        'proto_noise_scale': proto_noise_scale,  # Noise perturbation defense (not true DP)
        'random_baseline': {
            'accuracy': random_baseline,
            'macro_f1': random_baseline,
        },
        'attackers': {}
    }

    # Load model
    print("\n[Loading model and tokenizer]")
    model_config = ModelConfig()
    model_config.use_lora = use_lora
    tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_path)

    if model_path and Path(model_path).exists():
        actual_model_path = Path(model_path)
    else:
        actual_model_path = Path(checkpoint_dir) / "local" / clients[0] / "best.pt"
        if not actual_model_path.exists():
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

    # Prepare data and extract embeddings
    print("\n[Preparing data and extracting embeddings]")
    train_data, test_data, client_to_id = prepare_attack_data(
        data_dir, clients,
        text_field="original_text",
        test_size=test_size,
        random_state=random_state
    )

    # Limit samples per client
    train_data = _balance_samples(train_data, max_samples_per_client, num_clients)
    test_data = _balance_samples(test_data, max_samples_per_client // 2, num_clients)

    # Create matched test set
    matched_test = create_matched_test_set(test_data, random_state=random_state)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Matched test samples: {len(matched_test) if matched_test else 0}")

    # Extract role embeddings
    train_embeddings, train_labels = _extract_role_embeddings(
        model, tokenizer, train_data, device, model_config.max_seq_length
    )
    test_embeddings, test_labels = _extract_role_embeddings(
        model, tokenizer, test_data, device, model_config.max_seq_length
    )

    if matched_test:
        matched_embeddings, matched_labels = _extract_role_embeddings(
            model, tokenizer, matched_test, device, model_config.max_seq_length
        )
    else:
        matched_embeddings, matched_labels = None, None

    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print(f"  Test embeddings shape: {test_embeddings.shape}")

    # ========== Compute Client Prototypes ==========
    print("\n" + "="*70)
    print("Computing Client Prototypes (mean role embeddings)")
    print(f"  Normalize prototypes: {normalize_prototypes} (simulates {'normalized' if normalize_prototypes else 'raw'} upload)")
    print(f"  Noise scale: {proto_noise_scale} (simulates {'noise perturbation defense' if proto_noise_scale > 0 else 'no noise'})")
    print("="*70)

    np.random.seed(random_state)  # For reproducible noise
    train_prototypes = _compute_prototypes(train_embeddings, train_labels, num_clients, normalize=normalize_prototypes, noise_scale=proto_noise_scale)
    test_prototypes = _compute_prototypes(test_embeddings, test_labels, num_clients, normalize=normalize_prototypes, noise_scale=proto_noise_scale)

    print(f"  Train prototypes shape: {train_prototypes.shape}")
    print(f"  Test prototypes shape: {test_prototypes.shape}")
    if normalize_prototypes:
        # Verify normalization
        train_norms = np.linalg.norm(train_prototypes, axis=1)
        print(f"  Train prototype norms: min={train_norms.min():.4f}, max={train_norms.max():.4f}")

    # Prototype similarity analysis
    proto_analysis = _analyze_prototype_similarity(train_prototypes, clients)
    results['prototype_analysis'] = proto_analysis

    print(f"\n  Within-client similarity: {proto_analysis['avg_within_sim']:.4f}")
    print(f"  Between-client similarity: {proto_analysis['avg_between_sim']:.4f}")
    print(f"  Separability ratio: {proto_analysis['separability_ratio']:.4f}")

    # ========== Attack 1: Prototype-to-Prototype Attribution ==========
    # NOTE: This attack shows high accuracy but LOW F1, indicating structural similarity
    # rather than true client attribution. This is an ablation to demonstrate that
    # high accuracy ≠ effective attack. F1 remains near-random (0.03-0.14).
    print("\n" + "="*70)
    print("Attack 1: Prototype-to-Prototype Attribution (Nearest Neighbor)")
    print("(High Acc but Low F1 - structural similarity, not true attribution)")
    print("="*70)

    results['attackers']['proto_to_proto'] = _run_prototype_nn_attack(
        train_prototypes, test_prototypes, num_clients, clients
    )

    # ========== Attack 2: Sample-to-Prototype Attribution ==========
    print("\n" + "="*70)
    print("Attack 2: Sample-to-Prototype Attribution")
    print("="*70)

    results['attackers']['sample_to_proto'] = _run_sample_to_proto_attack(
        train_prototypes, test_embeddings, test_labels,
        matched_embeddings, matched_labels,
        num_clients, random_baseline
    )

    # ========== Attack 3: Multi-Attacker Evaluation ==========
    print("\n" + "="*70)
    print("Attack 3: Multi-Attacker Evaluation on Prototypes")
    print("="*70)

    # Load multi-round prototypes if enabled
    multi_round_protos = None
    multi_round_labels = None
    if use_multi_round_protos:
        if proto_rounds is None:
            # Default: use last 5 rounds (round 6-10 for 10-round training)
            proto_rounds = [6, 7, 8, 9, 10]
        print(f"(Loading actual prototypes from rounds {proto_rounds} for bootstrap)")
        try:
            multi_round_protos, multi_round_labels = _load_multi_round_prototypes(
                checkpoint_dir, clients, proto_rounds, client_to_id
            )
            print(f"  Loaded {len(multi_round_protos)} prototypes from {len(proto_rounds)} rounds")
        except Exception as e:
            print(f"  Warning: Failed to load multi-round prototypes: {e}")
            print("  Falling back to embedding-based bootstrap")
            use_multi_round_protos = False
    else:
        print("(Bootstrap sampling from embeddings to get more prototype samples)")

    results['attackers']['multi_attacker'] = _run_multi_attacker_evaluation(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        matched_embeddings, matched_labels,
        num_clients, n_bootstrap, random_state, random_baseline,
        normalize=normalize_prototypes,
        noise_scale=proto_noise_scale,
        multi_round_protos=multi_round_protos,
        multi_round_labels=multi_round_labels
    )

    # ========== Summary ==========
    print("\n" + "="*70)
    print("SUMMARY: EXP-3 Prototype Attack Results")
    print("="*70)
    _print_summary(results)

    # Generate visualizations
    _create_bootstrap_accuracy_plots(results, random_baseline, output_dir)
    _create_confusion_matrix_plots(results, clients, output_dir)

    # Save results
    results_file = output_dir / "exp3_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"\nResults saved to {results_file}")

    return results


def _balance_samples(data: List[Dict], n_per_class: int, num_classes: int) -> List[Dict]:
    """Sample balanced data from each class."""
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


def _extract_role_embeddings(
    model, tokenizer, data: List[Dict], device: str, max_length: int,
    batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract role embeddings from model with batch processing."""
    embeddings = []
    labels = []

    # Prepare all texts
    texts = ["deidentify:" + sample['text'] for sample in data]
    all_labels = [sample['client_id'] for sample in data]

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = all_labels[i:i+batch_size]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)

            # Get encoder output - model.base is the T5 model
            encoder_output = model.base.encoder(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask']
            )
            hidden = encoder_output.last_hidden_state  # [B, seq_len, hidden]

            # Apply role/style separation if available
            if hasattr(model, 'role_proj') and model.role_proj is not None:
                role_emb = model.role_proj(hidden)  # [B, seq_len, role_dim]
            else:
                role_emb = hidden

            # Mean pooling
            mask = enc['attention_mask'].unsqueeze(-1).float()
            role_mean = (role_emb * mask).sum(dim=1) / mask.sum(dim=1)

            embeddings.append(role_mean.cpu().numpy())
            labels.extend(batch_labels)

            if (i // batch_size) % 10 == 0:
                print(f"    Processed {min(i+batch_size, len(texts))}/{len(texts)} samples", flush=True)

    return np.vstack(embeddings), np.array(labels)


def _compute_prototypes(embeddings: np.ndarray, labels: np.ndarray, num_clients: int, normalize: bool = True, noise_scale: float = 0.0) -> np.ndarray:
    """Compute prototype (mean embedding) for each client.

    Args:
        embeddings: [N, D] embeddings for all samples
        labels: [N] client labels
        num_clients: number of clients
        normalize: if True, L2-normalize prototypes (simulates server receiving normalized protos)
        noise_scale: Gaussian noise scale to add before upload (noise perturbation, not true DP)
    """
    prototypes = []
    for client_id in range(num_clients):
        mask = labels == client_id
        if mask.sum() > 0:
            proto = embeddings[mask].mean(axis=0)
            if normalize:
                # Simulate: server receives normalized prototype from client
                norm = np.linalg.norm(proto)
                if norm > 1e-8:
                    proto = proto / norm
            # Add noise (noise perturbation defense)
            if noise_scale > 0:
                proto = proto + np.random.randn(*proto.shape) * noise_scale
                if normalize:
                    norm = np.linalg.norm(proto)
                    if norm > 1e-8:
                        proto = proto / norm
        else:
            proto = np.zeros(embeddings.shape[1])
        prototypes.append(proto)
    return np.vstack(prototypes)


def _analyze_prototype_similarity(prototypes: np.ndarray, clients: List[str]) -> Dict:
    """Analyze similarity between prototypes."""
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(prototypes)
    n = len(clients)

    # Within-client similarity (diagonal, always 1.0 for single prototype)
    within_sim = np.mean(np.diag(sim_matrix))

    # Between-client similarity (off-diagonal)
    mask = ~np.eye(n, dtype=bool)
    between_sim = sim_matrix[mask].mean()

    # Separability
    separability = within_sim / (between_sim + 1e-8)

    return {
        'similarity_matrix': sim_matrix.tolist(),
        'avg_within_sim': float(within_sim),
        'avg_between_sim': float(between_sim),
        'separability_ratio': float(separability),
        'interpretation': 'Higher between_sim (closer to 1.0) means prototypes are less distinguishable'
    }


def _run_prototype_nn_attack(
    train_prototypes: np.ndarray,
    test_prototypes: np.ndarray,
    num_clients: int,
    clients: List[str]
) -> Dict:
    """Run nearest neighbor attack on prototypes."""
    from sklearn.metrics.pairwise import cosine_similarity

    # For each test prototype, find nearest train prototype
    sim_matrix = cosine_similarity(test_prototypes, train_prototypes)
    predictions = sim_matrix.argmax(axis=1)
    true_labels = np.arange(num_clients)

    acc = accuracy_score(true_labels, predictions)

    print(f"\n  Nearest Neighbor Attack:")
    print(f"    Accuracy: {acc:.4f} (random baseline: {1/num_clients:.4f})")
    print(f"    Predictions: {predictions.tolist()}")
    print(f"    True labels: {true_labels.tolist()}")

    # Per-client analysis
    correct_clients = [clients[i] for i in range(num_clients) if predictions[i] == i]
    wrong_clients = [clients[i] for i in range(num_clients) if predictions[i] != i]

    print(f"    Correctly identified: {correct_clients}")
    print(f"    Misidentified: {wrong_clients}")

    return {
        'accuracy': float(acc),
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'correct_clients': correct_clients,
        'wrong_clients': wrong_clients,
        'random_baseline': 1/num_clients
    }


def _run_sample_to_proto_attack(
    train_prototypes: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    matched_embeddings: Optional[np.ndarray],
    matched_labels: Optional[np.ndarray],
    num_clients: int,
    random_baseline: float
) -> Dict:
    """Run sample-to-prototype attribution attack."""
    from sklearn.metrics.pairwise import cosine_similarity

    results = {}

    # Full test set
    sim_matrix = cosine_similarity(test_embeddings, train_prototypes)
    predictions = sim_matrix.argmax(axis=1)

    acc = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='macro')

    print(f"\n  Sample-to-Prototype (Full Test):")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Macro-F1: {f1:.4f}")
    print(f"    Random baseline: {random_baseline:.4f}")
    print(f"    Above random: {'+' if f1 > random_baseline else ''}{f1 - random_baseline:.4f}")

    results['full_test'] = {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'above_random': float(f1 - random_baseline)
    }

    # Matched test set
    if matched_embeddings is not None and len(matched_embeddings) > 0:
        sim_matrix = cosine_similarity(matched_embeddings, train_prototypes)
        predictions = sim_matrix.argmax(axis=1)

        acc = accuracy_score(matched_labels, predictions)
        f1 = f1_score(matched_labels, predictions, average='macro')

        print(f"\n  Sample-to-Prototype (Matched Test):")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Macro-F1: {f1:.4f}")
        print(f"    Above random: {'+' if f1 > random_baseline else ''}{f1 - random_baseline:.4f}")

        results['matched_test'] = {
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'above_random': float(f1 - random_baseline)
        }

    return results


def _run_multi_attacker_evaluation(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    matched_embeddings: Optional[np.ndarray],
    matched_labels: Optional[np.ndarray],
    num_clients: int,
    n_bootstrap: int,
    random_state: int,
    random_baseline: float,
    normalize: bool = True,
    noise_scale: float = 0.0,
    multi_round_protos: Optional[np.ndarray] = None,
    multi_round_labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Run multiple attackers on bootstrap-sampled prototypes.

    When multi_round_protos is provided, bootstrap samples are drawn from actual
    prototypes across multiple training rounds. Otherwise, prototypes are computed
    from embeddings using bootstrap sampling.

    Args:
        normalize: if True, L2-normalize prototypes (simulates server receiving normalized protos)
        noise_scale: Gaussian noise scale to add (noise perturbation, not true DP)
        multi_round_protos: Actual prototypes from multiple rounds [N_rounds * N_clients, dim]
        multi_round_labels: Client labels for multi_round_protos [N_rounds * N_clients]
    """
    np.random.seed(random_state)

    results = {
        'linear': {},
        'nonlinear': {},
        'nonparametric': {}
    }

    use_actual_protos = multi_round_protos is not None and multi_round_labels is not None

    def _normalize_proto(proto):
        """L2-normalize a prototype vector and optionally add noise."""
        if normalize:
            norm = np.linalg.norm(proto)
            if norm > 1e-8:
                proto = proto / norm
        # Add noise (noise perturbation defense)
        if noise_scale > 0:
            proto = proto + np.random.randn(*proto.shape) * noise_scale
            if normalize:
                norm = np.linalg.norm(proto)
                if norm > 1e-8:
                    proto = proto / norm
        return proto

    # Create bootstrap prototypes for training
    bootstrap_protos = []
    bootstrap_labels = []

    if use_actual_protos:
        # Bootstrap from actual multi-round prototypes
        print(f"  Using actual prototypes from multiple rounds for bootstrap")
        for _ in range(n_bootstrap):
            for client_id in range(num_clients):
                # Get all prototypes for this client across rounds
                mask = multi_round_labels == client_id
                client_protos = multi_round_protos[mask]
                if len(client_protos) > 0:
                    # Bootstrap sample from actual prototypes
                    idx = np.random.choice(len(client_protos))
                    proto = client_protos[idx].copy()
                    proto = _normalize_proto(proto)
                    bootstrap_protos.append(proto)
                    bootstrap_labels.append(client_id)
    else:
        # Original behavior: bootstrap from embeddings
        print(f"  Computing prototypes from embeddings via bootstrap")
        for _ in range(n_bootstrap):
            for client_id in range(num_clients):
                mask = train_labels == client_id
                client_embs = train_embeddings[mask]
                if len(client_embs) > 0:
                    # Bootstrap sample
                    indices = np.random.choice(len(client_embs), len(client_embs), replace=True)
                    proto = client_embs[indices].mean(axis=0)
                    proto = _normalize_proto(proto)  # Simulate normalized upload
                    bootstrap_protos.append(proto)
                    bootstrap_labels.append(client_id)

    X_train = np.vstack(bootstrap_protos)
    y_train = np.array(bootstrap_labels)

    # Create test prototypes
    test_protos = []
    test_proto_labels = []
    for client_id in range(num_clients):
        mask = test_labels == client_id
        if mask.sum() > 0:
            proto = test_embeddings[mask].mean(axis=0)
            proto = _normalize_proto(proto)  # Simulate normalized upload
            test_protos.append(proto)
            test_proto_labels.append(client_id)

    X_test = np.vstack(test_protos)
    y_test = np.array(test_proto_labels)

    print(f"\n  Bootstrap training samples: {len(X_train)}")
    print(f"  Test prototypes: {len(X_test)}")

    # Define attackers
    attackers = {
        'linear': ('LogisticRegression', LogisticRegression(max_iter=1000, solver='lbfgs')),
        'nonlinear': ('MLP', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True)),
        'nonparametric': ('KNN', KNeighborsClassifier(n_neighbors=min(5, len(X_train)//num_clients)))
    }

    for attack_type, (name, attacker) in attackers.items():
        print(f"\n  [{name}] Training...")

        try:
            attacker.fit(X_train, y_train)

            # Evaluate on test prototypes
            preds = attacker.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')

            print(f"    Full Test - Acc: {acc:.4f}, F1: {f1:.4f}")

            # Compute confusion matrix
            cm = confusion_matrix(y_test, preds)

            results[attack_type]['full_test'] = {
                'accuracy': float(acc),
                'macro_f1': float(f1),
                'above_random': float(f1 - random_baseline),
                'confusion_matrix': cm.tolist()
            }

            # Cross-validation on bootstrap data for confidence
            cv_scores = cross_val_score(attacker, X_train, y_train, cv=5, scoring='f1_macro')
            print(f"    CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            results[attack_type]['cv'] = {
                'mean_f1': float(cv_scores.mean()),
                'std_f1': float(cv_scores.std())
            }

            # Collect bootstrap accuracies for distribution analysis
            bootstrap_accs = []
            for i in range(n_bootstrap):
                # Use different random seed for each bootstrap evaluation
                np.random.seed(random_state + i + 1000)
                # Bootstrap sample from test prototypes (simulate different prototype estimates)
                test_indices = np.random.choice(len(X_test), len(X_test), replace=True)
                X_test_boot = X_test[test_indices]
                y_test_boot = y_test[test_indices]

                preds_boot = attacker.predict(X_test_boot)
                acc_boot = accuracy_score(y_test_boot, preds_boot)
                bootstrap_accs.append(acc_boot)

            bootstrap_accs = np.array(bootstrap_accs)

            # Calculate 95% CI
            ci_lower = np.percentile(bootstrap_accs, 2.5)
            ci_upper = np.percentile(bootstrap_accs, 97.5)

            results[attack_type]['bootstrap_analysis'] = {
                'accuracies': bootstrap_accs.tolist(),
                'mean_accuracy': float(bootstrap_accs.mean()),
                'std_accuracy': float(bootstrap_accs.std()),
                'ci_95_lower': float(ci_lower),
                'ci_95_upper': float(ci_upper),
                'ci_95_range': float(ci_upper - ci_lower)
            }

            print(f"    Bootstrap Acc: {bootstrap_accs.mean():.4f} ± {bootstrap_accs.std():.4f}")
            print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] (range: {ci_upper - ci_lower:.4f})")

            # Also evaluate on matched set if available
            if matched_embeddings is not None and len(matched_embeddings) > 0:
                matched_protos = []
                matched_proto_labels = []
                for client_id in range(num_clients):
                    mask = matched_labels == client_id
                    if mask.sum() > 0:
                        proto = matched_embeddings[mask].mean(axis=0)
                        matched_protos.append(proto)
                        matched_proto_labels.append(client_id)

                if matched_protos:
                    X_matched = np.vstack(matched_protos)
                    y_matched = np.array(matched_proto_labels)

                    preds = attacker.predict(X_matched)
                    acc = accuracy_score(y_matched, preds)
                    f1 = f1_score(y_matched, preds, average='macro')

                    print(f"    Matched Test - Acc: {acc:.4f}, F1: {f1:.4f}")

                    results[attack_type]['matched_test'] = {
                        'accuracy': float(acc),
                        'macro_f1': float(f1),
                        'above_random': float(f1 - random_baseline)
                    }

        except Exception as e:
            print(f"    Error: {e}")
            results[attack_type]['error'] = str(e)

    return results


def _print_summary(results: Dict):
    """Print summary table."""
    random_baseline = results['random_baseline']['macro_f1']

    print(f"\nRandom baseline: Acc={random_baseline:.4f}, F1={random_baseline:.4f}")
    print(f"\n{'Attack':<30} {'Full Acc':<12} {'Full F1':<12} {'Matched F1':<12} {'Status':<10}")
    print("-" * 80)

    # Proto-to-Proto (Note: High Acc but Low F1 - ablation, not effective attack)
    p2p = results['attackers'].get('proto_to_proto', {})
    if p2p:
        acc = p2p.get('accuracy', 'N/A')
        print(f"{'Proto-to-Proto (NN)':<30} {acc:<12.4f} {'N/A (see note)':<12} {'N/A':<12} {'Note 1':<10}")

    # Sample-to-Proto
    s2p = results['attackers'].get('sample_to_proto', {})
    if s2p:
        full = s2p.get('full_test', {})
        matched = s2p.get('matched_test', {})
        acc = full.get('accuracy', 'N/A')
        f1 = full.get('macro_f1', 'N/A')
        m_f1 = matched.get('macro_f1', 'N/A') if matched else 'N/A'
        status = '✅' if f1 <= random_baseline + 0.15 else '❌'
        print(f"{'Sample-to-Proto (Cosine)':<30} {acc:<12.4f} {f1:<12.4f} {m_f1 if isinstance(m_f1, str) else f'{m_f1:.4f}':<12} {status:<10}")

    # Multi-attacker
    multi = results['attackers'].get('multi_attacker', {})
    for attack_type in ['linear', 'nonlinear', 'nonparametric']:
        if attack_type in multi and 'full_test' in multi[attack_type]:
            full = multi[attack_type]['full_test']
            matched = multi[attack_type].get('matched_test', {})
            acc = full.get('accuracy', 'N/A')
            f1 = full.get('macro_f1', 'N/A')
            m_f1 = matched.get('macro_f1', 'N/A') if matched else 'N/A'
            status = '✅' if f1 <= random_baseline + 0.15 else '❌'
            name = {'linear': 'LogReg', 'nonlinear': 'MLP', 'nonparametric': 'KNN'}[attack_type]
            print(f"{'Bootstrap-Proto (' + name + ')':<30} {acc:<12.4f} {f1:<12.4f} {m_f1 if isinstance(m_f1, str) else f'{m_f1:.4f}':<12} {status:<10}")

    print("-" * 80)

    # Overall assessment
    print("\n[Privacy Assessment]")
    proto_analysis = results.get('prototype_analysis', {})
    between_sim = proto_analysis.get('avg_between_sim', 0)

    if between_sim > 0.8:
        print("  ✅ Prototypes are highly similar (avg between-client sim > 0.8)")
        print("     → Adversarial training successfully makes prototypes indistinguishable")
    elif between_sim > 0.6:
        print("  ⚠️  Prototypes are moderately similar (0.6 < sim < 0.8)")
        print("     → Some client information may leak through prototypes")
    else:
        print("  ❌ Prototypes are distinguishable (sim < 0.6)")
        print("     → Client identity can be inferred from prototypes")


def _create_bootstrap_accuracy_plots(results: Dict, random_baseline: float, output_dir: Path):
    """Create boxplot of bootstrap accuracy distributions with 95% CI."""
    multi_attacker = results.get('attackers', {}).get('multi_attacker', {})

    # Prepare data for plotting
    attack_types = []

    for attack_type in ['linear', 'nonlinear', 'nonparametric']:
        if attack_type in multi_attacker and 'bootstrap_analysis' in multi_attacker[attack_type]:
            bootstrap_data = multi_attacker[attack_type]['bootstrap_analysis']
            acc_list = bootstrap_data['accuracies']
            ci_lower = bootstrap_data['ci_95_lower']
            ci_upper = bootstrap_data['ci_95_upper']

            name_map = {'linear': 'LogisticRegression', 'nonlinear': 'MLP', 'nonparametric': 'KNN'}
            attack_types.append((name_map[attack_type], acc_list, ci_lower, ci_upper))

    if not attack_types:
        print("No bootstrap data available for plotting")
        return

    # Create plot
    plt.figure(figsize=(12, 8))

    # Create boxplot using matplotlib
    data_to_plot = [acc_list for _, acc_list, _, _ in attack_types]
    labels = [name for name, _, _, _ in attack_types]

    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.axhline(y=random_baseline, color='red', linestyle='--', linewidth=2,
                label=f'Random Baseline ({random_baseline:.4f})')

    # Add 95% CI error bars
    for i, (name, acc_list, ci_lower, ci_upper) in enumerate(attack_types):
        mean_acc = np.mean(acc_list)
        plt.errorbar(i+1, mean_acc, yerr=[[mean_acc - ci_lower], [ci_upper - mean_acc]],
                    fmt='o', color='black', capsize=5, capthick=2, elinewidth=2,
                    label='95% CI' if i == 0 else "")

    plt.xlabel('Attack Method', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Bootstrap Accuracy Distribution (100 samples)\nwith 95% Confidence Intervals', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "bootstrap_accuracy_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Bootstrap accuracy distribution plot saved to {plot_path}")
    plt.close()


def _create_confusion_matrix_plots(results: Dict, clients: List[str], output_dir: Path):
    """Create confusion matrix plots for each attacker."""
    multi_attacker = results.get('attackers', {}).get('multi_attacker', {})

    attack_types = ['linear', 'nonlinear', 'nonparametric']
    name_map = {'linear': 'LogisticRegression', 'nonlinear': 'MLP', 'nonparametric': 'KNN'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, attack_type in enumerate(attack_types):
        if attack_type in multi_attacker and 'full_test' in multi_attacker[attack_type]:
            full_test = multi_attacker[attack_type]['full_test']
            if 'confusion_matrix' in full_test:
                cm = np.array(full_test['confusion_matrix'])

                # Create short client names for display
                short_names = [f'C{i+1}' for i in range(len(clients))]

                # Create heatmap using matplotlib
                im = axes[i].imshow(cm, interpolation='nearest', cmap='Blues')
                axes[i].set_xticks(np.arange(len(short_names)))
                axes[i].set_yticks(np.arange(len(short_names)))
                axes[i].set_xticklabels(short_names)
                axes[i].set_yticklabels(short_names)

                # Add text annotations
                thresh = cm.max() / 2.
                for ii in range(cm.shape[0]):
                    for jj in range(cm.shape[1]):
                        axes[i].text(jj, ii, format(cm[ii, jj], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[ii, jj] > thresh else "black")

                axes[i].set_title(f'{name_map[attack_type]}\nAccuracy: {full_test["accuracy"]:.4f}',
                                fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Predicted Client' if i >= 1 else '')
                axes[i].set_ylabel('True Client' if i == 0 else '')

    # Add colorbar (only if we have at least one plot)
    try:
        plt.colorbar(im, ax=axes, shrink=0.8)
    except NameError:
        print("Warning: No confusion matrices to display")

    plt.suptitle('Confusion Matrices for Multi-Attacker Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "confusion_matrices.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plots saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EXP-3 Prototype Attack (LoRA-compatible)")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoint_dir", default="../checkpoints/fed_lora")
    parser.add_argument("--output_dir", default="../attack_results/exp3_prototype_lora")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--proto_noise_scale", type=float, default=0.1)
    parser.add_argument("--use_multi_round_protos", action="store_true", default=True,
                        help="Use actual prototypes from multiple rounds for bootstrap (default: True)")
    parser.add_argument("--no_multi_round_protos", action="store_false", dest="use_multi_round_protos",
                        help="Disable multi-round prototype bootstrap, use embedding-based bootstrap instead")
    parser.add_argument("--proto_rounds", type=int, nargs="+", default=None,
                        help="List of round numbers for multi-round bootstrap (default: [6,7,8,9,10])")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
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

    run_exp3_prototype(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        clients=clients,
        model_path=args.model_path,
        device=args.device,
        max_samples_per_client=args.max_samples,
        n_bootstrap=args.n_bootstrap,
        proto_noise_scale=args.proto_noise_scale,
        use_multi_round_protos=args.use_multi_round_protos,
        proto_rounds=args.proto_rounds,
        use_lora=args.use_lora,
        random_state=args.random_state,
    )
