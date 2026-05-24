"""
EXP-1-Text: Text-level Stylometry Attack (Source Attribution from Sanitized Text)

Threat Model: Attacker observes sanitized text ẑd and attempts to infer source client.
This directly evaluates Goal G2: "Reduced source attribution - an adversary observing ẑd
has substantially degraded ability to identify the source agent beyond the public tag prior."

Attack methods:
- A1: TF-IDF (word 1-2gram) + Logistic Regression
- A2: TF-IDF (char 2-5gram) + Logistic Regression
- A3: TF-IDF (word) + SVM
- A4: TF-IDF (word) + Random Forest
- A5: Bag of Words + Naive Bayes

Two evaluation settings:
1. Full Test: 7-way classification across all clients (realistic scenario)
2. Controlled Test: Binary classification on shared doc-types (controls for confounding)

The controlled test addresses the concern that doc_type may confound client attribution.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Loading
# =============================================================================

def load_all_client_data(data_dir: str, clients: List[str]) -> List[Dict]:
    """Load all samples from all clients."""
    data_dir = Path(data_dir)
    all_samples = []

    for client in clients:
        for jsonl_file in data_dir.glob(f"{client}*.jsonl"):
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
                        data['client'] = client
                        data['doc_type'] = doc_type
                        data['doc_id'] = f"{jsonl_file.stem}_{line_idx}"
                        all_samples.append(data)
                    except:
                        continue

    return all_samples


def get_shared_doctype_pairs(data_dir: str) -> Dict:
    """Identify doc_types shared by multiple clients for controlled tests."""
    data_dir = Path(data_dir)

    # Hardcoded based on dataset structure
    return {
        'Financial_Regulatory_Compliance_Report': {
            'clients': ['Client_1_CorporateBank', 'Client_6_ComplianceConsult'],
            'files': [
                data_dir / 'Client_1_CorporateBank_Financial_Regulatory_Compliance_Report_annotated.jsonl',
                data_dir / 'Client_6_ComplianceConsult_Financial_Regulatory_Compliance_Report_annotated.jsonl',
            ]
        },
        'Financial_Risk_Assessment': {
            'clients': ['Client_1_CorporateBank', 'Client_4_CorpGroup'],
            'files': [
                data_dir / 'Client_1_CorporateBank_Financial_Risk_Assessment_annotated.jsonl',
                data_dir / 'Client_4_CorpGroup_Financial_Risk_Assessment_annotated.jsonl',
            ]
        }
    }


# =============================================================================
# Attackers
# =============================================================================

class TextAttacker:
    """Text-based stylometry attacker."""

    def __init__(self, vectorizer, classifier, name: str):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.name = name

    def fit(self, texts: List[str], labels: np.ndarray):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        return self

    def evaluate(self, texts: List[str], labels: np.ndarray) -> Dict:
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        return {
            'accuracy': accuracy_score(labels, y_pred),
            'macro_f1': f1_score(labels, y_pred, average='macro'),
        }


def create_attackers() -> List[TextAttacker]:
    """Create suite of stylometry attackers."""
    return [
        TextAttacker(
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1),
            "TF-IDF(word)+LR"
        ),
        TextAttacker(
            TfidfVectorizer(max_features=5000, ngram_range=(2, 5), analyzer='char', sublinear_tf=True),
            LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1),
            "TF-IDF(char)+LR"
        ),
        TextAttacker(
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True),
            LinearSVC(max_iter=2000),
            "TF-IDF(word)+SVM"
        ),
        TextAttacker(
            TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            "TF-IDF(word)+RF"
        ),
        TextAttacker(
            CountVectorizer(max_features=5000, ngram_range=(1, 2)),
            MultinomialNB(),
            "BoW+NB"
        ),
    ]


# =============================================================================
# Experiment: Full 7-way Classification
# =============================================================================

def run_full_test(
    samples: List[Dict],
    text_field: str,
    clients: List[str],
    n_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Run 7-way classification with cross-validation.
    """
    client_to_id = {c: i for i, c in enumerate(clients)}

    # Prepare data
    texts = []
    labels = []
    for s in samples:
        text = s.get(text_field, '')
        if text and len(text.strip()) > 10 and s['client'] in client_to_id:
            texts.append(text)
            labels.append(client_to_id[s['client']])

    texts = np.array(texts)
    labels = np.array(labels)

    print(f"  Data: {len(texts)} samples, {len(clients)} classes")

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {name: {'accs': [], 'f1s': []} for name in [a.name for a in create_attackers()]}

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        train_texts, test_texts = texts[train_idx], texts[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        for attacker in create_attackers():
            attacker.fit(train_texts.tolist(), train_labels)
            metrics = attacker.evaluate(test_texts.tolist(), test_labels)
            results[attacker.name]['accs'].append(metrics['accuracy'])
            results[attacker.name]['f1s'].append(metrics['macro_f1'])

    # Aggregate
    summary = {}
    for name, data in results.items():
        summary[name] = {
            'acc_mean': np.mean(data['accs']),
            'acc_std': np.std(data['accs']),
            'f1_mean': np.mean(data['f1s']),
            'f1_std': np.std(data['f1s']),
        }

    return summary


# =============================================================================
# Experiment: Controlled Binary Classification (Same Doc-Type)
# =============================================================================

def run_controlled_test(
    data_dir: str,
    text_field: str,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Run binary classification on shared doc-types (controlled for doc_type confounding).
    """
    shared_pairs = get_shared_doctype_pairs(data_dir)

    all_results = {}

    for doc_type, info in shared_pairs.items():
        print(f"\n  [{doc_type}]")
        print(f"    Clients: {info['clients'][0]} vs {info['clients'][1]}")

        # Load data
        texts = []
        labels = []

        for label_id, filepath in enumerate(info['files']):
            if not filepath.exists():
                print(f"    Warning: {filepath} not found")
                continue
            with open(filepath) as f:
                for line in f:
                    s = json.loads(line)
                    text = s.get(text_field, '')
                    if text and len(text.strip()) > 10:
                        texts.append(text)
                        labels.append(label_id)

        texts = np.array(texts)
        labels = np.array(labels)

        print(f"    Data: {len(texts)} samples (class0={sum(labels==0)}, class1={sum(labels==1)})")

        if len(texts) < 20:
            print(f"    Skipping: insufficient data")
            continue

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        results = {name: {'accs': [], 'f1s': []} for name in [a.name for a in create_attackers()]}

        for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]

            for attacker in create_attackers():
                attacker.fit(train_texts.tolist(), train_labels)
                metrics = attacker.evaluate(test_texts.tolist(), test_labels)
                results[attacker.name]['accs'].append(metrics['accuracy'])
                results[attacker.name]['f1s'].append(metrics['macro_f1'])

        # Aggregate
        summary = {}
        for name, data in results.items():
            summary[name] = {
                'acc_mean': np.mean(data['accs']),
                'acc_std': np.std(data['accs']),
                'f1_mean': np.mean(data['f1s']),
                'f1_std': np.std(data['f1s']),
            }

        all_results[doc_type] = {
            'clients': info['clients'],
            'n_samples': len(texts),
            'results': summary
        }

    return all_results


# =============================================================================
# Main Experiment
# =============================================================================

def run_text_stylometry_experiment(
    data_dir: str,
    output_dir: str,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Run complete text-level stylometry attack experiment.

    Reports:
    1. Full Test: 7-way classification (realistic scenario)
    2. Controlled Test: Binary on shared doc-types (controls confounding)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clients = [
        "Client_1_CorporateBank",
        "Client_2_AssetManager",
        "Client_3_FinTechPay",
        "Client_4_CorpGroup",
        "Client_5_MarketForecaster",
        "Client_6_ComplianceConsult",
        "Client_7_SupplierCo",
    ]

    # Load all data
    all_samples = load_all_client_data(data_dir, clients)
    print(f"Loaded {len(all_samples)} total samples")

    results = {
        'experiment': 'EXP-1-Text: Stylometry Attack on Sanitized Text',
        'threat_model': 'Attacker observes sanitized text and attempts source attribution',
        'random_baseline_7way': 1.0 / len(clients),
        'random_baseline_binary': 0.5,
        'full_test': {},
        'controlled_test': {}
    }

    # ==========================================================================
    # Full Test: 7-way
    # ==========================================================================
    print("\n" + "="*70)
    print("FULL TEST: 7-way Client Attribution")
    print("="*70)

    for baseline_name, text_field in [('B0_raw', 'original_text'), ('B3_sanitized', 'rewritten_text')]:
        print(f"\n[{baseline_name}] text_field={text_field}")
        full_results = run_full_test(all_samples, text_field, clients, n_folds, random_state)
        results['full_test'][baseline_name] = full_results

        # Print
        best = max(full_results.items(), key=lambda x: x[1]['f1_mean'])
        print(f"  Best attacker: {best[0]}")
        print(f"    Acc: {best[1]['acc_mean']:.4f} ± {best[1]['acc_std']:.4f}")
        print(f"    F1:  {best[1]['f1_mean']:.4f} ± {best[1]['f1_std']:.4f}")

    # ==========================================================================
    # Controlled Test: Binary (same doc-type)
    # ==========================================================================
    print("\n" + "="*70)
    print("CONTROLLED TEST: Binary Attribution (Same Doc-Type)")
    print("This controls for doc_type as a confounding variable.")
    print("="*70)

    for baseline_name, text_field in [('B0_raw', 'original_text'), ('B3_sanitized', 'rewritten_text')]:
        print(f"\n[{baseline_name}] text_field={text_field}")
        controlled_results = run_controlled_test(data_dir, text_field, n_folds, random_state)
        results['controlled_test'][baseline_name] = controlled_results

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n[Full Test: 7-way] Random baseline = 0.1429")
    print(f"{'Baseline':<15} {'Best Attacker':<20} {'Acc':<20} {'F1':<20}")
    print("-"*75)
    for baseline, res in results['full_test'].items():
        best = max(res.items(), key=lambda x: x[1]['f1_mean'])
        acc_str = f"{best[1]['acc_mean']:.4f}±{best[1]['acc_std']:.4f}"
        f1_str = f"{best[1]['f1_mean']:.4f}±{best[1]['f1_std']:.4f}"
        print(f"{baseline:<15} {best[0]:<20} {acc_str:<20} {f1_str:<20}")

    print("\n[Controlled Test: Binary] Random baseline = 0.5000")
    print(f"{'Baseline':<15} {'Doc-Type':<45} {'Best F1':<20}")
    print("-"*80)
    for baseline, doc_results in results['controlled_test'].items():
        for doc_type, doc_data in doc_results.items():
            best = max(doc_data['results'].items(), key=lambda x: x[1]['f1_mean'])
            f1_str = f"{best[1]['f1_mean']:.4f}±{best[1]['f1_std']:.4f}"
            doc_short = doc_type[:42] + "..." if len(doc_type) > 42 else doc_type
            print(f"{baseline:<15} {doc_short:<45} {f1_str:<20}")

    # F1 drop analysis
    print("\n[Analysis] F1 Drop (B0 → B3)")
    print("-"*60)

    print("\nFull Test (7-way):")
    b0_full = results['full_test'].get('B0_raw', {})
    b3_full = results['full_test'].get('B3_sanitized', {})
    for attacker_name in b0_full.keys():
        if attacker_name in b3_full:
            b0_f1 = b0_full[attacker_name]['f1_mean']
            b3_f1 = b3_full[attacker_name]['f1_mean']
            drop = b0_f1 - b3_f1
            drop_pct = (drop / b0_f1 * 100) if b0_f1 > 0 else 0
            print(f"  {attacker_name:<20}: {b0_f1:.4f} → {b3_f1:.4f} (Δ={drop:+.4f}, {drop_pct:+.1f}%)")

    print("\nControlled Test (Binary):")
    b0_ctrl = results['controlled_test'].get('B0_raw', {})
    b3_ctrl = results['controlled_test'].get('B3_sanitized', {})
    for doc_type in b0_ctrl.keys():
        if doc_type in b3_ctrl:
            b0_best = max(b0_ctrl[doc_type]['results'].values(), key=lambda x: x['f1_mean'])['f1_mean']
            b3_best = max(b3_ctrl[doc_type]['results'].values(), key=lambda x: x['f1_mean'])['f1_mean']
            drop = b0_best - b3_best
            status = "✓ near-random" if b3_best < 0.55 else "⚠ above random"
            print(f"  {doc_type[:35]:<35}: {b0_best:.4f} → {b3_best:.4f} [{status}]")

    # Save
    results_file = output_dir / "exp1_text_stylometry_results.json"

    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_for_json)
    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EXP-1-Text: Stylometry Attack on Sanitized Text")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../attack_results/exp1_text_stylometry")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    run_text_stylometry_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        random_state=args.random_state
    )
