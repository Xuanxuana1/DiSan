"""
Attack models for privacy evaluation.

A1: TF-IDF + Logistic Regression (strong baseline for stylometry)
A2: Neural text classifier (Transformer-based)
P1: Logistic Regression probe on embeddings
P2: MLP probe on embeddings
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AttackMetrics:
    """Compute and store attack metrics."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
        self.y_true = y_true
        self.y_pred = y_pred
        self.num_classes = num_classes

        self.accuracy = accuracy_score(y_true, y_pred)
        self.macro_f1 = f1_score(y_true, y_pred, average='macro')
        self.conf_matrix = confusion_matrix(y_true, y_pred)

        # Attack advantage = Acc - random guess (1/num_classes)
        self.attack_advantage = self.accuracy - (1.0 / num_classes)

    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'attack_advantage': self.attack_advantage,
            'confusion_matrix': self.conf_matrix.tolist(),
        }

    def __str__(self) -> str:
        return (f"Accuracy: {self.accuracy:.4f}, "
                f"Macro-F1: {self.macro_f1:.4f}, "
                f"Attack Advantage: {self.attack_advantage:.4f}")


class TFIDFAttacker:
    """
    A1: TF-IDF + Logistic Regression attacker.
    Strong baseline that captures word frequency and stylistic patterns.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        max_iter: int = 1000
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver='lbfgs',
            n_jobs=-1
        )

    def fit(self, texts: List[str], labels: np.ndarray):
        """Train the attacker."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict client labels."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def evaluate(self, texts: List[str], labels: np.ndarray, num_classes: int) -> AttackMetrics:
        """Evaluate attack performance."""
        y_pred = self.predict(texts)
        return AttackMetrics(labels, y_pred, num_classes)

    def get_top_features(self, class_names: List[str], top_k: int = 10) -> Dict[str, List[str]]:
        """Get top discriminative features for each class."""
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = {}

        for i, class_name in enumerate(class_names):
            # Get coefficients for this class
            if len(self.classifier.classes_) == 2:
                coef = self.classifier.coef_[0]
            else:
                coef = self.classifier.coef_[i]

            # Get top positive features
            top_indices = np.argsort(coef)[-top_k:][::-1]
            top_features[class_name] = [
                f"{feature_names[idx]} ({coef[idx]:.3f})"
                for idx in top_indices
            ]

        return top_features


class NeuralAttacker:
    """
    A2: Neural text classifier using a simple Transformer encoder.
    Uses a pre-trained model as feature extractor + classification head.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.encoder.eval()  # Freeze encoder

        # Classification head
        encoder_dim = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)

        self.num_classes = num_classes

    def _encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.encoder(**enc)
                # Mean pooling
                attention_mask = enc["attention_mask"]
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = torch.sum(hidden * mask, dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embeddings.append(pooled.cpu().numpy())

        return np.vstack(embeddings)

    def fit(
        self,
        texts: List[str],
        labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3
    ):
        """Train the classifier head."""
        # Encode all texts first
        print("[NeuralAttacker] Encoding training texts...")
        X = self._encode(texts, batch_size)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(labels).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.classifier(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        return self

    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict client labels."""
        X = self._encode(texts, batch_size)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            return logits.argmax(dim=1).cpu().numpy()

    def evaluate(self, texts: List[str], labels: np.ndarray) -> AttackMetrics:
        """Evaluate attack performance."""
        y_pred = self.predict(texts)
        return AttackMetrics(labels, y_pred, self.num_classes)


class EmbeddingProbe:
    """
    Probe models for embedding analysis.
    Supports multiple classifiers to test embedding leakage.

    Classifiers:
        - logreg: Logistic Regression
        - mlp: Multi-Layer Perceptron
        - svm: Support Vector Machine
        - rf: Random Forest
        - knn: K-Nearest Neighbors
        - xgb: XGBoost (if available)
    """
    SUPPORTED_CLASSIFIERS = ['logreg', 'mlp', 'svm', 'rf', 'knn', 'xgb']

    def __init__(
        self,
        probe_type: str = "mlp",
        num_classes: int = 7,
        hidden_dims: List[int] = [256, 128],
        max_iter: int = 1000,
        random_state: int = 42
    ):
        self.probe_type = probe_type
        self.num_classes = num_classes
        self.random_state = random_state

        if probe_type == "logreg":
            self.model = LogisticRegression(
                max_iter=max_iter,
                solver='lbfgs',
                n_jobs=-1,
                random_state=random_state
            )
        elif probe_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                learning_rate_init=0.001,
                alpha=0.01,
                batch_size=64,
                solver='adam',
                random_state=random_state
            )
        elif probe_type == "svm":
            from sklearn.svm import SVC
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=random_state
            )
        elif probe_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=random_state
            )
        elif probe_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
        elif probe_type == "xgb":
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    n_jobs=-1
                )
            except ImportError:
                print("  XGBoost not available, falling back to RandomForest")
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=random_state
                )
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}. Supported: {self.SUPPORTED_CLASSIFIERS}")

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """Train the probe."""
        self.model.fit(embeddings, labels)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict client labels."""
        return self.model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(embeddings)

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> AttackMetrics:
        """Evaluate probe performance."""
        y_pred = self.predict(embeddings)
        return AttackMetrics(labels, y_pred, self.num_classes)


class BaselineGenerator:
    """
    Generate baseline outputs for comparison:
    - B0: Raw (original text)
    - B1: Placeholder-only (NER replacement, no model)
    - B2: Role/Style without GRL
    - B3: Full method
    """

    @staticmethod
    def generate_placeholder_only(text: str, pii_spans: List[Dict]) -> str:
        """
        B1: Simple placeholder replacement without model.
        Replace detected spans with their type labels.
        """
        if not pii_spans:
            return text

        # Sort spans by start position (descending) to replace from end
        sorted_spans = sorted(pii_spans, key=lambda x: x['start'], reverse=True)

        result = text
        for span in sorted_spans:
            start = span['start']
            end = span['end']
            label = span.get('label', 'PII')
            placeholder = f"[{label}]"
            result = result[:start] + placeholder + result[end:]

        return result

    @staticmethod
    def load_raw_data(data_dir: str, clients: List[str]) -> List[Dict]:
        """B0: Load raw original text."""
        from .data_utils import load_client_data

        all_samples = []
        for client in clients:
            samples = load_client_data(data_dir, client)
            for s in samples:
                s['text'] = s.get('original_text', '')
            all_samples.extend(samples)
        return all_samples

    @staticmethod
    def generate_b1_placeholder(data_dir: str, clients: List[str]) -> List[Dict]:
        """B1: Generate placeholder-only baseline."""
        import json
        from .data_utils import load_client_data

        all_samples = []
        for client in clients:
            samples = load_client_data(data_dir, client)
            for s in samples:
                original = s.get('original_text', '')
                pii_spans = s.get('pii_spans', [])
                if isinstance(pii_spans, str):
                    try:
                        pii_spans = json.loads(pii_spans)
                    except:
                        pii_spans = []

                s['text'] = BaselineGenerator.generate_placeholder_only(original, pii_spans)
            all_samples.extend(samples)
        return all_samples
