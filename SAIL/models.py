"""
SAIL ML Models:
1. Change Prediction Model  - Random Forest classifier
   Predicts whether a key-gate locality underwent synthesis changes.

2. Reconstruction Model     - Multi-layer perceptron ensemble
   Reverts post-synthesis localities to pre-synthesis form.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
from locality import locality_to_feature_vector, subgraph_snapshot
from bench_parser import GATE_TYPES


# ─────────────────────────────────────────────
#  Change Prediction Model
# ─────────────────────────────────────────────

class ChangePredictionModel:
    """
    Binary classifier: predicts whether a key-gate locality was changed
    by the synthesis tool (1 = changed, 0 = no change).
    Uses a Random Forest as in the paper.
    """

    def __init__(self, locality_size: int = 6, n_estimators: int = 200):
        self.locality_size = locality_size
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def _extract_features(self, localities: List[Tuple[nx.DiGraph, str]]) -> np.ndarray:
        features = []
        for subgraph, center in localities:
            fv = locality_to_feature_vector(subgraph, center)
            features.append(fv)
        if not features:
            return np.zeros((0, 1))
        # Pad/truncate to uniform length
        max_len = max(len(f) for f in features)
        padded = [np.pad(f, (0, max_len - len(f))) for f in features]
        return np.array(padded, dtype=np.float32)

    def fit(self, localities: List[Tuple[nx.DiGraph, str]],
            labels: List[int]) -> 'ChangePredictionModel':
        """
        localities: list of (subgraph, center_node)
        labels: 0 = no change (Level-1), 1 = changed (Level-2/3)
        """
        X = self._extract_features(localities)
        y = np.array(labels)
        if len(X) == 0:
            return self
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True
        self._feature_dim = X.shape[1]
        return self

    def predict(self, localities: List[Tuple[nx.DiGraph, str]]) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(localities), dtype=int)
        X = self._extract_features(localities)
        if len(X) == 0:
            return np.array([], dtype=int)
        # Pad to training feature dimension
        if X.shape[1] < self._feature_dim:
            X = np.pad(X, ((0, 0), (0, self._feature_dim - X.shape[1])))
        elif X.shape[1] > self._feature_dim:
            X = X[:, :self._feature_dim]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, localities: List[Tuple[nx.DiGraph, str]]) -> np.ndarray:
        if not self.fitted:
            return np.zeros((len(localities), 2))
        X = self._extract_features(localities)
        if len(X) == 0:
            return np.zeros((0, 2))
        if X.shape[1] < self._feature_dim:
            X = np.pad(X, ((0, 0), (0, self._feature_dim - X.shape[1])))
        elif X.shape[1] > self._feature_dim:
            X = X[:, :self._feature_dim]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, localities: List[Tuple[nx.DiGraph, str]],
                 labels: List[int]) -> Dict:
        preds = self.predict(localities)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'predictions': preds.tolist()}


# ─────────────────────────────────────────────
#  Reconstruction Model (single locality size)
# ─────────────────────────────────────────────

class ReconstructionModel:
    """
    Predicts the pre-synthesis gate configuration given a post-synthesis locality.
    Implemented as a multi-layer perceptron.

    The output is a gate-type sequence for the snapshot (center + neighbors).
    We encode this as a multi-class classification over gate types per node slot.
    """

    # Output: predict gate type of center node + up to 2 neighbors = 3 slots
    OUTPUT_SLOTS = 3
    NUM_CLASSES = len(GATE_TYPES) + 3  # gate types + INPUT + WIRE + UNKNOWN

    def __init__(self, post_synthesis_locality_size: int = 5,
                 hidden_layer_sizes: Tuple = (256, 128, 64)):
        self.locality_size = post_synthesis_locality_size
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self._feature_dim = None

    def _extract_features(self, post_localities: List[Tuple[nx.DiGraph, str]]) -> np.ndarray:
        features = []
        for subgraph, center in post_localities:
            fv = locality_to_feature_vector(subgraph, center)
            features.append(fv)
        if not features:
            return np.zeros((0, 1))
        max_len = max(len(f) for f in features)
        padded = [np.pad(f, (0, max_len - len(f))) for f in features]
        return np.array(padded, dtype=np.float32)

    def _encode_snapshot(self, pre_locality: nx.DiGraph, center: str) -> int:
        """Encode the pre-synthesis center gate type as label."""
        if center not in pre_locality:
            return self.NUM_CLASSES - 1  # UNKNOWN
        type_id = pre_locality.nodes[center].get('type_id', -1)
        if 0 <= type_id < len(GATE_TYPES):
            return type_id
        if pre_locality.nodes[center].get('is_input', False):
            return len(GATE_TYPES)
        return self.NUM_CLASSES - 1

    def fit(self,
            post_localities: List[Tuple[nx.DiGraph, str]],
            pre_localities: List[Tuple[nx.DiGraph, str]]) -> 'ReconstructionModel':
        """
        post_localities: (post-synthesis subgraph, center_node)
        pre_localities:  (pre-synthesis subgraph, center_node)
        """
        X = self._extract_features(post_localities)
        y = np.array([
            self._encode_snapshot(pre, center)
            for (pre, center) in pre_localities
        ])
        if len(X) == 0:
            return self
        self._feature_dim = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True
        return self

    def predict(self, post_localities: List[Tuple[nx.DiGraph, str]]) -> List[int]:
        """Returns predicted pre-synthesis gate type id for each center node."""
        if not self.fitted:
            return [self.NUM_CLASSES - 1] * len(post_localities)
        X = self._extract_features(post_localities)
        if len(X) == 0:
            return []
        if X.shape[1] < self._feature_dim:
            X = np.pad(X, ((0, 0), (0, self._feature_dim - X.shape[1])))
        elif X.shape[1] > self._feature_dim:
            X = X[:, :self._feature_dim]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).tolist()

    def predict_proba(self, post_localities: List[Tuple[nx.DiGraph, str]]) -> np.ndarray:
        if not self.fitted:
            return np.zeros((len(post_localities), self.NUM_CLASSES))
        X = self._extract_features(post_localities)
        if len(X) == 0:
            return np.zeros((0, self.NUM_CLASSES))
        if X.shape[1] < self._feature_dim:
            X = np.pad(X, ((0, 0), (0, self._feature_dim - X.shape[1])))
        elif X.shape[1] > self._feature_dim:
            X = X[:, :self._feature_dim]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate_snapshot(self,
                           post_localities: List[Tuple[nx.DiGraph, str]],
                           pre_localities: List[Tuple[nx.DiGraph, str]]) -> Dict:
        """
        Compute Gate Error and R-Metric as defined in the paper.
        R = GE[0]*1 + GE[1]*0.66 + GE[2]*0.33
        """
        preds = self.predict(post_localities)
        true_labels = [self._encode_snapshot(pre, center)
                       for (pre, center) in pre_localities]

        ge0 = sum(1 for p, t in zip(preds, true_labels) if p == t)
        ge1 = sum(1 for p, t in zip(preds, true_labels) if abs(p - t) == 1)
        ge2 = sum(1 for p, t in zip(preds, true_labels) if abs(p - t) == 2)
        n = len(preds)

        r_metric = (ge0 / n * 1.0 + ge1 / n * 0.66 + ge2 / n * 0.33) * 100 if n > 0 else 0
        complete_recovery = (ge0 / n * 100) if n > 0 else 0

        return {
            'complete_recovery_pct': complete_recovery,
            'r_metric': r_metric,
            'gate_error_0': ge0,
            'gate_error_1': ge1,
            'gate_error_2': ge2,
            'n_samples': n,
        }


# ─────────────────────────────────────────────
#  Reconstruction Ensemble (multiple sizes)
# ─────────────────────────────────────────────

class ReconstructionEnsemble:
    """
    Ensemble of Reconstruction Models trained on different locality sizes (3–10).
    Uses cumulative confidence voting as described in the paper.
    """

    def __init__(self, locality_sizes: List[int] = None):
        if locality_sizes is None:
            locality_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
        self.locality_sizes = locality_sizes
        self.models: Dict[int, ReconstructionModel] = {
            s: ReconstructionModel(post_synthesis_locality_size=s)
            for s in locality_sizes
        }

    def fit(self,
            training_data: List[Tuple[nx.DiGraph, nx.DiGraph, str, int]]) -> 'ReconstructionEnsemble':
        """
        training_data: list of (pre_locality, post_locality, center, change_level)
        """
        # Build per-size datasets
        for size in self.locality_sizes:
            post_locs = [(post, center) for pre, post, center, _ in training_data]
            pre_locs = [(pre, center) for pre, post, center, _ in training_data]
            if post_locs:
                self.models[size].fit(post_locs, pre_locs)
        return self

    def predict(self, post_localities: List[Tuple[nx.DiGraph, str]]) -> List[int]:
        """
        Cumulative confidence voting across all ensemble members.
        Returns majority-voted gate type predictions.
        """
        if not post_localities:
            return []

        n = len(post_localities)
        num_classes = ReconstructionModel.NUM_CLASSES
        vote_matrix = np.zeros((n, num_classes))

        for size, model in self.models.items():
            if model.fitted:
                try:
                    proba = model.predict_proba(post_localities)
                    if proba.shape[1] < num_classes:
                        proba = np.pad(proba, ((0, 0), (0, num_classes - proba.shape[1])))
                    vote_matrix += proba
                except Exception:
                    continue

        return vote_matrix.argmax(axis=1).tolist()

    def evaluate_snapshot(self,
                           post_localities: List[Tuple[nx.DiGraph, str]],
                           pre_localities: List[Tuple[nx.DiGraph, str]]) -> Dict:
        preds = self.predict(post_localities)
        true_labels = [
            self.models[self.locality_sizes[0]]._encode_snapshot(pre, center)
            for (pre, center) in pre_localities
        ]
        ge0 = sum(1 for p, t in zip(preds, true_labels) if p == t)
        ge1 = sum(1 for p, t in zip(preds, true_labels) if abs(p - t) == 1)
        ge2 = sum(1 for p, t in zip(preds, true_labels) if abs(p - t) == 2)
        n = len(preds)
        r_metric = (ge0 / n * 1.0 + ge1 / n * 0.66 + ge2 / n * 0.33) * 100 if n > 0 else 0
        complete_recovery = (ge0 / n * 100) if n > 0 else 0
        return {
            'complete_recovery_pct': complete_recovery,
            'r_metric': r_metric,
            'gate_error_0': ge0,
            'gate_error_1': ge1,
            'gate_error_2': ge2,
            'n_samples': n,
        }


# ─────────────────────────────────────────────
#  Change Prediction Boosted Reconstruction
# ─────────────────────────────────────────────

class SAILModel:
    """
    Full SAIL Attack model:
    1. Change Prediction Model filters which localities to reconstruct
    2. Reconstruction Ensemble performs actual recovery
    """

    def __init__(self):
        self.change_predictor = ChangePredictionModel()
        self.reconstruction_ensemble = ReconstructionEnsemble()

    def fit(self, training_data: List[Tuple[nx.DiGraph, nx.DiGraph, str, int]]):
        """
        training_data: list of (pre_locality, post_locality, center, change_level)
        """
        # Prepare change prediction data
        cp_localities = [(post, center) for pre, post, center, cl in training_data]
        cp_labels = [1 if cl > 0 else 0 for pre, post, center, cl in training_data]

        print(f"  Training Change Prediction Model on {len(cp_localities)} samples...")
        self.change_predictor.fit(cp_localities, cp_labels)

        print(f"  Training Reconstruction Ensemble on {len(training_data)} samples...")
        self.reconstruction_ensemble.fit(training_data)

        return self

    def attack(self, post_localities: List[Tuple[nx.DiGraph, str]],
               pre_localities: Optional[List[Tuple[nx.DiGraph, str]]] = None) -> Dict:
        """
        Run SAIL attack on post-synthesis localities.
        If pre_localities provided, compute accuracy metrics.
        """
        # Step 1: Change prediction
        change_predictions = self.change_predictor.predict(post_localities)
        changed_indices = [i for i, p in enumerate(change_predictions) if p == 1]
        unchanged_indices = [i for i, p in enumerate(change_predictions) if p == 0]

        # Step 2: Reconstruct only changed localities
        changed_localities = [post_localities[i] for i in changed_indices]
        reconstructed_types = {}

        if changed_localities:
            rec_preds = self.reconstruction_ensemble.predict(changed_localities)
            for idx, pred in zip(changed_indices, rec_preds):
                reconstructed_types[idx] = pred

        # Mark unchanged localities as keeping their current center type
        for idx in unchanged_indices:
            subgraph, center = post_localities[idx]
            if center in subgraph:
                reconstructed_types[idx] = subgraph.nodes[center].get('type_id', -1)

        results = {
            'n_total': len(post_localities),
            'n_predicted_changed': len(changed_indices),
            'n_predicted_unchanged': len(unchanged_indices),
            'reconstructed_gate_types': reconstructed_types,
            'change_predictions': change_predictions.tolist() if hasattr(change_predictions, 'tolist') else list(change_predictions),
        }

        # If ground truth available, compute metrics
        if pre_localities:
            encode_fn = self.reconstruction_ensemble.models[
                self.reconstruction_ensemble.locality_sizes[0]
            ]._encode_snapshot

            true_labels = [encode_fn(pre, center) for pre, center in pre_localities]
            preds = [reconstructed_types.get(i, -1) for i in range(len(post_localities))]

            ge0 = sum(1 for p, t in zip(preds, true_labels) if p == t)
            ge1 = sum(1 for p, t in zip(preds, true_labels) if abs(p - t) == 1)
            ge2 = sum(1 for p, t in zip(preds, true_labels) if abs(p - t) == 2)
            n = len(preds)
            r_metric = (ge0 / n + ge1 / n * 0.66 + ge2 / n * 0.33) * 100 if n > 0 else 0

            results['metrics'] = {
                'complete_recovery_pct': ge0 / n * 100 if n > 0 else 0,
                'r_metric': r_metric,
                'gate_error_0': ge0,
                'gate_error_1': ge1,
                'gate_error_2': ge2,
                'change_prediction_accuracy': (
                    sum(1 for p, t in zip(
                        results['change_predictions'],
                        [1 if encode_fn(pre, c) != post.nodes.get(c, {}).get('type_id', -2)
                         else 0 for (pre, c), (post, _) in zip(pre_localities, post_localities)]
                    ) if p == t) / n * 100 if n > 0 else 0
                )
            }

        return results