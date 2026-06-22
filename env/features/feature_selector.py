import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class FeatureSelector:
    """
    Full production-ready feature selection module:
      ✓ Elastic Net per pair
      ✓ Feature importance ranking
      ✓ Scree curve + automatic 'elbow' K
      ✓ Optionally override K manually
      ✓ Returns reduced feature matrices per pair
    """

    def __init__(
        self,
        alpha=0.001,
        l1_ratio=0.5,
        max_iter=5000,
        auto_k=True,
        k_override=None,
        importance_threshold=0.0,
    ):
        """
        alpha: elastic net regularization strength
        l1_ratio: 1.0 = LASSO, 0.0 = Ridge, 0.5 = ElasticNet
        auto_k: automatically detect K from elbow of importance curve
        k_override: force K to a manual number (e.g., 40)
        importance_threshold: drop all features below this absolute weight
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.auto_k = auto_k
        self.k_override = k_override
        self.importance_threshold = importance_threshold

        self.selected_features = {}     # {(A,B): array of idx}
        self.importances = {}           # {(A,B): array importance}
        self.k_values = {}              # {(A,B): int}
        self.scalers = {}               # {(A,B): train-fitted scaler}

    # -----------------------------------------------------------
    # Fit + select features for ONE pair
    # -----------------------------------------------------------
    def fit_pair(self, pair, X, y):
        """
        X: np.array shape (T, F)
        y: target vector (T,) — usually future spread returns or zscore change
        """

        # standardize X (important for Elastic Net)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self.scalers[pair] = scaler

        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
        )

        model.fit(Xs, y)

        importance = np.abs(model.coef_)
        self.importances[pair] = importance

        # ------------------------------------
        # 1) Remove features below threshold
        # ------------------------------------
        valid_idx = np.where(importance > self.importance_threshold)[0]
        sorted_valid = valid_idx[np.argsort(importance[valid_idx])[::-1]]

        if len(sorted_valid) == 0:
            variances = np.nan_to_num(np.var(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
            fallback_k = min(5, X.shape[1])
            sorted_valid = np.argsort(variances)[::-1][:fallback_k]

        # ------------------------------------
        # 2) Automatic elbow detection (optional)
        # ------------------------------------
        if self.k_override is not None:
            K = min(self.k_override, len(sorted_valid))
        elif self.auto_k:
            K = self._find_elbow_k(importance[sorted_valid])
        else:
            K = len(sorted_valid)

        self.k_values[pair] = K
        self.selected_features[pair] = sorted_valid[:K]

        return self.selected_features[pair]

    # -----------------------------------------------------------
    # ELBOW / SCREE CURVE DETECTION
    # -----------------------------------------------------------
    def _find_elbow_k(self, importance_sorted):
        """
        Detect the elbow of the importance curve.
        Uses "maximum distance from chord" method.
        """

        if len(importance_sorted) <= 3:
            return len(importance_sorted)

        y = importance_sorted
        n = len(y)

        # line from first to last point
        start = np.array([0, y[0]])
        end   = np.array([n - 1, y[-1]])
        chord = np.linalg.norm(end - start)
        if chord == 0:
            return min(5, n)

        max_dist = -np.inf
        elbow_k = 10  # default fallback

        for i in range(1, n - 1):
            p = np.array([i, y[i]])

            # distance from point to line
            dist = np.abs(np.cross(end - start, p - start)) / chord

            if dist > max_dist:
                max_dist = dist
                elbow_k = i

        return elbow_k

    # -----------------------------------------------------------
    # Transform features for ONE pair
    # -----------------------------------------------------------
    def transform_pair(self, pair, X):
        """Return reduced feature matrix for the pair."""
        idx = self.selected_features[pair]
        Xs = self.scalers[pair].transform(X)
        return Xs[:, idx]

    # -----------------------------------------------------------
    # Fit + transform ALL pairs
    # -----------------------------------------------------------
    def fit_transform(self, features, targets):
        """
        features: dict {(A,B): np.array(T, F)}
        targets:  dict {(A,B): np.array(T)} (future return / spread change)

        returns:
            reduced_features dict {(A,B): np.array(T, K)}
        """
        reduced = {}

        for pair, X in features.items():
            y = targets[pair]

            idx = self.fit_pair(pair, X, y)
            reduced[pair] = X[:, idx]

        return reduced

    # -----------------------------------------------------------
    # Apply an already-fitted selector to new features
    # -----------------------------------------------------------
    def transform(self, features):
        reduced = {}
        for pair, X in features.items():
            reduced[pair] = self.transform_pair(pair, X)
        return reduced
