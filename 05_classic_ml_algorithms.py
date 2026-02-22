"""
Classic Machine Learning Algorithms
=====================================
These algorithms form the foundation of ML and are still widely used.

Implemented from scratch with NumPy, then shown with scikit-learn (the
standard library for classical ML in Python).

Covered:
  1. Linear Regression       — predict continuous values
  2. Logistic Regression     — binary classification
  3. K-Nearest Neighbours    — instance-based learning
  4. Decision Tree           — rule-based, interpretable
  5. K-Means Clustering      — unsupervised grouping
"""

import numpy as np
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LINEAR REGRESSION (closed-form + gradient descent)
# ═══════════════════════════════════════════════════════════════════════════════

class LinearRegression:
    """
    Fits: y = Xw + b   by minimising MSE loss.

    Closed-form solution: w = (X^T X)^{-1} X^T y
    or via gradient descent (preferred when X is large).
    """

    def __init__(self, lr=0.01, epochs=1000, method="gd"):
        self.lr = lr
        self.epochs = epochs
        self.method = method   # "gd" = gradient descent, "normal" = closed-form
        self.w = self.b = None

    def fit(self, X, y):
        n, d = X.shape
        if self.method == "normal":
            # Add bias column, solve analytically
            Xb = np.hstack([X, np.ones((n, 1))])
            # w = pinv(X) @ y  (more numerically stable than matrix inverse)
            params = np.linalg.pinv(Xb) @ y
            self.w, self.b = params[:-1], params[-1]
        else:
            self.w = np.zeros(d)
            self.b = 0.0
            for _ in range(self.epochs):
                pred = X @ self.w + self.b
                err  = pred - y
                self.w -= self.lr * (2 / n) * X.T @ err
                self.b -= self.lr * (2 / n) * err.sum()

    def predict(self, X):
        return X @ self.w + self.b

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticRegression:
    """
    Binary classifier using the sigmoid function.
    Minimises Binary Cross-Entropy via gradient descent.

    P(y=1|x) = σ(w·x + b)
    """

    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.w = self.b = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * X.T @ err / n
            self.b -= self.lr * err.mean()

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. K-NEAREST NEIGHBOURS (KNN)
# ═══════════════════════════════════════════════════════════════════════════════

class KNearestNeighbours:
    """
    No training phase — stores training data, classifies by majority vote
    among the k closest neighbours (Euclidean distance).

    Simple but O(n) per prediction — slow on large datasets.
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_idx]
            predictions.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(predictions)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DECISION TREE (ID3-style, classification)
# ═══════════════════════════════════════════════════════════════════════════════

class DecisionTree:
    """
    Recursively partitions data by the feature/threshold that best reduces
    Gini impurity. Greedy, axis-aligned splits.

    Gini(t) = 1 - Σ p_k²    (0 = pure node, 0.5 = maximum impurity for binary)
    """

    class _Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature   = feature    # split feature index
            self.threshold = threshold  # split threshold value
            self.left      = left       # left subtree  (x <= threshold)
            self.right     = right      # right subtree (x > threshold)
            self.value     = value      # leaf prediction (not None ⟹ leaf)

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    @staticmethod
    def _gini(y):
        counts = np.bincount(y)
        probs  = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)
        n = len(y)
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left  = y[X[:, feat] <= t]
                right = y[X[:, feat] >  t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_gini - (len(left)/n * self._gini(left) +
                                      len(right)/n * self._gini(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        # Stopping criteria
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return self._Node(value=Counter(y).most_common(1)[0][0])
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return self._Node(value=Counter(y).most_common(1)[0][0])
        mask  = X[:, feat] <= thresh
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return self._Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(X, y.astype(int), depth=0)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. K-MEANS CLUSTERING (unsupervised)
# ═══════════════════════════════════════════════════════════════════════════════

class KMeans:
    """
    Partitions data into k clusters by iteratively:
      1. Assigning each point to the nearest centroid
      2. Moving centroids to the mean of assigned points

    Converges when assignments no longer change.
    """

    def __init__(self, k=3, max_iters=300, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        # K-Means++ initialisation — better than random for avoiding bad starts
        rng = np.random.default_rng(42)
        centroids = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            dists = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            probs = dists ** 2 / (dists ** 2).sum()
            centroids.append(X[rng.choice(len(X), p=probs)])
        self.centroids = np.array(centroids)

        for _ in range(self.max_iters):
            labels = self._assign(X)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.k)])
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        return self

    def _assign(self, X):
        dists = np.stack([np.linalg.norm(X - c, axis=1) for c in self.centroids], axis=1)
        return np.argmin(dists, axis=1)

    def predict(self, X):
        return self._assign(X)

    def inertia(self, X):
        """Sum of squared distances to nearest centroid (lower = tighter clusters)."""
        labels = self.predict(X)
        return sum(np.linalg.norm(X[labels == k] - self.centroids[k]) ** 2
                   for k in range(self.k))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    # ── Regression
    print("=" * 50)
    print("LINEAR REGRESSION")
    X_r = np.random.randn(200, 2)
    y_r = 3 * X_r[:, 0] - 2 * X_r[:, 1] + np.random.randn(200) * 0.5
    lr_model = LinearRegression(lr=0.05, epochs=500, method="gd")
    lr_model.fit(X_r, y_r)
    print(f"  Weights: {lr_model.w}  (true: [3, -2])")
    print(f"  MSE: {lr_model.mse(X_r, y_r):.4f}")

    # ── Classification
    print("\nLOGISTIC REGRESSION + KNN + DECISION TREE")
    X_c, y_c = make_classification(n_samples=500, n_features=4, n_informative=2, random_state=0)
    X_c = StandardScaler().fit_transform(X_c)
    X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=0)

    for name, clf in [
        ("Logistic",     LogisticRegression(lr=0.2, epochs=300)),
        ("KNN(k=5)",     KNearestNeighbours(k=5)),
        ("DecisionTree", DecisionTree(max_depth=5)),
    ]:
        clf.fit(X_tr, y_tr)
        acc = (clf.predict(X_te) == y_te).mean()
        print(f"  {name:14s} accuracy: {acc:.4f}")

    # ── Clustering
    print("\nK-MEANS CLUSTERING")
    X_blob, y_blob = make_blobs(n_samples=300, centers=3, random_state=42)
    km = KMeans(k=3)
    km.fit(X_blob)
    labels = km.predict(X_blob)
    print(f"  Inertia: {km.inertia(X_blob):.2f}")
    print(f"  Centroids:\n{km.centroids}")

    # ── scikit-learn equivalents (production use)
    print("\n" + "=" * 50)
    print("scikit-learn equivalents (same results, production-ready):")
    from sklearn.linear_model import LinearRegression as SKLinReg, LogisticRegression as SKLogReg
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans as SKMeans

    sk_models = {
        "sklearn LogReg":  SKLogReg(max_iter=300).fit(X_tr, y_tr),
        "sklearn KNN":     KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr),
        "sklearn DTree":   DecisionTreeClassifier(max_depth=5).fit(X_tr, y_tr),
    }
    for name, m in sk_models.items():
        print(f"  {name:18s} accuracy: {m.score(X_te, y_te):.4f}")
