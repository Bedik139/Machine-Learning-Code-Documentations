"""
Neural Network (Multi-Layer Perceptron) — built from scratch with NumPy
=======================================================================
Demonstrates:
  - Forward pass
  - Loss computation (MSE, Cross-Entropy)
  - Backpropagation (automatic gradient chaining through layers)
  - Mini-batch training loop
"""

import numpy as np
from typing import List


# ── Loss Functions ────────────────────────────────────────────────────────────

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray):
    """Mean Squared Error — used for regression."""
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2 * diff / y_true.size
    return loss, grad

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray):
    """Binary Cross-Entropy — used for binary classification."""
    eps = 1e-9  # prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)
    return loss, grad


# ── Activation helpers ────────────────────────────────────────────────────────

def relu(x):         return np.maximum(0, x)
def relu_d(x):       return (x > 0).astype(float)
def sigmoid(x):      return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_d(x):    s = sigmoid(x); return s * (1 - s)
def linear(x):       return x
def linear_d(x):     return np.ones_like(x)

ACTIVATIONS = {
    "relu":    (relu,    relu_d),
    "sigmoid": (sigmoid, sigmoid_d),
    "linear":  (linear,  linear_d),
}


# ── Layer ─────────────────────────────────────────────────────────────────────

class Layer:
    def __init__(self, n_in, n_out, activation="relu"):
        scale = np.sqrt(2 / n_in) if activation == "relu" else np.sqrt(1 / n_in)
        self.W  = np.random.randn(n_in, n_out) * scale
        self.b  = np.zeros(n_out)
        self.act, self.act_d = ACTIVATIONS[activation]
        self._x = self._z = None

    def forward(self, x):
        self._x = x
        self._z = x @ self.W + self.b
        return self.act(self._z)

    def backward(self, d_out):
        dz = d_out * self.act_d(self._z)
        dW = self._x.T @ dz
        db = dz.sum(axis=0)
        dx = dz @ self.W.T
        return dx, dW, db


# ── Neural Network ────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    A fully-connected feed-forward neural network.

    Example
    -------
    net = NeuralNetwork(layer_sizes=[2, 16, 8, 1],
                        activations=["relu", "relu", "sigmoid"])
    """

    def __init__(self, layer_sizes: List[int], activations: List[str]):
        assert len(activations) == len(layer_sizes) - 1
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            for i in range(len(activations))
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray):
        """Backpropagate loss gradient through all layers (right to left)."""
        grad = loss_grad
        grads = []
        for layer in reversed(self.layers):
            grad, dW, db = layer.backward(grad)
            grads.append((dW, db))
        return list(reversed(grads))

    def update(self, grads, lr: float):
        """Vanilla gradient descent weight update."""
        for layer, (dW, db) in zip(self.layers, grads):
            layer.W -= lr * dW
            layer.b -= lr * db

    def fit(self, X, y, epochs=200, lr=0.01, batch_size=32, loss_fn=mse_loss, verbose=True):
        history = []
        n = len(X)
        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            epoch_loss = 0
            for start in range(0, n, batch_size):
                Xb = X[start:start + batch_size]
                yb = y[start:start + batch_size]

                # Forward
                pred = self.forward(Xb)
                loss, grad = loss_fn(pred, yb)
                epoch_loss += loss

                # Backward + update
                grads = self.backward(grad)
                self.update(grads, lr)

            epoch_loss /= (n // batch_size)
            history.append(epoch_loss)

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:4d}/{epochs}  loss: {epoch_loss:.6f}")

        return history

    def predict(self, X):
        return self.forward(X)


# ── Demo: XOR problem ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)

    # XOR is not linearly separable — needs at least 1 hidden layer
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)

    net = NeuralNetwork(
        layer_sizes=[2, 8, 1],
        activations=["relu", "sigmoid"]
    )

    print("Training on XOR (should reach near-zero loss)\n")
    net.fit(X, y, epochs=1000, lr=0.1, batch_size=4, loss_fn=binary_cross_entropy)

    print("\nPredictions:")
    preds = net.predict(X)
    for xi, yi, pi in zip(X, y, preds):
        print(f"  {xi} → target={int(yi[0])}  pred={pi[0]:.4f}")
