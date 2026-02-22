"""
Neural Network Building Blocks: Neuron and Layer
=================================================
A neuron is the fundamental unit of a neural network.
It computes: output = activation(weights · inputs + bias)

Libraries: NumPy (the standard for numerical computing in Python)
"""

import numpy as np


# ── Activation Functions ──────────────────────────────────────────────────────

def sigmoid(x):
    """Squashes input to (0, 1). Used in binary classification output layers."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """Rectified Linear Unit. Most common activation for hidden layers."""
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    """Hyperbolic tangent. Squashes to (-1, 1)."""
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    """Converts logits to a probability distribution. Used for multi-class output."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))  # subtract max for numerical stability
    return e / e.sum(axis=-1, keepdims=True)


# ── Single Neuron ─────────────────────────────────────────────────────────────

class Neuron:
    """
    A single artificial neuron.

    Computes: z = w · x + b,  a = activation(z)

    Parameters
    ----------
    n_inputs : int   — number of input features
    activation : str — 'sigmoid' | 'relu' | 'tanh'
    """

    ACTIVATIONS = {
        "sigmoid": (sigmoid, sigmoid_derivative),
        "relu":    (relu,    relu_derivative),
        "tanh":    (tanh,    tanh_derivative),
    }

    def __init__(self, n_inputs: int, activation: str = "relu"):
        # He initialisation for relu, Xavier for others
        scale = np.sqrt(2 / n_inputs) if activation == "relu" else np.sqrt(1 / n_inputs)
        self.w = np.random.randn(n_inputs) * scale   # weight vector
        self.b = 0.0                                  # bias scalar
        self.act, self.act_d = self.ACTIVATIONS[activation]

        # cache for backprop
        self._x = None
        self._z = None

    def forward(self, x: np.ndarray) -> float:
        self._x = x
        self._z = np.dot(self.w, x) + self.b
        return self.act(self._z)

    def backward(self, d_out: float) -> np.ndarray:
        """
        d_out : gradient of loss w.r.t. this neuron's output
        Returns gradient w.r.t. inputs (for chaining layers).
        """
        dz = d_out * self.act_d(self._z)   # chain rule through activation
        dw = dz * self._x                  # gradient w.r.t. weights
        db = dz                            # gradient w.r.t. bias
        dx = dz * self.w                   # gradient w.r.t. inputs
        return dx, dw, db


# ── Dense (Fully-Connected) Layer ─────────────────────────────────────────────

class DenseLayer:
    """
    A layer of neurons — each output connects to every input.

    Shape: (batch, n_inputs) → (batch, n_outputs)
    """

    def __init__(self, n_inputs: int, n_outputs: int, activation: str = "relu"):
        scale = np.sqrt(2 / n_inputs) if activation == "relu" else np.sqrt(1 / n_inputs)
        self.W = np.random.randn(n_inputs, n_outputs) * scale  # weight matrix
        self.b = np.zeros(n_outputs)                           # bias vector

        self.act, self.act_d = Neuron.ACTIVATIONS[activation]

        self._x = None
        self._z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: (batch, n_inputs)"""
        self._x = x
        self._z = x @ self.W + self.b          # matrix multiply
        return self.act(self._z)

    def backward(self, d_out: np.ndarray):
        """
        d_out shape: (batch, n_outputs)
        Returns (d_inputs, dW, db)
        """
        dz = d_out * self.act_d(self._z)       # element-wise through activation
        dW = self._x.T @ dz                    # (n_inputs, n_outputs)
        db = dz.sum(axis=0)                    # (n_outputs,)
        dx = dz @ self.W.T                     # (batch, n_inputs)
        return dx, dW, db


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("=== Single Neuron ===")
    neuron = Neuron(n_inputs=3, activation="relu")
    x = np.array([0.5, -1.2, 0.8])
    out = neuron.forward(x)
    print(f"Input:  {x}")
    print(f"Output: {out:.4f}")

    dx, dw, db = neuron.backward(d_out=1.0)
    print(f"dW: {dw},  db: {db:.4f}")

    print("\n=== Dense Layer (batch of 4 samples) ===")
    layer = DenseLayer(n_inputs=3, n_outputs=2, activation="sigmoid")
    X = np.random.randn(4, 3)
    out = layer.forward(X)
    print(f"Output shape: {out.shape}")
    print(out)
