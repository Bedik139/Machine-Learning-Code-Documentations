"""
Optimization Algorithms
=======================
Optimizers determine HOW the model's weights are updated using gradients.

Covered:
  1. SGD (Stochastic Gradient Descent)
  2. SGD with Momentum
  3. RMSProp
  4. Adam  ← most commonly used in practice
  5. AdaGrad

All optimizers follow the same interface:
    optimizer.step(param, grad)  →  updated param
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Base class ────────────────────────────────────────────────────────────────

class Optimizer:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ── 1. SGD ────────────────────────────────────────────────────────────────────

class SGD(Optimizer):
    """
    param ← param - lr * grad

    Simple but sensitive to learning rate. No memory of past gradients.
    """
    def step(self, param, grad):
        return param - self.lr * grad


# ── 2. SGD with Momentum ──────────────────────────────────────────────────────

class SGDMomentum(Optimizer):
    """
    Accumulates a velocity vector to dampen oscillations and speed convergence.

    v ← β·v + (1-β)·grad
    param ← param - lr·v

    β (beta) ≈ 0.9 is standard.
    """
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def step(self, param, grad):
        if self.v is None:
            self.v = np.zeros_like(param)
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return param - self.lr * self.v


# ── 3. RMSProp ────────────────────────────────────────────────────────────────

class RMSProp(Optimizer):
    """
    Adapts the learning rate per parameter using a running average of squared gradients.

    s ← β·s + (1-β)·grad²
    param ← param - lr / sqrt(s + ε) · grad

    Prevents the learning rate from vanishing (AdaGrad problem).
    """
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.s = None

    def step(self, param, grad):
        if self.s is None:
            self.s = np.zeros_like(param)
        self.s = self.beta * self.s + (1 - self.beta) * grad ** 2
        return param - self.lr * grad / (np.sqrt(self.s) + self.eps)


# ── 4. Adam (Adaptive Moment Estimation) ─────────────────────────────────────

class Adam(Optimizer):
    """
    Combines Momentum + RMSProp. Currently the default choice for most tasks.

    m ← β1·m + (1-β1)·grad          (1st moment — mean)
    v ← β2·v + (1-β2)·grad²         (2nd moment — uncentred variance)

    Bias correction (critical for early steps):
    m̂ = m / (1-β1^t)
    v̂ = v / (1-β2^t)

    param ← param - lr · m̂ / (sqrt(v̂) + ε)

    Defaults: β1=0.9, β2=0.999, ε=1e-8, lr=0.001
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, param, grad):
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)   # bias correction
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── 5. AdaGrad ────────────────────────────────────────────────────────────────

class AdaGrad(Optimizer):
    """
    Accumulates ALL past squared gradients — learning rate shrinks over time.
    Good for sparse data; can stop learning too early on dense problems.

    G ← G + grad²
    param ← param - lr / sqrt(G + ε) · grad
    """
    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.G = None

    def step(self, param, grad):
        if self.G is None:
            self.G = np.zeros_like(param)
        self.G += grad ** 2
        return param - self.lr * grad / (np.sqrt(self.G) + self.eps)


# ── Demo: minimise the Rosenbrock function ────────────────────────────────────
# f(x,y) = (1-x)² + 100(y-x²)²   — global minimum at (1, 1)

def rosenbrock(p):
    x, y = p
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(p):
    x, y = p
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])


if __name__ == "__main__":
    start = np.array([-1.5, 2.0])
    optimizers = {
        "SGD":          SGD(lr=0.001),
        "Momentum":     SGDMomentum(lr=0.001, beta=0.9),
        "RMSProp":      RMSProp(lr=0.005),
        "Adam":         Adam(lr=0.01),
        "AdaGrad":      AdaGrad(lr=0.1),
    }

    steps = 2000
    histories = {}

    for name, opt in optimizers.items():
        p = start.copy()
        losses = []
        for _ in range(steps):
            g = rosenbrock_grad(p)
            p = opt.step(p, g)
            losses.append(rosenbrock(p))
        histories[name] = losses
        print(f"{name:12s}  final loss: {losses[-1]:.6f}  param: {p}")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, losses in histories.items():
        ax.semilogy(losses, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Optimizer Comparison on Rosenbrock Function")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=150)
    print("\nPlot saved to optimizer_comparison.png")
