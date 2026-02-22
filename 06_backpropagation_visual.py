"""
Backpropagation — Step by Step
================================
Backprop is the algorithm that computes gradients through the network
by applying the CHAIN RULE from calculus, right to left.

This file walks through a tiny 2-layer network manually so you can
see exactly what each number represents.

Network: x → [Linear → ReLU] → [Linear → Sigmoid] → Loss (BCE)
"""

import numpy as np


def sigmoid(z):    return 1 / (1 + np.exp(-z))
def relu(z):       return np.maximum(0, z)
def relu_d(z):     return (z > 0).astype(float)


# ── Toy network (1 hidden unit, 1 output unit, 1 input feature) ───────────────

np.random.seed(7)

# Input
x     = np.array([2.0])     # single feature
y_true = np.array([1.0])    # binary label

# Layer 1 weights (1 input → 1 hidden unit)
w1, b1 = np.array([0.5]),  0.1
# Layer 2 weights (1 hidden → 1 output unit)
w2, b2 = np.array([-0.3]), 0.2

print("=" * 60)
print("FORWARD PASS")
print("=" * 60)

# Layer 1
z1 = w1 * x + b1
a1 = relu(z1)
print(f"z1 = w1·x + b1 = {w1[0]}·{x[0]} + {b1} = {z1[0]:.4f}")
print(f"a1 = ReLU(z1)  = {a1[0]:.4f}")

# Layer 2
z2 = w2 * a1 + b2
a2 = sigmoid(z2)
print(f"\nz2 = w2·a1 + b2 = {w2[0]}·{a1[0]:.4f} + {b2} = {z2[0]:.4f}")
print(f"a2 = σ(z2)      = {a2[0]:.4f}  ← predicted probability")

# Loss: Binary Cross-Entropy
loss = -(y_true * np.log(a2 + 1e-9) + (1 - y_true) * np.log(1 - a2 + 1e-9))
print(f"\nLoss (BCE) = {loss[0]:.4f}  (y_true={y_true[0]})")

print("\n" + "=" * 60)
print("BACKWARD PASS (chain rule, right to left)")
print("=" * 60)

# ∂Loss/∂a2  — gradient of BCE loss w.r.t. sigmoid output
dL_da2 = -(y_true / (a2 + 1e-9)) + (1 - y_true) / (1 - a2 + 1e-9)
print(f"\ndL/da2 = {dL_da2[0]:.4f}")

# ∂a2/∂z2   — sigmoid derivative: σ(z)·(1-σ(z))
da2_dz2 = a2 * (1 - a2)
dL_dz2 = dL_da2 * da2_dz2
print(f"dL/dz2 = dL/da2 · da2/dz2 = {dL_da2[0]:.4f} · {da2_dz2[0]:.4f} = {dL_dz2[0]:.4f}")

# ∂z2/∂w2 = a1,  ∂z2/∂b2 = 1,  ∂z2/∂a1 = w2
dL_dw2 = dL_dz2 * a1
dL_db2 = dL_dz2
dL_da1 = dL_dz2 * w2
print(f"\ndL/dw2 = dL/dz2 · a1    = {dL_dz2[0]:.4f} · {a1[0]:.4f} = {dL_dw2[0]:.4f}  ← update w2")
print(f"dL/db2 = dL/dz2          = {dL_db2[0]:.4f}              ← update b2")
print(f"dL/da1 = dL/dz2 · w2    = {dL_dz2[0]:.4f} · {w2[0]:.4f} = {dL_da1[0]:.4f}  ← pass back")

# ∂a1/∂z1 — ReLU derivative
da1_dz1 = relu_d(z1)
dL_dz1  = dL_da1 * da1_dz1
dL_dw1  = dL_dz1 * x
dL_db1  = dL_dz1
print(f"\ndL/dz1 = dL/da1 · ReLU'(z1) = {dL_da1[0]:.4f} · {da1_dz1[0]:.4f} = {dL_dz1[0]:.4f}")
print(f"dL/dw1 = dL/dz1 · x         = {dL_dz1[0]:.4f} · {x[0]} = {dL_dw1[0]:.4f}  ← update w1")
print(f"dL/db1 = dL/dz1              = {dL_db1[0]:.4f}              ← update b1")

# ── One gradient descent step ─────────────────────────────────────────────────
lr = 0.1
print("\n" + "=" * 60)
print(f"WEIGHT UPDATE  (lr={lr})")
print("=" * 60)
print(f"w1: {w1[0]:.4f} → {(w1 - lr * dL_dw1)[0]:.4f}")
print(f"b1: {b1:.4f}   → {(b1 - lr * dL_db1[0]):.4f}")
print(f"w2: {w2[0]:.4f} → {(w2 - lr * dL_dw2)[0]:.4f}")
print(f"b2: {b2:.4f}   → {(b2 - lr * dL_db2[0]):.4f}")

# ── Verify with PyTorch autograd ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION: PyTorch autograd (should match our manual values)")
print("=" * 60)
try:
    import torch

    x_t  = torch.tensor([2.0], requires_grad=False)
    y_t  = torch.tensor([1.0])
    w1_t = torch.tensor([0.5], requires_grad=True)
    b1_t = torch.tensor([0.1], requires_grad=True)
    w2_t = torch.tensor([-0.3], requires_grad=True)
    b2_t = torch.tensor([0.2], requires_grad=True)

    z1_t = w1_t * x_t + b1_t
    a1_t = torch.relu(z1_t)
    z2_t = w2_t * a1_t + b2_t
    a2_t = torch.sigmoid(z2_t)
    loss_t = torch.nn.functional.binary_cross_entropy(a2_t, y_t)
    loss_t.backward()

    print(f"dL/dw1 = {w1_t.grad.item():.4f}  (manual: {dL_dw1[0]:.4f})")
    print(f"dL/db1 = {b1_t.grad.item():.4f}  (manual: {dL_db1[0]:.4f})")
    print(f"dL/dw2 = {w2_t.grad.item():.4f}  (manual: {dL_dw2[0]:.4f})")
    print(f"dL/db2 = {b2_t.grad.item():.4f}  (manual: {dL_db2[0]:.4f})")
except ImportError:
    print("(PyTorch not installed — skipping verification)")
