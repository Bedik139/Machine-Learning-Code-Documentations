"""
Neural Network with PyTorch
===========================
PyTorch is THE standard framework for deep learning research and production.
It handles:
  - Automatic differentiation (autograd) — no manual backprop
  - GPU acceleration via CUDA
  - Pre-built layers, loss functions, optimizers

This file shows the same concepts as 01 & 02 but using PyTorch's real API.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ── 1. Defining a Neural Network ──────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron using PyTorch.

    nn.Module is the base class for ALL neural networks in PyTorch.
    You must implement forward() — backward() is handled automatically.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout=0.0):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),   # fully-connected layer (weights + bias)
                nn.BatchNorm1d(h),    # normalise activations — stabilises training
                nn.ReLU(),            # activation
                nn.Dropout(dropout),  # randomly zero neurons — prevents overfitting
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)   # chains layers in order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 2. Custom Layer Example ───────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """
    The core operation of the Transformer architecture.

    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    Q = queries, K = keys, V = values, d_k = key dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_k = d_model ** 0.5

    def forward(self, Q, K, V, mask=None):
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.d_k   # (batch, seq, seq)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, V), weights


# ── 3. Training Loop ──────────────────────────────────────────────────────────

def train(model, dataloader, optimizer, loss_fn, epochs=20, device="cpu"):
    model.to(device)
    model.train()
    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()        # clear gradients from last step
            preds = model(X_batch)       # forward pass
            loss = loss_fn(preds, y_batch)
            loss.backward()              # compute gradients (autograd)
            optimizer.step()             # update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        if epoch % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch:3d}/{epochs}  loss: {avg_loss:.6f}")

    return history


@torch.no_grad()  # disable gradient tracking for inference
def evaluate(model, X_tensor, y_tensor, device="cpu"):
    model.eval()
    model.to(device)
    preds = model(X_tensor.to(device))
    # for binary classification
    predicted = (torch.sigmoid(preds) > 0.5).float().cpu()
    accuracy = (predicted == y_tensor).float().mean().item()
    return accuracy


# ── 4. Available PyTorch optimizers (reference) ───────────────────────────────

"""
torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)   ← Adam + proper L2 reg
torch.optim.RMSprop(params, lr=1e-3, alpha=0.99)

Learning rate schedulers:
  torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
"""


# ── 5. Demo: Binary classification on synthetic data ─────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate two Gaussian blobs
    n = 500
    X_pos = np.random.randn(n // 2, 2) + np.array([2, 2])
    X_neg = np.random.randn(n // 2, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float32).reshape(-1, 1)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    # Build model
    model = MLP(input_dim=2, hidden_dims=[16, 8], output_dim=1, dropout=0.1)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # AdamW + BCE loss is a common pairing
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn   = nn.BCEWithLogitsLoss()   # numerically stable sigmoid + BCE

    # Learning rate scheduler: halve LR every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print("\nTraining...")
    for epoch in range(1, 51):
        total_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}  loss: {total_loss/len(loader):.4f}  lr: {scheduler.get_last_lr()[0]:.6f}")

    acc = evaluate(model, X_t, y_t)
    print(f"\nFinal accuracy: {acc:.4f}")

    # Save / load model weights
    torch.save(model.state_dict(), "mlp_weights.pt")
    print("Weights saved to mlp_weights.pt")

    # Reload:
    # model2 = MLP(input_dim=2, hidden_dims=[16, 8], output_dim=1)
    # model2.load_state_dict(torch.load("mlp_weights.pt"))
