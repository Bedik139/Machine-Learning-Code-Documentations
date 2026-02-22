# Machine Learning Code Documentations

A from-scratch implementation of core machine learning algorithms and neural network building blocks using NumPy, with PyTorch equivalents and scikit-learn validation. Each file is self-contained, documented, and runnable.

---

## Repository Structure

| File | Topic | Key Concepts |
|------|-------|-------------|
| `01_neuron_and_layer.py` | Building Blocks | Single neuron, dense layer, activation functions, weight initialization |
| `02_neural_network.py` | Full Neural Network | Multi-layer perceptron, loss functions, backpropagation, mini-batch training |
| `03_optimizers.py` | Optimization Algorithms | SGD, Momentum, RMSProp, Adam, AdaGrad, visual comparison |
| `04_neural_network_pytorch.py` | PyTorch Implementation | BatchNorm, Dropout, learning rate scheduling, model persistence |
| `05_classic_ml_algorithms.py` | Classical ML | Linear/Logistic Regression, KNN, Decision Trees, K-Means |
| `06_backpropagation_visual.py` | Backprop Walkthrough | Manual forward/backward pass with PyTorch verification |

---

## File-by-File Breakdown

### `01_neuron_and_layer.py` — Building Blocks

This file implements the two most fundamental units of a neural network: a single neuron and a dense layer.

#### Activation Functions

Four activation functions are defined. These are nonlinear transformations applied after the linear computation in a neuron. Without them, stacking layers would collapse into one big linear transformation (since a linear function of a linear function is still linear).

- **Sigmoid** maps any real number to the range (0, 1) using the formula `1 / (1 + e^(-x))`. Useful when interpreting the output as a probability. Its derivative is `σ(x) · (1 - σ(x))`, needed during backpropagation.
- **ReLU** (Rectified Linear Unit) returns `max(0, x)`. If the input is negative, output is 0; if positive, output equals the input. It is the most commonly used hidden layer activation because it is cheap to compute and avoids the vanishing gradient problem that plagues sigmoid in deep networks. Its derivative is 1 for positive inputs, 0 for negative — essentially a step function.
- **Tanh** squashes to (-1, 1) instead of (0, 1). It is zero-centered, which sometimes helps optimization. Its derivative is `1 - tanh²(x)`.
- **Softmax** converts a vector of raw scores (logits) into a probability distribution where all outputs sum to 1. The `np.max` subtraction is a numerical stability trick: since `e^x` grows explosively, subtracting the max prevents overflow without changing the result (because `softmax(x) = softmax(x - c)` for any constant c).

#### The Neuron Class

A single neuron computes:

```
z = w · x + b       (linear combination)
a = activation(z)   (nonlinear transformation)
```

Where `w` is a weight vector (same dimension as the input), `b` is a scalar bias, and `x` is the input vector.

**Weight initialization** matters significantly. The code uses He initialization for ReLU (`scale = sqrt(2/n)`) and Xavier for others (`scale = sqrt(1/n)`). If weights are too large, activations explode through layers; too small, they vanish to zero. These scaling factors are derived mathematically to keep the variance of activations roughly constant across layers. He initialization specifically accounts for the fact that ReLU zeroes out roughly half the inputs, so a larger scale is needed to compensate.

**Forward pass** computes `z = dot(w, x) + b` then applies the activation. It caches `x` and `z` because backpropagation needs them later.

**Backward pass** receives `d_out` — the gradient of the loss with respect to this neuron's output. It computes three things using the chain rule:

- `dw` (gradient w.r.t. weights) — needed to update the weights
- `db` (gradient w.r.t. bias) — needed to update the bias
- `dx` (gradient w.r.t. inputs) — needed to pass the gradient backward to the previous layer

The math: `dz = d_out * activation'(z)` (chain rule through the activation), then `dw = dz * x`, `db = dz`, and `dx = dz * w`. Each follows directly from the partial derivatives of `z = w · x + b`.

#### The DenseLayer Class

This is the vectorized, batched version of multiple neurons working in parallel. Instead of one weight vector, there is a weight matrix `W` of shape `(n_inputs, n_outputs)` — each column is essentially one neuron's weight vector.

The forward pass computes `Z = X @ W + b` where `X` is a batch of inputs with shape `(batch_size, n_inputs)`. The `@` operator is matrix multiplication, computing all neurons' outputs for all samples simultaneously.

The backward pass follows the same chain rule logic but with matrix operations: `dW = X^T @ dZ` accumulates gradients across the batch, `db = sum(dZ, axis=0)` sums across the batch dimension, and `dX = dZ @ W^T` propagates gradients backward. These formulas come from the rules of matrix calculus.

---

### `02_neural_network.py` — Full Network

This builds on file 1 to create a complete trainable neural network.

#### Loss Functions

Loss functions measure how wrong the model's predictions are and provide the starting gradient for backpropagation.

**MSE (Mean Squared Error)**: `L = mean((pred - true)²)`. The gradient is `2(pred - true) / n`. Used for regression (predicting continuous values). It penalizes large errors quadratically, meaning outliers have outsized influence.

**Binary Cross-Entropy**: `L = -mean(y·log(p) + (1-y)·log(1-p))`. Derived from information theory and maximum likelihood estimation. When the true label is 1, the penalty is `-log(p)` — the further `p` is from 1, the larger the penalty, growing to infinity as `p` approaches 0. The `eps = 1e-9` clipping prevents `log(0)`. The gradient formula `(p - y) / (p(1-p)n)` is the derivative of this expression.

#### The NeuralNetwork Class

This chains multiple `Layer` objects together.

**Forward pass**: feed the output of each layer as input to the next. `x → Layer1 → Layer2 → ... → output`.

**Backward pass**: starting from the loss gradient, it walks through the layers in reverse order. Each layer's `backward()` returns `dx` (gradient for the previous layer), `dW`, and `db`. The `dx` from one layer becomes the `d_out` for the layer before it. This is the chain rule propagating through the entire network.

**Update** applies vanilla gradient descent: `W = W - lr * dW` for every layer. The learning rate `lr` controls the step size — too large and the optimizer overshoots; too small and training is very slow.

**The fit method** implements mini-batch training:

1. Shuffle the data each epoch (prevents the model from learning the order of examples)
2. Split into mini-batches of size `batch_size`
3. For each batch: forward pass → compute loss → backward pass → update weights
4. Track the average loss per epoch

Mini-batches are a compromise between full-batch gradient descent (uses all data per update — accurate but slow) and stochastic gradient descent (one sample per update — noisy but fast). Typical batch sizes are 16–128.

#### XOR Demo

XOR is the classic neural network test case. The four input-output pairs `(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0` are not linearly separable — no single straight line can separate the 0s from the 1s. This means a single neuron (which is a linear separator plus activation) cannot solve it. At least one hidden layer is needed, which is what the `[2, 8, 1]` architecture provides: 2 inputs, 8 hidden neurons, 1 output. The network learns to create an internal representation where XOR becomes linearly separable.

---

### `03_optimizers.py` — How Weights Get Updated

All optimizers solve the same problem: given a gradient (the direction of steepest increase in loss), update the parameters to decrease the loss. They differ in how they use gradient history.

#### SGD (Stochastic Gradient Descent)

```
param = param - lr * grad
```

The simplest possible optimizer. Move in the opposite direction of the gradient, scaled by the learning rate. It treats all parameters equally and has no memory — the update at step 1000 knows nothing about the gradients at steps 1–999. This makes it sensitive to learning rate choice and prone to oscillation in ravines (directions where the loss surface curves sharply).

#### SGD with Momentum

```
v = β·v + (1-β)·grad
param = param - lr·v
```

Adds a "velocity" vector that accumulates past gradients. Like a ball rolling downhill, it picks up speed in consistent directions and dampens oscillations in inconsistent directions. `β = 0.9` means 90% of the velocity carries over from the previous step. This helps the optimizer push through noisy gradients and accelerate in the right direction.

#### RMSProp

```
s = β·s + (1-β)·grad²
param = param - lr * grad / sqrt(s + ε)
```

Adapts the learning rate per parameter. `s` tracks the running average of squared gradients. Parameters with consistently large gradients get their effective learning rate reduced (divided by a large `sqrt(s)`), while parameters with small gradients get boosted. The `ε = 1e-8` prevents division by zero. This was invented to fix AdaGrad's problem of the learning rate shrinking to zero over time (since RMSProp uses an exponential moving average instead of a cumulative sum).

#### Adam (Adaptive Moment Estimation)

```
m = β1·m + (1-β1)·grad        (first moment — tracks mean gradient)
v = β2·v + (1-β2)·grad²       (second moment — tracks variance)
m̂ = m / (1 - β1^t)            (bias correction)
v̂ = v / (1 - β2^t)            (bias correction)
param = param - lr * m̂ / (sqrt(v̂) + ε)
```

Adam combines momentum (through `m`) with RMSProp (through `v`). The bias correction is critical and often overlooked. Since `m` and `v` are initialized to zero, they are biased toward zero in early steps. Dividing by `(1 - β^t)` corrects this — when `t=1` with `β1=0.9`, the correction factor is `1/0.1 = 10x`, which is significant. As `t` grows, `β^t → 0` and the correction vanishes. Adam is the default choice for most deep learning tasks because it works well across a wide range of problems without much tuning.

#### AdaGrad

```
G = G + grad²
param = param - lr * grad / sqrt(G + ε)
```

Similar idea to RMSProp but `G` never decays — it monotonically increases. This means the effective learning rate always shrinks. Good for sparse data (NLP, recommender systems) where large updates are desirable on rare features and small updates on frequent ones. Bad for dense problems because it eventually stops learning entirely.

#### Rosenbrock Demo

The Rosenbrock function `f(x,y) = (1-x)² + 100(y-x²)²` is a standard optimization benchmark with a global minimum at (1, 1) sitting at the bottom of a narrow, curved valley. The results show that Adam and RMSProp converge much closer to (1, 1) than plain SGD, which barely moves, demonstrating why adaptive learning rates matter in practice. A comparison plot is saved as `optimizer_comparison.png`.

---

### `04_neural_network_pytorch.py` — PyTorch Version

This reimplements the concepts from files 1–3 using PyTorch, where backpropagation is handled automatically.

#### The MLP Class

The constructor builds a network dynamically from a list of hidden dimensions. For each hidden layer, it stacks four components:

- **nn.Linear**: the `z = Wx + b` computation with learnable parameters.
- **nn.BatchNorm1d**: normalizes activations within each mini-batch to have zero mean and unit variance, then applies a learnable scale and shift. This addresses "internal covariate shift" — as earlier layers change during training, the distribution of inputs to later layers shifts, making training unstable. BatchNorm smooths the loss landscape and allows higher learning rates.
- **nn.ReLU**: activation function.
- **nn.Dropout**: during training, randomly sets a fraction of neurons to zero. This forces the network to not rely on any single neuron, acting as regularization against overfitting. During evaluation, dropout is disabled and activations are scaled accordingly.

`nn.Sequential` chains these into a pipeline. The `forward` method calls `self.net(x)` — PyTorch handles the rest.

#### ScaledDotProductAttention

A bonus implementation of the core mechanism of Transformers (GPT, BERT, etc.). The formula `softmax(QK^T / √d_k) · V` computes attention weights by:

1. `QK^T` — dot product between queries and keys measures similarity.
2. `/ √d_k` — scaling prevents dot products from growing too large (which would make softmax outputs very peaked, causing vanishing gradients).
3. `softmax` — normalizes to a probability distribution.
4. `· V` — weighted combination of values based on attention weights.

The optional `mask` is for autoregressive models (like GPT) where position `i` should not attend to positions `j > i`.

#### Training Loop

The PyTorch training loop follows a strict pattern:

1. `optimizer.zero_grad()` — clear accumulated gradients (PyTorch accumulates by default).
2. `preds = model(X_batch)` — forward pass.
3. `loss = loss_fn(preds, y_batch)` — compute loss.
4. `loss.backward()` — PyTorch's autograd computes all gradients automatically by tracing the computation graph backward.
5. `optimizer.step()` — update all parameters.

This replaces the entire manual backward pass and update code from file 2.

#### The Demo

Generates a binary classification problem: two Gaussian blobs centered at (2, 2) and (-2, -2). Uses `BCEWithLogitsLoss` which combines sigmoid + BCE in a numerically stable way (avoids computing `log(sigmoid(x))` directly, which can underflow). The `StepLR` scheduler halves the learning rate every 20 epochs — starting with larger steps for fast progress, then finer steps for precision. Model weights are saved with `torch.save` for later reuse.

---

### `05_classic_ml_algorithms.py` — Classical ML

#### Linear Regression

Fits `y = Xw + b` by minimizing MSE. Two methods:

**Normal equation** (closed form): `w = (X^T X)^{-1} X^T y`. Directly computes the optimal weights in one step using linear algebra. `np.linalg.pinv` computes the pseudoinverse, which is more numerically stable than explicitly computing the inverse (handles rank-deficient matrices gracefully). Works well for small datasets but is O(d³) in the number of features, making it impractical when d is large.

**Gradient descent**: iteratively updates `w -= lr * (2/n) * X^T @ (pred - y)`. This gradient comes from differentiating `MSE = (1/n) Σ(Xw + b - y)²` with respect to `w`. Scales better to large datasets since each step is O(nd).

#### Logistic Regression

Despite the name, this is a classifier. It models the probability of class 1 as `P(y=1|x) = σ(w·x + b)`. The sigmoid squashes the linear output to [0, 1], interpretable as a probability. Training minimizes binary cross-entropy via gradient descent. The gradient `X^T @ (p - y) / n` has a beautifully simple form — the sigmoid and log-loss derivatives cancel out.

#### K-Nearest Neighbours

KNN is fundamentally different from the other algorithms: it has no training phase. It stores the data. At prediction time, for each test point, it computes the Euclidean distance to every training point, finds the k closest ones, and takes a majority vote. This makes "training" O(1) but prediction O(n·d) per sample, which is why KNN does not scale well. The `k` parameter controls the bias-variance tradeoff: small k = low bias, high variance (sensitive to noise); large k = high bias, low variance (over-smoothed boundaries).

#### Decision Tree

Implements a CART-style classifier using Gini impurity as the splitting criterion. `Gini(t) = 1 - Σ p_k²` measures how "impure" a node is — 0 means all samples belong to one class (pure), 0.5 is maximum impurity for binary classification.

The algorithm greedily searches for the best feature and threshold to split on at each node. "Best" means the split that maximally reduces impurity, measured by information gain: `parent_gini - weighted_average(child_ginis)`. It tries every unique value of every feature as a potential threshold, making it O(n·d·unique_values) per split.

The tree grows recursively until hitting stopping criteria: max depth, minimum samples per node, or pure nodes. Without stopping criteria, a decision tree will perfectly fit the training data (zero training error) by creating one leaf per sample — this is severe overfitting.

#### K-Means Clustering

This is unsupervised — there are no labels. The goal is to partition data into k groups where points within each group are close to each other.

**K-Means++ initialization** is used instead of random initialization. Random centroids can lead to poor convergence (e.g., two centroids in the same cluster). K-Means++ selects the first centroid randomly, then each subsequent centroid is chosen with probability proportional to its squared distance from the nearest existing centroid. This spreads initial centroids apart and provably leads to an O(log k) competitive approximation to the optimal clustering.

The main loop alternates between two steps: (1) assign each point to its nearest centroid, (2) recompute centroids as the mean of assigned points. This is guaranteed to converge because each step reduces or maintains the total inertia (sum of squared distances). However, it only finds a local minimum, not necessarily the global one.

**Inertia** is the evaluation metric: total squared distance from each point to its assigned centroid. Lower is better (tighter clusters).

#### Sklearn Comparison

The demo runs both the scratch implementations and sklearn equivalents on the same data. Identical accuracies (0.85, 0.91, 0.86) validate the scratch implementations. In production, sklearn is preferred because it is optimized, handles edge cases, and provides additional functionality (cross-validation, grid search, etc.).

---

### `06_backpropagation_visual.py` — Manual Backprop Walkthrough

This file is pedagogical — it manually computes every single number in a forward and backward pass through a tiny 2-layer network with 1 input, 1 hidden unit, and 1 output.

#### Forward Pass

Starting with input `x = 2.0`:

- **Layer 1**: `z1 = 0.5 · 2.0 + 0.1 = 1.1`, then `a1 = ReLU(1.1) = 1.1` (positive, so ReLU passes it through unchanged).
- **Layer 2**: `z2 = -0.3 · 1.1 + 0.2 = -0.13`, then `a2 = σ(-0.13) = 0.4675` (predicted probability).
- **BCE Loss** = 0.7603 (relatively high since the prediction 0.47 is far from the target 1.0).

#### Backward Pass

The chain rule propagates gradients right-to-left:

1. `dL/da2 = -2.1388` — the loss signals the output should be higher (negative gradient means "increase this").
2. `dL/dz2 = dL/da2 · σ'(z2) = -2.1388 · 0.2489 = -0.5325` — chain through sigmoid derivative.
3. From `z2 = w2·a1 + b2`, partial derivatives give:
   - `dL/dw2 = dL/dz2 · a1 = -0.5857` — w2 should increase (make output larger).
   - `dL/db2 = dL/dz2 = -0.5325` — bias should increase too.
   - `dL/da1 = dL/dz2 · w2 = 0.1597` — this propagates backward to layer 1.
4. Through layer 1's ReLU (derivative = 1 since z1 > 0): `dL/dz1 = 0.1597`.
5. `dL/dw1 = 0.3195`, `dL/db1 = 0.1597`.

#### Weight Update

With `lr = 0.1`, the weights shift to reduce the loss. For instance, `w2` goes from -0.3 to -0.2414 — moving in the positive direction as the gradient suggested, which will make the output larger and closer to the target of 1.0.

The PyTorch verification section confirms these manual computations match autograd's results exactly, validating the correctness of the math.

---

## The Big Picture

These 6 files tell a coherent story. File 1 builds a single brick (neuron) and a wall (layer). File 2 assembles walls into a building (network) and teaches it to learn (backpropagation and training loop). File 3 explores smarter ways to learn (optimizers). File 4 shows how professionals do it (PyTorch). File 5 covers the classical algorithms that do not require neural networks. File 6 zooms back in to make sure the math is understood at the atomic level.

---

## Requirements

```
numpy
matplotlib
torch        # for files 04 and 06 (verification)
scikit-learn # for file 05 (validation)
```

## Usage

Each file is self-contained and runnable:

```bash
python 01_neuron_and_layer.py
python 02_neural_network.py
python 03_optimizers.py
python 04_neural_network_pytorch.py
python 05_classic_ml_algorithms.py
python 06_backpropagation_visual.py
```
