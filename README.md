# Machine Learning Code Documentation

Python implementations of core ML concepts using the standard tools.

## Files

| File | Concepts |
|------|----------|
| `01_neuron_and_layer.py` | Artificial neuron, activation functions (ReLU, sigmoid, tanh, softmax), dense layer, forward/backward pass |
| `02_neural_network.py` | Multi-layer perceptron from scratch, loss functions, full training loop, XOR demo |
| `03_optimizers.py` | SGD, Momentum, RMSProp, **Adam**, AdaGrad — all from scratch with visual comparison |
| `04_neural_network_pytorch.py` | Same MLP using PyTorch (real-world API), BatchNorm, Dropout, LR scheduler, save/load |
| `05_classic_ml_algorithms.py` | Linear Regression, Logistic Regression, KNN, Decision Tree, K-Means — from scratch + scikit-learn |
| `06_backpropagation_visual.py` | Backprop chain rule shown step-by-step on a tiny network, verified with PyTorch autograd |

## Libraries Used

| Library | Role |
|---------|------|
| **NumPy** | All numerical operations, arrays, linear algebra |
| **PyTorch** | Deep learning framework (autograd, GPU, pre-built layers) |
| **scikit-learn** | Classical ML, datasets, preprocessing, evaluation |
| **Matplotlib** | Plotting (optimizer comparison) |

## Install

```bash
pip install numpy torch scikit-learn matplotlib
```

## Run

```bash
python 01_neuron_and_layer.py
python 02_neural_network.py
python 03_optimizers.py
python 04_neural_network_pytorch.py
python 05_classic_ml_algorithms.py
python 06_backpropagation_visual.py
```
