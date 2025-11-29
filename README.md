# MNIST Digit Classifier with Live Weight Visualization 🧠

A neural network built **from scratch** using only NumPy that classifies handwritten digits from the MNIST dataset, featuring **real-time weight heatmaps** that visualize how the network learns during training.

## ✨ Features

- 🔢 **3-layer neural network** (784 → 128 → 64 → 10)
- 📊 **Live weight visualization** - Watch the network's weights light up and adjust in real-time!
- 🎯 **93.61% test accuracy** - Competitive performance with no frameworks
- 📈 **Real-time metrics** - Loss and accuracy plots update during training
- 💾 **Model persistence** - Save and load trained models
- 🎨 **Sample predictions viewer** - See what the network got right and wrong

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy matplotlib
```

### 2. Run Training

```bash
python train.py
```

The script will:
- Download MNIST dataset automatically (~12 MB)
- Train the neural network for 10 epochs
- Show live weight heatmaps updating in real-time
- Save the best model to `best_model.pkl`
- Display sample predictions

### 3. Watch the Magic! ✨

As training progresses, you'll see:
- **Weight heatmaps** changing from random noise to structured patterns
- **Loss curve** decreasing smoothly
- **Accuracy** climbing from ~30% to 93%+

## 📊 What You'll See

The visualization window shows:

1. **Hidden Layer 1 Weights** (128 × 256 subset)
   - Blue = negative weights, Red = positive weights
   - Patterns emerge for edges, curves, and digit features

2. **Hidden Layer 2 Weights** (64 × 128)
   - Higher-level feature combinations
   - More abstract digit representations

3. **Training Loss** - Real-time cross-entropy loss
4. **Training Accuracy** - Batch-level accuracy tracking
5. **Sample Predictions** - Visual feedback on model performance

## 🏗️ Project Structure

```
.
├── mnist_classifier.py    # Neural network implementation
├── data_loader.py         # MNIST data handling
├── weight_visualizer.py   # Real-time visualization
├── train.py              # Training orchestration
├── best_model.pkl        # Saved trained model
└── mnist_data/           # Downloaded dataset
    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz
```

## 🎯 Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.61% |
| Validation Accuracy | 94.17% |
| Training Time | ~2.5 min (10 epochs) |
| Final Loss | 0.2447 |

## 🔧 Customization

Edit hyperparameters in `train.py`:

```python
EPOCHS = 10                           # Number of training epochs
BATCH_SIZE = 128                      # Mini-batch size
LEARNING_RATE = 0.01                  # Learning rate
VISUALIZATION_UPDATE_INTERVAL = 5     # Update every N batches
```

Modify network architecture in `mnist_classifier.py`:

```python
network = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],  # Customize layer sizes
    learning_rate=0.01
)
```

## 🧪 Using the Trained Model

```python
from mnist_classifier import NeuralNetwork
from data_loader import MNISTLoader
import numpy as np

# Load the trained model
model = NeuralNetwork()
model.load('best_model.pkl')

# Load test data
loader = MNISTLoader()
X_train, y_train, X_test, y_test = loader.load_data()

# Make predictions
predictions = model.predict(X_test)
accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

## 🎓 What Makes This Special

### Built from Scratch
- **No TensorFlow, PyTorch, or Keras** - Pure NumPy implementation
- **Manual backpropagation** - Computed gradients using chain rule
- **Educational** - Every line of code is understandable

### Live Visualization
- **See learning happen** - Watch weights evolve in real-time
- **Intuitive** - Visual feedback on what the network is learning
- **Performance-aware** - Updates efficiently without slowing training

### Production-Ready
- **Modular design** - Clean separation of concerns
- **Robust data loading** - Handles MNIST IDX format natively
- **Model persistence** - Save/load functionality built-in

## 📚 Technical Details

### Activation Functions
- **Hidden layers**: ReLU (`max(0, x)`)
- **Output layer**: Softmax (normalized probabilities)

### Training Algorithm
- **Loss**: Cross-entropy
- **Optimizer**: Mini-batch gradient descent
- **Initialization**: He initialization for ReLU networks

### Architecture Choices
- **784 input neurons**: 28×28 pixel images (flattened)
- **128 hidden neurons**: First layer feature detection
- **64 hidden neurons**: Second layer feature combination
- **10 output neurons**: Digit classes (0-9)

## 🐛 Troubleshooting

**No visualization appearing?**
- Make sure you're not running in a headless environment
- Check that matplotlib is installed correctly
- Try `plt.switch_backend('TkAgg')` before importing visualizer

**Training too slow?**
- Reduce `VISUALIZATION_UPDATE_INTERVAL` (updates less frequently)
- Use GPU-accelerated NumPy (cupy) if available
- Decrease batch size or number of epochs

**Low accuracy?**
- Increase learning rate (try 0.1)
- Train for more epochs
- Experiment with network architecture
- Check data normalization

## 📝 License

This is an educational project. Feel free to use, modify, and learn from it!

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun and collaborators
- Built as a learning exercise in neural network fundamentals

---

**Enjoy watching your neural network learn! 🎉**
