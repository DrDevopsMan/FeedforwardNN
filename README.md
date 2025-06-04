# MNIST Digit Classifier using Multi-Layer Perceptron (MLP) and data augmentation.

This project implements a high-performance digit classifier trained on the MNIST dataset using a deep fully connected neural network (MLP). It leverages both original and augmented data to achieve robust performance, reaching over **99% test accuracy** on unseen digits.
I wanted to see how much extra accuracy I can obtain by augmenting the data and getting double the original data size.
---

## 🧠 Model Overview

- Input: Flattened 28×28 grayscale MNIST images (784 features)
- Architecture:
  - 7 fully connected layers with ReLU activation
  - Layer sizes: 784 → 700 → 650 → 650 → 500 → 450 → 400 → 10
- Output: Raw class scores for digits 0–9 (no softmax; handled by `CrossEntropyLoss`)
- Optimizer: Adam (`lr=0.0001`)
- Loss: CrossEntropyLoss
- Training Epochs: 14

---

## 📈 Data Augmentation Strategy

To boost generalization and robustness, the training dataset is **doubled** by combining:

1. Original MNIST training data
2. Augmented MNIST training data with:
   - Random rotation (±15°)
   - Random translation (±10%)
   - Random scaling (90%–110%)
   - Random shear (±10°)

The test set is left untouched and clean for reliable evaluation.

---

## 🗂️ Directory Structure
project/
├── MLP.py # Main training and evaluation script
├── data/ # MNIST dataset will download here
└── README.md # This file
---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install torch torchvision
```

### 2. Run the Training Script
bash
Copy
Edit
python MLP.py
The script will:

Automatically download MNIST

Train on 120,000 training samples (60k original + 60k augmented)

Evaluate on 10,000 clean test samples

Print accuracy at the end of training

## 📊 Sample Output
yaml
Copy
Edit
Using device: cuda
Epoch [14/14], Loss: 0.0017
Test Accuracy: 99.13%
### 💡 Notes
This is a pure MLP model — no convolutional layers are used.

Even without CNNs, the combination of deep layers and data augmentation yields excellent performance.

GPU acceleration is supported via PyTorch (torch.device("cuda" if available)).

🛠️ Future Work Ideas
Add a CNN-based architecture for comparison

Visualize misclassified digits

Export to ONNX or TorchScript

Extend to fashion-MNIST or EMNIST

📜 License
MIT License

python
Copy
Edit

---

Let me know if you'd like a version that includes images or model diagrams, or if you're planning to turn this into a notebook or blog post — I can format it accordingly!
