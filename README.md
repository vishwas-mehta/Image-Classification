# 🖼️ Image Classification with CNN on CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-orange.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> **Last Updated:** December 2024

A comprehensive deep learning project implementing **Convolutional Neural Networks (CNNs)** for image classification on the **CIFAR-10** dataset. This project demonstrates CNN fundamentals, data augmentation, regularization techniques, and model comparison.

## 🎯 Project Highlights

- ✅ **Custom CNN Architecture** - Built from scratch with BatchNorm & Dropout
- ✅ **ResNet18 Implementation** - Adapted for CIFAR-10 comparison
- ✅ **Data Augmentation** - Multiple augmentation strategies
- ✅ **Comprehensive Evaluation** - Accuracy, confusion matrix, per-class metrics
- ✅ **Overfitting Analysis** - Regularization techniques demonstration
- ✅ **Professional Code Structure** - Modular, well-documented codebase

## 📊 Results

| Model | Parameters | Test Accuracy |
|-------|------------|---------------|
| Custom CNN | ~1.0M | ~78% |
| ResNet18 | ~11.2M | ~85% |

> *Note: Results may vary based on training duration and hardware.*

## 📁 Project Structure

```
Image-Classification/
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt          # Dependencies
├── 📄 config.py                 # Hyperparameters & settings
├── 📂 src/
│   ├── 📄 __init__.py
│   ├── 📄 data_loader.py        # CIFAR-10 data loading
│   ├── 📄 augmentation.py       # Data augmentation utilities
│   ├── 📄 train.py              # Training loop & early stopping
│   ├── 📄 evaluate.py           # Evaluation metrics
│   ├── 📄 utils.py              # Visualization utilities
│   ├── 📄 overfitting_analysis.py
│   └── 📂 models/
│       ├── 📄 __init__.py
│       ├── 📄 custom_cnn.py     # Custom CNN architecture
│       └── 📄 resnet.py         # ResNet18 implementation
├── 📂 notebooks/
│   └── 📓 Image_Classification_CNN.ipynb
├── 📂 models/                   # Saved model checkpoints
└── 📂 results/                  # Training plots & visualizations
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision
- **Data Processing**: NumPy, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: scikit-learn
- **Progress Bar**: tqdm

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishwas-mehta/Image-Classification.git
   cd Image-Classification
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Quick Start with Jupyter Notebook
```bash
cd notebooks
jupyter notebook Image_Classification_CNN.ipynb
```

#### Train Custom CNN
```python
from src.data_loader import get_data_loaders
from src.models.custom_cnn import CustomCNN
from src.train import train_model

# Load data
train_loader, val_loader, test_loader = get_data_loaders()

# Create and train model
model = CustomCNN(num_classes=10, dropout_rate=0.5)
history = train_model(model, train_loader, val_loader, epochs=50)
```

#### Train ResNet18
```python
from src.models.resnet import ResNet18

model = ResNet18(num_classes=10)
history = train_model(model, train_loader, val_loader, epochs=50)
```

#### Evaluate Model
```python
from src.evaluate import evaluate_and_report

results = evaluate_and_report(model, test_loader)
```

## 📈 Training Details

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Epochs | 50 (with early stopping) |

### Data Augmentation
- Random Crop (32x32, padding=4)
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter
- Normalization

### Regularization Techniques
- **Dropout** (rate=0.5)
- **Batch Normalization**
- **Early Stopping** (patience=10)
- **Learning Rate Scheduling** (ReduceLROnPlateau)
- **Weight Decay** (L2 regularization)

## 🏗️ Model Architectures

### Custom CNN
```
Input: 3x32x32
├── ConvBlock1: Conv2D(3→32) + BN + ReLU + MaxPool → 32x16x16
├── ConvBlock2: Conv2D(32→64) + BN + ReLU + MaxPool → 64x8x8
├── ConvBlock3: Conv2D(64→128) + BN + ReLU + MaxPool → 128x4x4
├── ConvBlock4: Conv2D(128→256) + BN + ReLU → 256x4x4
├── Flatten → 4096
├── FC1: 4096→512 + ReLU + Dropout
├── FC2: 512→256 + ReLU + Dropout
└── FC3: 256→10 (Output)
```

### ResNet18 (Modified for CIFAR-10)
- Modified first conv layer (3x3 kernel, stride=1)
- Removed initial max pooling
- Output layer: 512→10

## 📊 Dataset

**CIFAR-10** consists of 60,000 32x32 color images in 10 classes:

| Class | Description |
|-------|-------------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

- **Training**: 45,000 images
- **Validation**: 5,000 images
- **Test**: 10,000 images

## 🔬 Overfitting Analysis

This project includes a detailed analysis of overfitting and the effect of regularization:

1. **Without Regularization**: Model overfits with large train-val gap
2. **With Regularization**: Reduced overfitting, better generalization

Key findings:
- Dropout + BatchNorm significantly reduce overfitting
- Data augmentation improves generalization
- Early stopping prevents training too long

## 📝 Future Improvements

- [ ] Implement VGG and DenseNet architectures
- [ ] Add transfer learning with ImageNet pretrained weights
- [ ] Experiment with CutOut and MixUp augmentation
- [ ] Hyperparameter tuning with Optuna
- [ ] Add model interpretability (Grad-CAM)
- [ ] Deploy model with FastAPI/Streamlit

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- [PyTorch](https://pytorch.org/) team for the excellent framework
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) paper by He et al.

---

<p align="center">
  Made with ❤️ for Deep Learning
</p>
