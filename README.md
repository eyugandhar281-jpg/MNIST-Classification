# MNIST Digit Classification

![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> A production-ready deep learning project for handwritten digit classification using TensorFlow/Keras. Achieves **97-98% accuracy** on the MNIST dataset with comprehensive visualizations and model analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🎯 Overview

This project implements a fully-connected neural network to classify handwritten digits (0-9) from the MNIST dataset. The model demonstrates fundamental deep learning concepts including data preprocessing, neural network design, training optimization, and performance evaluation.

**Key Achievements:**
- ✅ 97-98% test accuracy
- ✅ Comprehensive model evaluation with confusion matrix
- ✅ Professional visualizations of training metrics
- ✅ Full production-ready code with documentation
- ✅ Clean Git history suitable for portfolio

---

## ✨ Features

- **Complete ML Pipeline**: Data loading → Preprocessing → Training → Evaluation → Visualization
- **Professional Code**: Well-structured, commented code following Python best practices
- **Comprehensive Analysis**: Includes accuracy curves, loss curves, and confusion matrices
- **Model Persistence**: Saves trained model for inference and deployment
- **Reproducible Results**: Fixed random seeds for consistent outcomes
- **Educational Value**: Detailed comments explaining each step for beginners
- **Visualization Suite**:
  - Training and validation accuracy curves
  - Training and validation loss curves
  - Confusion matrix heatmap
  - Sample correct and incorrect predictions

---

## 📊 Dataset

**MNIST (Modified National Institute of Standards and Technology)**

| Aspect | Details |
|--------|---------|
| **Total Images** | 70,000 |
| **Training Set** | 60,000 images |
| **Test Set** | 10,000 images |
| **Image Size** | 28 × 28 pixels |
| **Classes** | 10 (digits 0-9) |
| **Color** | Grayscale |
| **Format** | NumPy arrays |

The MNIST dataset is a classic benchmark in machine learning, containing real-world handwritten digit samples with significant variation in writing styles, scales, and positions.

---

## 🧠 Model Architecture

```
Input: 28×28 grayscale images (784 features)
  ↓
Flatten Layer (784 neurons)
  ↓
Dense Layer (128 neurons, ReLU activation)
  ↓
Dense Layer (64 neurons, ReLU activation)
  ↓
Output Layer (10 neurons, Softmax activation)
  ↓
Prediction (digit 0-9)
```

**Training Configuration:**
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 128
- **Validation Split**: 10%
- **Total Parameters**: 109,386 (427.29 KB)

---

## 📦 Installation

### Prerequisites

- Python 3.13 or higher
- pip (Python package manager)
- 2GB free disk space
- ~5 minutes for initial setup

### Step-by-Step Setup

**1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/MNIST-Digit-Classification.git
cd MNIST
```

**2. Create Virtual Environment**

*Windows (PowerShell):*
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

*macOS/Linux:*
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r MNIST_Digit_Classification/requirements.txt
```

**Verify Installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
```

---

## 🚀 Usage

### Run Training and Evaluation

```bash
python MNIST_Digit_Classification/mnist_classifier.py
```

**Script Execution:**
The script will:
- ✓ Load the MNIST dataset automatically
- ✓ Preprocess and normalize the data
- ✓ Display model architecture
- ✓ Train the model for 10 epochs
- ✓ Evaluate on test set
- ✓ Generate comprehensive visualizations
- ✓ Save the trained model and results

**Expected Runtime:**
- CPU: 2-3 minutes
- GPU: 30-60 seconds

### Output Files Generated

After execution, check these directories for results:

```
MNIST_Digit_Classification/
├── saved_model/mnist_digit_classifier.h5    # Trained model
├── plots/
│   ├── accuracy.png                        # Accuracy curves
│   ├── loss.png                           # Loss curves
│   └── confusion_matrix.png               # Confusion matrix
└── images/
    ├── sample_images.png                  # MNIST samples
    └── sample_predictions.png             # Model predictions
```

---

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.84% |
| **Test Loss** | 0.0748 |
| **Training Time** | ~2-3 min (CPU) |
| **Model Size** | 427.29 KB |

### Key Statistics

- **Correct Predictions**: 9,784 / 10,000 (97.84%)
- **Incorrect Predictions**: 216 / 10,000 (2.16%)

### Per-Digit Accuracy

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.99 | 0.98 | 0.99 | 980 |
| 1 | 0.99 | 0.99 | 0.99 | 1,135 |
| 2 | 0.98 | 0.98 | 0.98 | 1,032 |
| 3 | 0.97 | 0.98 | 0.98 | 1,010 |
| 4 | 0.98 | 0.97 | 0.98 | 982 |
| 5 | 0.95 | 0.99 | 0.97 | 892 |
| 6 | 0.99 | 0.97 | 0.98 | 958 |
| 7 | 0.98 | 0.98 | 0.98 | 1,028 |
| 8 | 0.99 | 0.96 | 0.97 | 974 |
| 9 | 0.97 | 0.98 | 0.97 | 1,009 |

---

## 📁 Project Structure

```
MNIST/
├── MNIST_Digit_Classification/
│   ├── mnist_classifier.py              # Main training script (12 steps)
│   ├── requirements.txt                 # Python dependencies
│   │
│   ├── saved_model/
│   │   └── mnist_digit_classifier.h5   # Trained model (427.29 KB)
│   │
│   ├── plots/                           # Generated evaluation plots
│   │   ├── accuracy.png                # Training/Validation accuracy
│   │   ├── loss.png                    # Training/Validation loss
│   │   └── confusion_matrix.png        # Confusion matrix heatmap
│   │
│   └── images/                          # Sample visualizations
│       ├── sample_images.png           # MNIST samples
│       └── sample_predictions.png      # Model predictions
│
├── .git/                                # Git version control
├── .gitignore                           # Git ignore file
├── README.md                            # This documentation
└── [.venv/]                             # Virtual environment (not tracked)
```

---

## 🛠️ Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.13+ | Programming Language |
| **TensorFlow** | 2.21.0 | Deep Learning Framework |
| **Keras** | 3.14.1 | Neural Network API |
| **NumPy** | 2.4.6 | Numerical Computing |
| **Matplotlib** | 3.10.9 | Data Visualization |
| **Scikit-learn** | 1.9.0 | ML Metrics & Analysis |

---

## 🚀 Future Improvements

- [ ] **Convolutional Neural Network (CNN)** - More advanced architecture
- [ ] **Data Augmentation** - Rotation, scaling, and noise transformations
- [ ] **Regularization** - Dropout and batch normalization layers
- [ ] **Hyperparameter Tuning** - GridSearchCV for optimal parameters
- [ ] **Model Ensemble** - Combine multiple models for better accuracy
- [ ] **Web Interface** - Flask/FastAPI REST API for predictions
- [ ] **Mobile Deployment** - TensorFlow Lite for mobile devices
- [ ] **Real-Time Recognition** - Webcam digit recognition system
- [ ] **Model Optimization** - Quantization and pruning for deployment
- [ ] **Extended Datasets** - Test on Fashion-MNIST, EMNIST

---

## 💡 Learning Outcomes

**Core Deep Learning Skills:**
- Neural network architecture design and implementation
- Understanding forward pass and backpropagation
- Loss functions and optimization algorithms

**Data Science & ML:**
- Data preprocessing, normalization, and augmentation
- Train/validation/test data splitting
- Performance evaluation and metrics

**Software Engineering:**
- Clean, well-documented Python code
- Project structure and organization best practices
- Version control with Git and GitHub

**Professional Development:**
- Model persistence and serialization
- Comprehensive visualization and analysis
- Production-ready ML pipeline design

---

## 📚 Learning Resources

- **[TensorFlow Official Docs](https://www.tensorflow.org/documentation)** - Complete TensorFlow guide
- **[Keras API Reference](https://keras.io/api/)** - Layer and model documentation
- **[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)** - Original dataset information
- **[Deep Learning Book](https://www.deeplearningbook.org/)** - Comprehensive theory
- **[Scikit-learn Docs](https://scikit-learn.org/)** - ML metrics and utilities

---

## 📄 License

This project is licensed under the **MIT License** - free for personal and commercial use.

You are free to:
- ✓ Use this code in your projects
- ✓ Modify and distribute
- ✓ Include in your portfolio

---

## 🤝 Support

If you find this project helpful:
- ⭐ Star the repository
- 🐛 Report issues or bugs
- 💡 Suggest improvements
- 📢 Share with others

---

<div align="center">

**Made with ❤️ for learning and professional growth**

[⬆ Back to Top](#mnist-digit-classification)

</div>
