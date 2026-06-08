# MNIST Digit Classification

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end deep learning project that classifies handwritten digits (0–9) using TensorFlow and Keras. The project covers data preprocessing, neural network training, evaluation, visualization, and model persistence.

## Features

* MNIST handwritten digit classification
* Neural network built with TensorFlow/Keras
* Training and validation monitoring
* Confusion matrix generation
* Sample prediction visualization
* Saved model for future inference

## Dataset

The project uses the MNIST dataset:

* 70,000 grayscale images
* 28 × 28 pixel resolution
* 10 classes (digits 0–9)
* 60,000 training samples
* 10,000 testing samples

## Model Architecture

Input (28×28)

↓

Flatten

↓

Dense (128, ReLU)

↓

Dense (64, ReLU)

↓

Dense (10, Softmax)

## Installation

```bash
git clone https://github.com/<your-username>/MNIST-Digit-Classification.git
cd MNIST-Digit-Classification

python -m venv .venv
.venv\Scripts\activate

pip install -r MNIST_Digit_Classification/requirements.txt
```

## Usage

```bash
python MNIST_Digit_Classification/mnist_classifier.py
```

## Results

Expected performance:

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 97–98% |
| Loss     | ~0.07  |

### Accuracy Curve

(Add screenshot here)

### Loss Curve

(Add screenshot here)

### Confusion Matrix

(Add screenshot here)

### Sample Predictions

(Add screenshot here)

## Project Structure

```text
MNIST-Digit-Classification/
│
├── README.md
├── .gitignore
│
└── MNIST_Digit_Classification/
    ├── mnist_classifier.py
    ├── requirements.txt
    ├── saved_model/
    ├── plots/
    └── images/
```

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn

## Future Improvements

* Convolutional Neural Networks (CNNs)
* Data Augmentation
* Hyperparameter Tuning
* Web Deployment
* TensorFlow Lite Support

## License

MIT License
