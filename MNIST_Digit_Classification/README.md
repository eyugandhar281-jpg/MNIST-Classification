# MNIST Digit Classification

## Project Overview

This project trains a TensorFlow / Keras neural network to classify handwritten digits from the MNIST dataset. The model is built with a simple fully connected network and is evaluated with test accuracy, confusion matrix, and sample prediction visualizations.

## Dataset Description

The MNIST dataset contains 70,000 handwritten digit images:
- 60,000 training images
- 10,000 test images
- Image shape: 28x28 pixels
- Grayscale values: 0-255
- Classes: 10 digits (0-9)

## Installation Steps

1. Open the project folder in Visual Studio Code.
2. Create a virtual environment and activate it:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the main script from the terminal:

```bash
python mnist_classifier.py
```

## Expected Output

The script will:
- Load the MNIST dataset from TensorFlow
- Display dataset shapes and class count
- Save sample images to `images/sample_images.png`
- Train a neural network for 10 epochs
- Save accuracy and loss graphs to `plots/`
- Save a confusion matrix to `plots/confusion_matrix.png`
- Save sample predictions to `images/sample_predictions.png`
- Save the trained model as `saved_model/mnist_digit_classifier.h5`

## Concepts Used

- TensorFlow / Keras model building
- Data normalization and reshaping
- Dense neural network layers
- Model compilation with Adam optimizer
- Sparse categorical crossentropy loss
- Training / validation split
- Model evaluation and prediction
- Visualization of training curves, confusion matrix, and sample predictions
