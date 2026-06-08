# MNIST Digit Classification - TensorFlow Implementation

This folder contains the main training script and project structure for the MNIST digit classification project.

## 📂 Files

- **mnist_classifier.py** - Main training and evaluation script
- **requirements.txt** - Python package dependencies
- **README.md** - Project documentation (this file)
- **saved_model/** - Directory containing the trained model
- **plots/** - Generated training curves and confusion matrix
- **images/** - Sample images and predictions visualization

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training

```bash
python mnist_classifier.py
```

The script will:
1. Load the MNIST dataset
2. Preprocess and normalize the data
3. Build and train a neural network
4. Evaluate on test data
5. Generate visualizations and save the model

## 📊 Neural Network Architecture

```
Input: 28x28 images
  ↓
Flatten: 784 neurons
  ↓
Dense: 128 neurons (ReLU)
  ↓
Dense: 64 neurons (ReLU)
  ↓
Output: 10 neurons (Softmax)
```

## 🎯 Model Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 128
- **Validation Split**: 10%

## 📈 Output Files

After running the script, you'll find:

- `saved_model/mnist_digit_classifier.h5` - Trained model
- `plots/accuracy.png` - Training/Validation accuracy curve
- `plots/loss.png` - Training/Validation loss curve
- `plots/confusion_matrix.png` - Confusion matrix heatmap
- `images/sample_images.png` - Random MNIST samples
- `images/sample_predictions.png` - Correct and incorrect predictions

## 📊 Expected Performance

- **Training Accuracy**: ~99%+
- **Test Accuracy**: ~97-98%
- **Test Loss**: ~0.07-0.08

## 🐍 Requirements

- Python 3.13+
- TensorFlow 2.21+
- NumPy 2.4+
- Matplotlib 3.10+
- Scikit-learn 1.9+

## 💡 Code Walkthrough

The script follows 12 main steps:

1. Import libraries
2. Load MNIST dataset
3. Display dataset info and sample images
4. Normalize and preprocess data
5. Build neural network architecture
6. Compile model
7. Train the model
8. Evaluate on test data
9. Make predictions
10. Generate accuracy and loss graphs
11. Show correct/incorrect predictions
12. Save the trained model

Each step includes detailed print statements for tracking progress.

## 🔧 Customization

You can modify these parameters in `mnist_classifier.py`:

- `epochs` - Number of training epochs (default: 10)
- `batch_size` - Mini-batch size (default: 128)
- `validation_split` - Validation data percentage (default: 0.1)
- Dense layer sizes - Change network architecture
- Learning rate - Adjust optimizer parameters

## ⚠️ Notes

- First run will download MNIST dataset (~11MB)
- Training time depends on your CPU/GPU
- The `.h5` model file can be loaded for inference later
- Plots are saved as PNG files for easy viewing

## 📚 Further Learning

- Experiment with different network architectures
- Try adding dropout layers for regularization
- Implement data augmentation techniques
- Explore convolutional neural networks (CNN)
- Compare with other optimizers (SGD, RMSprop)

---

**Created**: June 2026  
**Framework**: TensorFlow/Keras  
**Dataset**: MNIST (70,000 handwritten digits)
