"""
MNIST Digit Classification - TensorFlow / Keras Version

This script trains a neural network on the MNIST dataset, evaluates it,
creates plots, saves predictions images, and saves the trained model.

Run with a Python interpreter that has TensorFlow installed:
python mnist_classifier.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# Create project directories relative to this script location
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / 'images'
PLOTS_DIR = BASE_DIR / 'plots'
MODEL_DIR = BASE_DIR / 'saved_model'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 80)
print('MNIST Digit Classification - TensorFlow / Keras')
print('=' * 80)

print('Step 1: Import required libraries')
print('TensorFlow version:', tf.__version__)
print('NumPy version:', np.__version__)
print('Matplotlib backend:', matplotlib.get_backend())
print('Scikit-learn version:', __import__('sklearn').__version__)
print()

print('Step 2: Load MNIST dataset from TensorFlow')
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
print(f'  Train images shape: {train_images.shape}')
print(f'  Train labels shape: {train_labels.shape}')
print(f'  Test images shape: {test_images.shape}')
print(f'  Test labels shape: {test_labels.shape}')
print('✓ Dataset loaded successfully\n')

print('Step 3: Display dataset shape and number of classes')
num_classes = len(np.unique(train_labels))
print(f'  Number of classes: {num_classes}')
print(f'  Input image size: {train_images.shape[1]} x {train_images.shape[2]}')

# Save a sample image grid for review
sample_fig, sample_axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(sample_axes.flat):
    ax.imshow(train_images[i], cmap='gray')
    ax.set_title(f'Label: {train_labels[i]}')
    ax.axis('off')
plt.tight_layout()
sample_path = IMAGES_DIR / 'sample_images.png'
sample_fig.savefig(sample_path, dpi=100, bbox_inches='tight')
plt.close(sample_fig)
print(f'  Sample images saved to {sample_path}\n')

print('Step 4: Preprocess data')
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape images for the neural network input
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
print(f'  Train images shape after reshape: {train_images.shape}')
print(f'  Test images shape after reshape: {test_images.shape}')
print('✓ Data normalized and reshaped\n')

print('Step 5: Build the neural network')
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1), name='flatten'),
    keras.layers.Dense(128, activation='relu', name='dense_128'),
    keras.layers.Dense(64, activation='relu', name='dense_64'),
    keras.layers.Dense(num_classes, activation='softmax', name='output')
], name='mnist_dense_model')

model.summary()
print('✓ Model architecture created\n')

print('Step 6: Compile model')
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print('✓ Model compiled\n')

print('Step 7: Train model')
history = model.fit(
    train_images,
    train_labels,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    verbose=2
)
print('✓ Training complete\n')

print('Step 8: Evaluate on test data')
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f'  Test loss: {test_loss:.4f}')
print(f'  Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n')

print('Step 9: Predict test images')
predictions = model.predict(test_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
print(f'  Predicted labels shape: {predicted_labels.shape}')
print('✓ Predictions completed\n')

print('Step 10: Generate accuracy and loss graphs')
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
accuracy_plot = PLOTS_DIR / 'accuracy.png'
plt.savefig(accuracy_plot, dpi=100, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
loss_plot = PLOTS_DIR / 'loss.png'
plt.savefig(loss_plot, dpi=100, bbox_inches='tight')
plt.close()
print(f'  Accuracy graph saved to {accuracy_plot}')
print(f'  Loss graph saved to {loss_plot}\n')

print('Step 10: Generate confusion matrix and classification report')
cm = confusion_matrix(test_labels, predicted_labels)
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar(cax)
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
confusion_plot = PLOTS_DIR / 'confusion_matrix.png'
plt.savefig(confusion_plot, dpi=100, bbox_inches='tight')
plt.close()
print(f'  Confusion matrix saved to {confusion_plot}')

report = classification_report(test_labels, predicted_labels, target_names=[str(i) for i in range(num_classes)])
print('\nClassification Report:')
print(report)

print('Step 11: Show correct and incorrect predictions')
correct_indices = np.where(predicted_labels == test_labels)[0]
incorrect_indices = np.where(predicted_labels != test_labels)[0]
print(f'  Correct predictions: {len(correct_indices)}')
print(f'  Incorrect predictions: {len(incorrect_indices)}')

sample_count = 10
sample_indices = np.concatenate([
    correct_indices[: min(5, len(correct_indices))],
    incorrect_indices[: min(5, len(incorrect_indices))]
])
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for ax, idx in zip(axes.flat, sample_indices):
    ax.imshow(test_images[idx].squeeze(), cmap='gray')
    ax.set_title(f'True: {test_labels[idx]}\nPred: {predicted_labels[idx]}')
    ax.axis('off')
plt.tight_layout()
prediction_path = IMAGES_DIR / 'sample_predictions.png'
plt.savefig(prediction_path, dpi=100, bbox_inches='tight')
plt.close()
print(f'  Sample predictions saved to {prediction_path}\n')

print('Step 12: Save trained model')
model_path = MODEL_DIR / 'mnist_digit_classifier.h5'
model.save(model_path)
print(f'  Model saved to {model_path}\n')

print('=' * 80)
print('Project completed successfully!')
print('=' * 80)
print(f'  Saved model: {model_path}')
print(f'  Plots directory: {PLOTS_DIR}')
print(f'  Images directory: {IMAGES_DIR}')
print('=' * 80)
