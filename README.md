# Flower Image Classification

## Project Overview
This project aims to classify **flower images** into different categories using **Convolutional Neural Networks (CNN)**. The model learns visual patterns from images and predicts the flower type based on those patterns.

The project demonstrates a complete deep learning pipeline including data preprocessing, model training, evaluation, and result interpretation.

---

## Dataset Description
The flower dataset contains images belonging to multiple flower categories, commonly including:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

The dataset is divided into:
- Training set
- Validation set
- Test set

Images vary in background, lighting, and orientation, making the task realistic.

---

## Methodology

### 1. Data Preprocessing
- Image resizing to a fixed input size
- Normalization of pixel values
- Train–validation–test split
- Data augmentation to improve model generalization

### 2. Model Architecture
- Convolutional Neural Network (CNN)
- Convolution and ReLU activation layers
- Max Pooling layers
- Fully connected layers
- Softmax output layer for multi-class classification

### 3. Model Training
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy
- Model performance monitored using validation accuracy and loss

### 4. Model Evaluation
The model is evaluated using:
- Classification accuracy
- Confusion matrix
- Training and validation performance curves

---

## Results
- The CNN model was able to correctly classify most flower categories
- Certain visually similar flowers showed minor misclassification
- Performance can be further improved using transfer learning

---

## Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV / PIL
- Jupyter Notebook

---

## Project Structure
flower-classification/
│
├── data/
│ ├── train/
│ ├── validation/
│ └── test/
├── notebooks/
│ └── Flower_Classification.ipynb
├── models/
│ └── cnn_model.h5
├── README.md
