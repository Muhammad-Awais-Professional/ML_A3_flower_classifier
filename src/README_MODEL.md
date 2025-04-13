# Iris Flower Classification Model

## Overview
This repository contains a trained Logistic Regression model for classifying iris flowers into three species: _setosa_, _versicolor_, and _virginica_. The model has been trained using the classic Iris dataset (numerical measurements) from scikit-learn and employs important machine learning techniques such as gradient descent optimization, L2 regularization, and early stopping.

The trained model and its associated scaler (used for feature preprocessing) are saved in the following files:
- **`iris_logistic_model.h5`** – The trained Keras model.
- **`iris_scaler.pkl`** – The StandardScaler instance used to preprocess the input features.

These files have been uploaded to the Hugging Face Model Hub and are available at:  
[https://huggingface.co/AwaisTheGenius/flower_classifier](https://huggingface.co/AwaisTheGenius/flower_classifier)

## Model Training
- **Dataset:** The Iris dataset (150 samples, 4 features per sample).
- **Model Architecture:** A simple neural network with a single dense layer using softmax activation (multi-class logistic regression).
- **Training Setup:**
  - **Optimizer:** SGD (Stochastic Gradient Descent) with a learning rate of 0.01.
  - **Loss Function:** Categorical Cross-Entropy.
  - **Regularization:** L2 regularization is applied in the Dense layer.
  - **Early Stopping:** Training is halted if the validation loss does not improve over a set number of epochs.

## Inference
An inference script (`inference.py`) is provided to load the model and scaler, process new input values, and perform predictions. The script accepts four numerical inputs representing sepal length, sepal width, petal length, and petal width.

### Usage:
```bash
python inference.py <sepal_length> <sepal_width> <petal_length> <petal_width>
