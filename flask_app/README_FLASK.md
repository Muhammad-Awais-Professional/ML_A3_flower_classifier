# Iris Flower Classification Web UI

## Overview
This repository contains a Flask web application that provides a user-friendly interface for the Iris Flower Classification model. The web app allows users to input iris flower measurements through a simple web form and returns the predicted species along with its confidence percentage.

The application loads the trained model and scaler from the Hugging Face Model Hub (using the provided token) during startup, ensuring that predictions are based on the latest model files.

## Files Structure
- **`app.py`**: Main Flask application file.
- **`templates/index.html`**: HTML template that renders the input form and displays the prediction result.
- **`requirements.txt`**: List of project dependencies.

## How It Works
- **Model Loading:**  
  Upon startup, the Flask app downloads `iris_logistic_model.h5` and `iris_scaler.pkl` from the Hugging Face repository and loads them into memory.
  
- **Web Interface:**  
  Navigate to the homepage (`/`) to see an input form. Enter the four iris measurements and submit the form to receive a prediction on the same page.
  
- **API Endpoint:**  
  An additional API endpoint is available at `/api/predict` that accepts JSON POST requests and returns the prediction in JSON format.

## Installation & Running the Web App

1. **Install Dependencies:**
   ```bash
   pip install flask tensorflow scikit-learn joblib huggingface_hub