import sys
import numpy as np
import tensorflow as tf
import joblib
from huggingface_hub import hf_hub_download


REPO_ID = "AwaisTheGenius/flower_classifier"


TOKEN = "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"

def load_model_and_scaler():
    """
    Downloads the model and scaler files from the Hugging Face repository
    and loads them using the provided authentication token.
    """
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename="iris_logistic_model.h5",
            revision="main",
            token=TOKEN
        )
    except Exception as e:
        print("Error downloading model file:", e)
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print("Error loading model:", e)
        sys.exit(1)

    try:
        scaler_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename="iris_scaler.pkl",
            revision="main",
            token=TOKEN
        )
    except Exception as e:
        print("Error downloading scaler file:", e)
        sys.exit(1)

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print("Error loading scaler:", e)
        sys.exit(1)

    return model, scaler

def predict(model, scaler, sample):
    """
    Preprocesses the sample, performs prediction, and returns the predicted
    class index and the confidence score.
    """
    
    sample = np.array(sample).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    probabilities = model.predict(sample_scaled)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return predicted_class, confidence

def main():
    
    if len(sys.argv) != 5:
        print("Usage: python inference.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        sys.exit(1)

    try:
        sample = [float(arg) for arg in sys.argv[1:5]]
    except ValueError:
        print("Please ensure all inputs are numeric values.")
        sys.exit(1)

    print("Downloading and loading model and scaler from Hugging Face...")
    model, scaler = load_model_and_scaler()

    predicted_class, confidence = predict(model, scaler, sample)

    classes = ['setosa', 'versicolor', 'virginica']
    print(f"Predicted Class: {classes[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")

if __name__ == '__main__':
    main()
