from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import joblib
from huggingface_hub import hf_hub_download


REPO_ID = "AwaisTheGenius/flower_classifier"
TOKEN = "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"

def load_model_and_scaler():
    """
    Downloads the model and scaler files from the Hugging Face repository
    and loads them using the provided token.
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
        raise

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print("Error loading model:", e)
        raise

    try:
        scaler_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename="iris_scaler.pkl",
            revision="main",
            token=TOKEN
        )
    except Exception as e:
        print("Error downloading scaler file:", e)
        raise

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print("Error loading scaler:", e)
        raise

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


app = Flask(__name__)



try:
    model, scaler = load_model_and_scaler()
    print("Model and scaler loaded successfully.")
except Exception as e:
    print("Failed to load model and scaler:", e)
    

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            
            sample = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            
            pred_class, confidence = predict(model, scaler, sample)
            classes = ['setosa', 'versicolor', 'virginica']
            result = f"Predicted Class: {classes[pred_class]}, Confidence: {confidence*100:.2f}%"
        except Exception as e:
            result = f"Error during prediction: {e}"
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    An API endpoint that accepts JSON input with the four features
    and returns a JSON with the prediction result.
    """
    try:
        data = request.get_json()
        sample = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]
        pred_class, confidence = predict(model, scaler, sample)
        classes = ['setosa', 'versicolor', 'virginica']
        response = {
            "predicted_class": classes[pred_class],
            "confidence": f"{confidence*100:.2f}%"
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
