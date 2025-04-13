from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import joblib
from huggingface_hub import hf_hub_download


REPO_ID = "AwaisTheGenius/flower_classifier"
TOKEN = "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"

def load_model_and_scaler():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="iris_logistic_model.h5", revision="main", token=TOKEN)
    model = tf.keras.models.load_model(model_path)
    scaler_path = hf_hub_download(repo_id=REPO_ID, filename="iris_scaler.pkl", revision="main", token=TOKEN)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(model, scaler, sample):
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
    print("Model loading failed:", e)

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
            result = f"Error: {e}"
    return render_template("index.html", result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    sample = [
        data.get("sepal_length"),
        data.get("sepal_width"),
        data.get("petal_length"),
        data.get("petal_width")
    ]
    pred_class, confidence = predict(model, scaler, sample)
    classes = ['setosa', 'versicolor', 'virginica']
    return jsonify({
        "predicted_class": classes[pred_class],
        "confidence": f"{confidence*100:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
