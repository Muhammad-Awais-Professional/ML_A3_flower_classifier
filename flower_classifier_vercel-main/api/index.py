import os
import json
import requests
import numpy as np
from flask import Flask, request, render_template, jsonify


HF_TOKEN = "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"
MODEL_PARAMS_URL = (
    "https://huggingface.co/AwaisTheGenius/flower_classifier/resolve/main/model_params.json"
)


TEMPLATE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "templates")
)
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def load_params():
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.get(MODEL_PARAMS_URL, headers=headers, timeout=10)
    resp.raise_for_status()
    params = resp.json()
    return (
        np.array(params["W"]),      
        np.array(params["b"]),      
        np.array(params["mean"]),   
        np.array(params["scale"]),  
    )

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


try:
    W, b, mean, scale = load_params()
    app.logger.info("✅ Loaded model_params.json from HF")
except Exception as e:
    W = b = mean = scale = None
    app.logger.error("❌ Failed to load params: %s", e)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if W is None:
            result = "Error: model parameters not loaded."
        else:
            try:
                sample = np.array([
                    float(request.form["sepal_length"]),
                    float(request.form["sepal_width"]),
                    float(request.form["petal_length"]),
                    float(request.form["petal_width"])
                ])
                x = (sample - mean) / scale
                logits = W.dot(x) + b
                probs = softmax(logits)
                idx = int(np.argmax(probs))
                classes = ["setosa", "versicolor", "virginica"]
                result = f"Predicted: {classes[idx]}, Confidence: {probs[idx]*100:.2f}%"
            except Exception as e:
                app.logger.exception("Error during prediction")
                result = f"Error: {e}"
    return render_template("index.html", result=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if W is None:
        return jsonify(error="model parameters not loaded"), 500

    try:
        data = request.get_json(force=True)
        sample = np.array([
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ], dtype=float)
        x = (sample - mean) / scale
        logits = W.dot(x) + b
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        classes = ["setosa", "versicolor", "virginica"]
        return jsonify({
            "predicted_class": classes[idx],
            "confidence": f"{probs[idx]*100:.2f}%"
        })
    except Exception as e:
        app.logger.exception("API prediction error")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
