
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "aqi_model.joblib"

FEATURES = [
    "max_wind_speed",
    "max_temperature",
    "min_temperature",
    "average_temperature",
    "average_wind_speed",
]


# --------------------------------------------------
# 2. INITIALIZE FLASK APPLICATION
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python train_model.py first."
        )

    loaded_model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

    return loaded_model


model = load_model()


# --------------------------------------------------
# 4. HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "home.html",
        prediction=None,
        error=None
    )


# --------------------------------------------------
# 5. INPUT VALIDATION AND PREDICTION
# --------------------------------------------------

def make_prediction(payload):

    if not hasattr(payload, "get"):
        raise ValueError("Input must be a valid dictionary or form data.")

    values = []

    for feature in FEATURES:

        value = payload.get(feature)

        if value is None or str(value).strip() == "":
            raise ValueError(
                f"Missing required field: {feature}"
            )

        try:
            value = float(value)

        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must contain a valid number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        values.append(value)

    # Convert input into a Pandas DataFrame.
    # Column names and their order must match model training.

    input_data = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    print("\nInput received:")
    print(input_data)

    # Generate prediction

    prediction = model.predict(input_data)

    predicted_aqi = float(prediction[0])

    return round(predicted_aqi, 2)


# --------------------------------------------------
# 6. HTML FORM PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_form():

    try:

        prediction = make_prediction(request.form)

        return render_template(
            "home.html",
            prediction=prediction,
            error=None
        )

    except ValueError as exc:

        return render_template(
            "home.html",
            prediction=None,
            error=str(exc)
        ), 400

    except Exception as exc:

        app.logger.exception("Prediction failed")

        return render_template(
            "home.html",
            prediction=None,
            error="Prediction failed. Check the Flask terminal."
        ), 500


# --------------------------------------------------
# 7. REST API PREDICTION
# --------------------------------------------------

@app.route("/predict_api", methods=["POST"])
def predict_api():

    if not request.is_json:

        return jsonify({
            "status": "error",
            "message": "Please send input in JSON format."
        }), 415

    try:

        payload = request.get_json(silent=True)

        if payload is None:
            return jsonify({
                "status": "error",
                "message": "Invalid or empty JSON request."
            }), 400

        prediction = make_prediction(payload)

        return jsonify({
            "status": "success",
            "predicted_aqi": prediction,
            "unit": "AQI"
        }), 200

    except ValueError as exc:

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 400

    except Exception:

        app.logger.exception("API prediction failed")

        return jsonify({
            "status": "error",
            "message": "Internal prediction error."
        }), 500


# --------------------------------------------------
# 8. HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


# --------------------------------------------------
# 9. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
