import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #  Suppress TensorFlow logs

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json
from datetime import datetime
import time

print("\n INITIALIZING NASA BEARING ANOMALY DETECTION API")
print("=" * 80)

# ------------------------------------------------------------------------------
# Flask App Initialization
# ------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  #  Allow cross-origin requests (for web or local testing

# Global Variables
api_model = None
api_scaler = None
api_threshold = None
api_config = None


# ------------------------------------------------------------------------------
# Load Production Model
# ------------------------------------------------------------------------------
def load_production_model(model_path: str):
    """Load trained LSTM Autoencoder and supporting files with detailed diagnostics"""
    global api_model, api_scaler, api_threshold, api_config

    try:
        print(f"\n Attempting to load model system from: {model_path}")

        # Check if directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" X Model directory not found: {model_path}")

        # 1️ Load model
        model_file = os.path.join(model_path, "model.h5")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f" X Missing model file: {model_file}")
        api_model = load_model(model_file, compile=False)
        print("\n Model system loaded successfully!")
        print(f" - Model Type: {api_config.get('model_architecture', 'Unknown')}")
        print(f" - Sequence Length: {api_config.get('sequence_length', '?')}")
        print(f" - Features: {api_config.get('n_features', '?')}")
        print(f" - Threshold: {api_config.get('threshold', '?')}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f" X Error loading model: {str(e)}")
        print(" ! Tip: Verify all files (model.h5, scaler.pkl, threshold.pkl, config.json) are present.")
        import traceback
        traceback.print_exc()
        return False



# ------------------------------------------------------------------------------
# ROUTE: Quick Ping
# ------------------------------------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    """Quick ping test"""
    return jsonify({"message": "API is alive!", "time": datetime.now().isoformat()})


# ------------------------------------------------------------------------------
# ROUTE: Health Check
# ------------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """API and Model Health"""
    return jsonify({
        "status": "healthy",
        "model_loaded": api_model is not None,
        "model_info": api_config if api_config else None,
        "timestamp": datetime.now().isoformat()
    })


# ------------------------------------------------------------------------------
# ROUTE: Model Info
# ------------------------------------------------------------------------------
@app.route("/model_info", methods=["GET"])
def get_model_info():
    """Model Configuration"""
    if api_config is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(api_config)


# ------------------------------------------------------------------------------
# ROUTE: Predict Anomalies
# ------------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_anomalies():
    """Predict anomalies in time-series sensor data"""

    if api_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.json
        if "sensor_data" not in data:
            return jsonify({
                "error": "Missing 'sensor_data' field",
                "expected_format": {"sensor_data": [[sensor1, sensor2, ..., sensor52]]}
            }), 400

        sensor_data = np.array(data["sensor_data"])
        n_features = api_config["n_features"]
        seq_len = api_config["sequence_length"]

        # Validate input dimensions
        if sensor_data.shape[1] != n_features:
            return jsonify({"error": f"Expected {n_features} sensors, got {sensor_data.shape[1]}"}), 400
        if len(sensor_data) < seq_len:
            return jsonify({"error": f"Need at least {seq_len} timesteps"}), 400

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Received /predict request with {len(sensor_data)} timesteps.")

        # Preprocess input
        sensor_data_scaled = api_scaler.transform(sensor_data)

        # Create overlapping sequences
        sequences = np.array([
            sensor_data_scaled[i:i + seq_len]
            for i in range(len(sensor_data_scaled) - seq_len + 1)
        ])

        # Reconstruct and compute errors
        reconstructed = api_model.predict(sequences, verbose=0)
        errors = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
        anomalies = (errors > api_threshold).astype(int)

        # Prepare response
        response = {
            "success": True,
            "statistics": {
                "mean_error": float(np.mean(errors)),
                "max_error": float(np.max(errors)),
                "min_error": float(np.min(errors)),
                "threshold": float(api_threshold)
            },
            "predictions": {
                "total_sequences": len(sequences),
                "anomalies_detected": int(np.sum(anomalies)),
                "anomaly_rate": float(np.mean(anomalies)),
                "anomaly_positions": np.where(anomalies == 1)[0].tolist()
            },
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


# ------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_impl_")

    print("\n Checking model path before loading...")
    if not os.path.exists(MODEL_PATH):
        print(f" ERROR: Model path '{MODEL_PATH}' not found!")
        print("Please check if you’re running app.py from the project root folder.")
    else:
        print(f" Found model path: {MODEL_PATH}")
        print("Files inside this folder:")
        print(os.listdir(MODEL_PATH))

        if load_production_model(MODEL_PATH):
            print("\n API ONLINE — ENDPOINTS READY:")
            print("    GET  /ping         - Quick test (instant response)")
            print("    GET  /health       - API health check")
            print("    GET  /model_info   - Model configuration")
            print("    POST /predict      - Predict anomalies")
            print("=" * 80)
        else:
            print("\n Model failed to load! Please check model path or files.")
            print("Hint: Ensure model.h5, scaler.pkl, threshold.pkl, and config.json exist.")

    # Start Flask server
    app.run(host="0.0.0.0", port=5000, debug=False)