# 🧠 Deep Learning (LSTM Autoencoder) for Anomaly Detection using Time-Series Data  

### 🚀 NASA Bearing Dataset | Predictive Maintenance Project  

---

## 📘 Overview
This project focuses on detecting anomalies in **industrial bearing systems** using **Deep Learning**.A **Long Short-Term Memory (LSTM) Autoencoder** is trained on **NASA Bearing Prognostics Data** to learn normal time-series patterns and identify faults based on reconstruction errors.
The model is deployed as a **Flask REST API**, enabling **real-time anomaly detection** in sensor data streams.

---

## 🎯 Objectives
- Build an **unsupervised Deep Learning model** for anomaly detection  
- Use **LSTM Autoencoder** to reconstruct normal behavior  
- Detect anomalies based on **reconstruction error thresholding**  
- Deploy the model via a **Flask API** for real-time monitoring  
- Generate automated reports and performance visualizations  

---

## 🧩 Dataset Information
**Dataset:** NASA Bearing Prognostics Data Repository  
**Sensors:** 52 vibration and temperature sensors  
**Records:** Over 220,000 samples  
**Use Case:** Fault prediction in turbofan engines, turbines, and industrial machines  

---

## ⚙️ System Workflow

```
        NASA Dataset 
           ↓
    Data Preprocessing & Normalization
           ↓
    LSTM Autoencoder Training
           ↓
    Reconstruction Error Calculation 
           ↓
    Threshold Estimation 
           ↓
      Anomaly Detection
           ↓
    Flask REST API for Real-Time Predictions
```

---

## 🧱 Model Architecture

| Layer | Type | Description |
|--------|--------|-------------|
| 1 | Input | 100 timesteps × 51 features |
| 2 | LSTM(128) | Encoder layer |
| 3 | LSTM(64) | Dimensionality reduction |
| 4 | LSTM(32) | Bottleneck |
| 5 | LSTM(64) | Decoder layer |
| 6 | LSTM(128) | Sequence reconstruction |
| 7 | TimeDistributed(Dense(51)) | Output reconstruction |

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam (learning rate = 0.001)  
**Training Strategy:** Unsupervised (normal-only data)  

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| Accuracy | 92.10% |
| Precision | 99.09% |
| Recall | 69.28% |
| F1-Score | 81.54% |
| Threshold | 7.0869 |

✅ **Target achieved:** > 92% accuracy on anomaly detection tasks  

---

## 🧮 Evaluation Process
1. **Training:** Model learns to reconstruct normal sequences  
2. **Threshold:** Computed as 95th percentile of reconstruction errors  
3. **Prediction:** Sequences with errors > threshold → **Anomalies**  
4. **Validation:** Evaluated using confusion matrix and F1-score  

---

## 💻 Flask API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/ping` | GET | Quick server status check |
| `/health` | GET | Returns model & API health info |
| `/model_info` | GET | Displays model configuration details |
| `/predict` | POST | Predicts anomalies for new sensor data |

---

Example Request:
```json
{
  "sensor_data": [[0.23, 0.42, 0.51, ...], [0.26, 0.41, 0.55, ...]]
}

{
  "success": true,
  "statistics": {
    "mean_error": 0.245,
    "threshold": 7.0869
  },
  "predictions": {
    "anomalies_detected": 3,
    "anomaly_positions": [0, 5, 9],
    "anomaly_rate": 0.1
  }
}
```

---

```
📂 Deep Learning (LSTM Autoencoder) For Anomaly Detection Using Time-Series Data/
│
├── app.py                        # Flask API for model deployment
├── project_notebook.ipynb        # Full Jupyter notebook
├── models1/
│   └── nasa_bearing_production_v2/
│       ├── model.h5              # Trained LSTM Autoencoder
│       ├── scaler.pkl            # Scaler used during training
│       ├── threshold.pkl         # Saved threshold value
│       ├── config.json           # Model configuration info
│       └── training_history.json # Training history log
│
├── report_materials/             # Generated reports and performance files
│   ├── performance_metrics.csv
│   ├── training_history.csv
│   ├── confusion_matrix_details.csv
│   ├── model_architecture.csv
│   └── project_report.txt
│
├── results_dashboard1.png
├── README.md                     # Project documentation (this file)
└── requirements.txt              # Python dependencies
```

---

##  Running Instructions
1. **Install dependencies:** pip install tensorflow flask flask-cors joblib numpy pandas matplotlib scikit-learn plotly  
2. **Run Jupyter Notebook:** Train model and generate results using: jupyter notebook project_notebook.ipyn
3. **Start Flask API:** python app.py
4. **Test API Endpoints Use Python or Postman:** import requests
   print(requests.get("http://127.0.0.1:5000/ping").json())

---

 ## Visualizations
 ```
  PCA-based anomaly clustering
  Confusion matrix visualization
  Time-series anomaly plots
  Interactive 3D feature representation
  Loss curve (training vs validation)
```
---

 ## Key Features

- Real-time anomaly detection from sensor data
- End-to-end workflow: preprocessing → model → API
- High accuracy and robust thresholding
- Scalable architecture for future datasets
- Ready for industrial deployment

---

 ## Report & Deliverables
 ```
All generated files are saved in report_materials/ including:
  Performance metrics
  Confusion matrix
  Model architecture summary
  Training logs
  Comprehensive project report
```
---

 ## Conclusion
 ```
This project demonstrates a complete pipeline for real-world anomaly detection using Deep Learning.
The system achieves 92%+ accuracy, is API-integrated, and supports real-time predictions for industrial IoT use cases such as predictive maintenance and fault diagnosis.
```
---

 ## Author's
 ``` 
 - P. Sai Raghuveer Reddy
 - Department of Artificial Intelligence & Machine Learning
 - RNS Institute of Technology, Bengaluru
 - Year: 2025
```
---

 ## Acknowledgements
 ```
 - Dataset: NASA Prognostics Data Repository
 - Tools: TensorFlow, Flask, Scikit-learn, NumPy, Matplotlib
 - Guidance: Dr. Ramesh Babu H S , Principal & Professor, Department of CSE/CSE (DS), RNSIT
```
---
 ## Keywords
 ```
LSTM Autoencoder 
Anomaly Detection 
Time-Series Data 
Predictive Maintenance 
Flask API 
Deep Learning
```
---
