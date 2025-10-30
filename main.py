
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import xgboost as xgb
import numpy as np

# Charger modèle et scaler
MODEL_PATH = "models/best_xgb_model.json"
SCALER_PATH = "models/scaler.joblib"

try:
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur chargement modèle: {e}")

app = FastAPI(title="KFS AI4Health Diabetes Prediction API")

# Schema des données pour la prédiction
class PatientData(BaseModel):
    gender: int
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Transformer en array
        input_data = np.array([[
            data.gender,
            data.age,
            data.hypertension,
            data.heart_disease,
            data.smoking_history,
            data.bmi,
            data.HbA1c_level,
            data.blood_glucose_level
        ]])

        # Normalisation
        input_scaled = scaler.transform(input_data)

        # DMatrix pour XGBoost
        dmatrix = xgb.DMatrix(input_scaled)
        pred = model.predict(dmatrix)
        pred_prob = float(pred[0])
        pred_class = int(pred_prob > 0.5)
        return {
            "prediction": pred_class,
            "probability": pred_prob,
            "message": "Diabetic" if pred_class else "Not Diabetic"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")
