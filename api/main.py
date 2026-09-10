from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import make_features
from src.maintenance import recommend_maintenance

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "predictive_maintenance_model.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"
model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())

app = FastAPI(title="Intelligent Predictive Maintenance API", version="1.0.0",
              description="Predict industrial machine failure risk and recommend maintenance actions.")

class SensorInput(BaseModel):
    machine_id: str = Field(default="M001", description="Equipment identifier")
    product_type: str = Field(default="M", pattern="^[LMH]$", description="Product quality variant")
    air_temperature: float = Field(..., ge=250, le=350, description="Air temperature in Kelvin")
    process_temperature: float = Field(..., ge=250, le=370, description="Process temperature in Kelvin")
    rotational_speed: float = Field(..., ge=500, le=4000, description="Rotational speed in rpm")
    torque: float = Field(..., ge=0, le=120, description="Torque in Nm")
    tool_wear: float = Field(..., ge=0, le=300, description="Tool wear in minutes")

@app.get("/health")
def health():
    return {"status": "healthy", "model": metadata["model_name"], "version": app.version}

@app.get("/model-info")
def model_info():
    return {"model_name": metadata["model_name"], "operating_threshold": metadata["threshold"],
            "features": metadata["features"], "target": metadata["target"]}

@app.post("/predict")
def predict(payload: SensorInput):
    raw = pd.DataFrame([{
        "UDI": 0, "Product ID": payload.machine_id, "Type": payload.product_type,
        "Air temperature [K]": payload.air_temperature,
        "Process temperature [K]": payload.process_temperature,
        "Rotational speed [rpm]": payload.rotational_speed,
        "Torque [Nm]": payload.torque,
        "Tool wear [min]": payload.tool_wear,
        "Machine failure": 0, "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    }])
    features = make_features(raw)
    probability = float(model.predict_proba(features)[0, 1])
    threshold = float(metadata["threshold"])
    prediction = int(probability >= threshold)
    rec = recommend_maintenance(probability, {
        "air_temperature": payload.air_temperature,
        "process_temperature": payload.process_temperature,
        "rotational_speed": payload.rotational_speed,
        "torque": payload.torque,
        "tool_wear": payload.tool_wear,
    })
    return {
        "machine_id": payload.machine_id,
        "failure_probability": round(probability, 4),
        "failure_probability_percent": round(probability * 100, 2),
        "prediction": "Potential failure" if prediction else "Normal operation",
        "risk_level": rec["risk_level"],
        "recommended_action": rec["recommended_action"],
        "inspection_items": rec["inspection_items"],
        "decision_threshold": threshold,
    }
