from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import os

app = FastAPI(title="Engine Size Predictor")

# Load model from MLflow Registry
MODEL_URI = os.getenv("MODEL_URI", "models:/Linear_Regression_Model/1")
model = mlflow.pyfunc.load_model(MODEL_URI)

class InputRow(BaseModel):
    ENGINESIZE: float

class InputData(BaseModel):
    inputs: list[InputRow]

@app.post("/predict-co2-emission")
def predict(data: InputData):
    try:
        df = pd.DataFrame([r.model_dump() for r in data.inputs])
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
