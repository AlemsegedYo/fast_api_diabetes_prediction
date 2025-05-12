# routers/predict_router.py

from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
# from train_model import preprocess_data

router = APIRouter()

model = joblib.load('model/diabetes_pipeline_model.pkl')

class InputData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@router.post("/")
async def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict()])
        # input_df['Outcome'] = 0
        # processed, dummy_y = preprocess_data(input_df)
        # scaled = scaler.transform(processed)
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction), "result": "Diabetic" if prediction == 1 else "Not Diabetic"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
