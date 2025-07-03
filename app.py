import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    TV: float
    Radio: float
    Newspaper: float

@app.get("/")
def read_root():
    return {"message": "ML Model API is up!"}

@app.post("/predict/")
def predict(data: InputData):
    features = np.array([[data.TV, data.Radio, data.Newspaper]])
    prediction = model.predict(features)
    return {"predicted_sales": prediction[0]}
