from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.sklearn import load_model
import json
from typing import Dict


with open("config.json","r") as f:
    config=json.load(f)

features=config["features"]
threshold=config["threshold"]


model = load_model("exported_model")

app=FastAPI()

class InputData(BaseModel):
    inp:Dict[str,float]

@app.get("/")
def welcome():
    return {"message":"API is running"}

@app.post("/predict")
def predict(data:InputData):
    missing=[f for f in features if f not in data.inp]
    if missing:
        return {"missing_columns":missing}
    
    model_input=[data.inp[f] for f in features]
    pred=model.predict_proba([model_input])[0][1]
    prediction="Pass" if pred>=threshold else "Fail"
    pass_prob=pred
    fail_prob=1-pred
    return {
    "prediction": prediction,
    "prob_pass": float(pass_prob),
    "prob_fail": float(fail_prob),
    "threshold":threshold}