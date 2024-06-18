from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class PredictionRequest(BaseModel):
	feature_vector: List[float]
	score: Optional[bool] = False

@app.post("/prediction")
async def prediction(request: PredictionRequest):
	prediction = model.predict([request.feature_vector])
	response = {"is_inliner": int(prediction[0])}

	if request.score:
		score = model.score_samples([request.feature_vector])
		response["anomaly_score"] = score[0]

	return response

@app.get("/model_information")
async def model_information():
	return model.get_params()
