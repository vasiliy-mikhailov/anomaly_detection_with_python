from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel
import joblib
import prometheus_client

app = FastAPI()
metrics_app = prometheus_client.make_asgi_app()
app.mount("/metrics", metrics_app)

model = joblib.load("model.joblib")

predictions_counter = prometheus_client.Counter('predictions_counter', 'Number of predictions')
predictions_output = prometheus_client.Histogram('predictions_output', 'Prediction output')
predictions_score = prometheus_client.Histogram('predictions_score', 'Predictions score')
predictions_latency = prometheus_client.Histogram('predictions_latency', 'Predictions latency')

class PredictionRequest(BaseModel):
	feature_vector: List[float]
	score: Optional[bool] = False

@predictions_latency.time()
@app.post("/prediction")
async def prediction(request: PredictionRequest):
	prediction = model.predict([request.feature_vector])
	response = {"is_inliner": int(prediction[0])}
	predictions_output.observe(int(prediction[0]))

	if request.score:
		score = model.score_samples([request.feature_vector])
		response["anomaly_score"] = score[0]
		predictions_score.observe(score[0])

	predictions_counter.inc()

	return response

@app.get("/model_information")
async def model_information():
	return model.get_params()
