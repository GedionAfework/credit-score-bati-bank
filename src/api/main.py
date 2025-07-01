from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, PredictionResponse
import mlflow.sklearn
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
model = mlflow.sklearn.load_model("models:/best_model/1")


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    logger.info("Received prediction request")
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[:, 1][0]
    credit_score = 300 + (1 - prob) * 550
    loan_amount = min(max(1000 * (1 - prob), 100), 5000)
    loan_duration = int(min(max(6 * (1 - prob), 3), 24))
    logger.info(
        f"Predicted risk probability: {prob}, Credit Score: {credit_score}, Loan Amount: {loan_amount}, Duration: {loan_duration}"
    )
    return {
        "risk_probability": prob,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_duration": loan_duration,
    }
