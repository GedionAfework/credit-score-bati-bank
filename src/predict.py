import pandas as pd
import mlflow.sklearn
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def predict(input_path):
    try:
        data = pd.read_csv(input_path)
        logger.info(f"Loaded data with columns: {data.columns.tolist()}")
    except FileNotFoundError:
        logger.error(f"File {input_path} not found")
        return

    model = mlflow.sklearn.load_model("models:/best_model/1")
    feature_cols = [
        col for col in data.columns if col not in ["is_high_risk", "CustomerId"]
    ]
    if not feature_cols:
        logger.error("No feature columns available for prediction")
        return
    X = data[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    data["risk_probability"] = probs
    data["credit_score"] = 300 + (1 - probs) * 550
    data["loan_amount"] = np.minimum(np.maximum(1000 * (1 - probs), 100), 5000)
    data["loan_duration"] = np.minimum(np.maximum(6 * (1 - probs), 3), 24).astype(int)
    logger.info(
        f"Predictions completed: {data[['CustomerId', 'risk_probability', 'credit_score', 'loan_amount', 'loan_duration']].head()}"
    )
    output_path = input_path.replace(".csv", "_predictions.csv")
    data.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    input_path = "data/processed/processed_data.csv"
    predict(input_path)
