import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model():
    try:
        data = pd.read_csv("data/processed/processed_data.csv")
        logger.info(f"Loaded data with columns: {data.columns.tolist()}")
    except FileNotFoundError:
        logger.error("Processed data file not found")
        return

    feature_cols = [
        col for col in data.columns if col not in ["is_high_risk", "CustomerId"]
    ]
    X = data[feature_cols]
    y = data["is_high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger.info(
        f"Model metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}"
    )

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("credit_score_model")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model", registered_model_name="best_model")
    logger.info("Model trained and logged as 'best_model'")


if __name__ == "__main__":
    train_model()
