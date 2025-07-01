import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(input_path):
    try:
        data = pd.read_csv(input_path)
        logger.info(f"Loaded data with columns: {data.columns.tolist()}")
    except FileNotFoundError:
        logger.error(f"File {input_path} not found")
        return

    X = data.drop(["is_high_risk", "CustomerId"], axis=1)
    y = data["is_high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LogisticRegression": (LogisticRegression(), {"C": [0.1, 1, 10]}),
        "GradientBoosting": (
            GradientBoostingClassifier(),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        ),
    }

    for name, (model, params) in models.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"Training {name}")
            grid = GridSearchCV(model, params, cv=5, scoring="roc_auc")
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred),
            }

            mlflow.log_params(grid.best_params_)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            mlflow.sklearn.log_model(grid.best_estimator_, "model")
            logger.info(f"Logged metrics for {name}: {metrics}")

    return grid.best_estimator_


if __name__ == "__main__":
    input_path = "data/processed/processed_data.csv"
    train_model(input_path)
